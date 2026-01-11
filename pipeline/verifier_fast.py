"""
FAST Verifier module for KDSH Track A pipeline.
Uses Groq API for fast, free LLM inference.

Key optimizations:
1. Groq API - Free, fast cloud inference (~1s per call vs ~120s local)
2. NO LLM-based reranking (uses score-based fusion instead)
3. Simplified prompts for reliable parsing
4. Citation-based verification
5. Parallel claim verification
6. Detailed result logging to numbered JSON files
"""

import re
import json
import os
import glob
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import time


# ============================================================================
# Result Logger - Saves detailed verification data to numbered files
# ============================================================================

class ResultLogger:
    """
    Logs detailed verification results to numbered JSON files.
    Creates test1/, test2/ folders with test_1.json, test_2.json, etc. inside.
    Each run creates a new folder (test1, test2, etc.)
    """
    
    def __init__(self, output_dir: str = "verification_results"):
        self.base_dir = Path(output_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.run_folder = self._create_run_folder()
        self.output_dir = self.run_folder  # For compatibility
        self.file_counter = 1  # Always start at 1 for new run folder
        
    def _create_run_folder(self) -> Path:
        """Create a new numbered run folder (test1, test2, etc.)."""
        existing_folders = [d for d in self.base_dir.iterdir() if d.is_dir() and d.name.startswith('test')]
        
        # Extract numbers from existing folders
        numbers = []
        for folder in existing_folders:
            match = re.search(r'^test(\d+)$', folder.name)
            if match:
                numbers.append(int(match.group(1)))
        
        next_num = max(numbers) + 1 if numbers else 1
        run_folder = self.base_dir / f"test{next_num}"
        run_folder.mkdir(parents=True, exist_ok=True)
        print(f"   📂 Created run folder: {run_folder}")
        return run_folder
    
    def _get_next_file_number(self) -> int:
        """Find the next available file number (deprecated, kept for compatibility)."""
        return self.file_counter
    
    def log_verification(
        self,
        sample_id: Any,
        character: str,
        book_name: str,
        backstory: str,
        claims: List[str],
        claim_results: List[Dict[str, Any]],
        prediction: int,
        prediction_explanation: str,
        true_label: Optional[str] = None,
        elapsed_time: float = 0.0,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a complete verification result to a new JSON file.
        
        Returns the path to the created file.
        """
        # Build comprehensive result structure
        result = {
            "file_info": {
                "file_number": self.file_counter,
                "created_at": datetime.now().isoformat(),
                "sample_id": sample_id
            },
            "sample": {
                "id": sample_id,
                "character": character,
                "book_name": book_name,
                "backstory": backstory,
                "true_label": true_label
            },
            "verification": {
                "claims_extracted": claims,
                "claim_count": len(claims),
                "claim_results": [
                    {
                        "claim_number": i + 1,
                        "claim_text": claims[i] if i < len(claims) else "N/A",
                        "verdict": r.get("verdict", "unknown"),
                        "confidence": r.get("confidence", 0.0),
                        "reasoning": r.get("reasoning", "No reasoning provided"),
                        "citation": r.get("citation", None),
                        "evidence_used": r.get("evidence_metadata", [])
                    }
                    for i, r in enumerate(claim_results)
                ]
            },
            "final_decision": {
                "prediction": prediction,
                "prediction_label": "consistent" if prediction == 1 else "contradict",
                "explanation": prediction_explanation,
                "is_correct": (true_label == ("consistent" if prediction == 1 else "contradict")) if true_label else None
            },
            "statistics": {
                "elapsed_time_seconds": elapsed_time,
                "verdicts_summary": {
                    "supports": sum(1 for r in claim_results if r.get("verdict") == "supports"),
                    "contradicts": sum(1 for r in claim_results if r.get("verdict") == "contradicts"),
                    "unclear": sum(1 for r in claim_results if r.get("verdict") == "unclear")
                },
                "avg_confidence": sum(r.get("confidence", 0) for r in claim_results) / max(1, len(claim_results)),
                "citations_found": sum(1 for r in claim_results if r.get("citation"))
            }
        }
        
        # Add any additional metadata
        if additional_metadata:
            result["additional_metadata"] = additional_metadata
        
        # Write to file in the run folder
        filename = f"test_{self.file_counter}.json"
        filepath = self.run_folder / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"   📁 Saved detailed results to: {filepath}")
        
        self.file_counter += 1
        return str(filepath)
    
    def get_all_results(self) -> List[Dict[str, Any]]:
        """Load and return all saved results from current run folder."""
        results = []
        for filepath in sorted(self.run_folder.glob("test_*.json")):
            with open(filepath, 'r', encoding='utf-8') as f:
                results.append(json.load(f))
        return results
    
    def get_run_folder(self) -> Path:
        """Return the current run folder path."""
        return self.run_folder


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class FastVerifierConfig:
    """Configuration for the fast verification pipeline."""
    # LLM settings (Groq API)
    llm_model: str = "llama-3.1-8b-instant"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 800
    
    # Retrieval settings
    top_k_retrieval: int = 15
    top_k_final: int = 8  # More evidence chunks
    bm25_weight: float = 0.5  # Increased for better exact term matching
    vector_weight: float = 0.5
    
    # Verification settings
    max_claims: int = 5
    
    # Retry settings (premium tier - higher limits)
    max_retries: int = 3
    parallel_workers: int = 10  # Premium tier allows more concurrent requests


DEFAULT_FAST_CONFIG = FastVerifierConfig()


# ============================================================================
# Groq LLM Wrapper
# ============================================================================

class GroqLLM:
    """LLM wrapper using Groq API."""
    
    def __init__(self, config: Optional[FastVerifierConfig] = None):
        self.config = config or DEFAULT_FAST_CONFIG
        self._client = None
        self.call_count = 0
        self.total_time = 0.0
        self.errors = 0
    
    def _get_client(self):
        if self._client is not None:
            return self._client
        
        from groq import Groq
        
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable not set!\n"
                "Get your free API key at: https://console.groq.com/keys\n"
                "Then set it: export GROQ_API_KEY='your-key-here'"
            )
        
        self._client = Groq(api_key=api_key)
        print(f"Groq API initialized (model: {self.config.llm_model})")
        return self._client
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None, system_prompt: Optional[str] = None) -> str:
        """Generate text using Groq API with retry logic."""
        client = self._get_client()
        
        start = time.time()
        self.call_count += 1
        
        max_tok = max_tokens or self.config.llm_max_tokens
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        result = "ERROR"
        
        for attempt in range(self.config.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=messages,
                    temperature=self.config.llm_temperature,
                    max_tokens=max_tok,
                )
                
                content = response.choices[0].message.content
                result = content.strip() if content else "UNCLEAR"
                break
                
            except Exception:
                self.errors += 1
                if attempt < self.config.max_retries:
                    time.sleep(0.5 * (attempt + 1))  # Short backoff for premium
                else:
                    result = "ERROR"
        
        elapsed = time.time() - start
        self.total_time += elapsed
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'call_count': self.call_count,
            'total_time': self.total_time,
            'avg_time': self.total_time / max(1, self.call_count),
            'errors': self.errors
        }


# ============================================================================
# Claim Extractor - Focus on verifiable facts
# ============================================================================

class FastClaimExtractor:
    """Extract specific, verifiable claims from backstories."""
    
    SYSTEM_PROMPT = """You extract SPECIFIC VERIFIABLE FACTS from character backstories that can be checked against the book.

Extract facts about:
- DATES, YEARS, TIME PERIODS (e.g., "X happened in 1815", "X spent 10 years doing Y")
- NAMES of people mentioned (e.g., "X's father was named Y")
- RELATIONSHIPS (e.g., "X is the son of Y", "X married Y")
- SPECIFIC EVENTS (e.g., "X was arrested", "X killed Y", "X traveled to Z")
- LOCATIONS (e.g., "X was born in Paris", "X was imprisoned at the Château d'If")

IMPORTANT:
- Each claim should be a FACTUAL STATEMENT that the backstory is making
- Do NOT make meta-comments like "no date is mentioned" 
- Do NOT extract claims about what is NOT in the backstory
- Extract what the backstory CLAIMS happened to the character"""

    USER_PROMPT = """Extract {max_claims} FACTUAL CLAIMS about {character} from this backstory.

CHARACTER: {character}
BACKSTORY: "{backstory}"

Extract specific factual claims that the backstory makes about {character}. Focus on:
- Any years or dates mentioned
- Names of other people involved  
- What events happened to {character}
- Where events took place
- Relationships mentioned

Each claim should be a statement that CAN BE TRUE OR FALSE based on the book.

Return exactly {max_claims} numbered factual claims:
1.
2.
3.
4.
5."""

    def __init__(self, llm: GroqLLM, config: Optional[FastVerifierConfig] = None):
        self.llm = llm
        self.config = config or DEFAULT_FAST_CONFIG
    
    def extract(self, backstory: str, character: str, book_name: str) -> List[str]:
        """Extract claims from backstory."""
        prompt = self.USER_PROMPT.format(
            character=character,
            backstory=backstory[:3000],
            max_claims=self.config.max_claims
        )
        
        response = self.llm.generate(
            prompt, 
            max_tokens=500,
            system_prompt=self.SYSTEM_PROMPT
        )
        
        # Parse numbered claims
        claims = []
        for line in response.split('\n'):
            line = line.strip()
            # Match various numbering formats
            match = re.match(r'^[\d]+[.\):\-]\s*(.+)', line)
            if match:
                claim = match.group(1).strip()
                # Filter out empty or meta claims
                if len(claim) > 20 and not any(x in claim.lower() for x in 
                    ['the claim', 'this claim', 'claim:', 'fact:', 'novel', 'book', 'author']):
                    claims.append(claim)
        
        return claims[:self.config.max_claims]


# ============================================================================
# BM25 Retriever
# ============================================================================

class FastBM25:
    """Optimized BM25 using rank-bm25 library."""
    
    def __init__(self, chunks: List[Dict[str, Any]], content_key: str = "content"):
        from rank_bm25 import BM25Okapi
        
        self.chunks = chunks
        self.content_key = content_key
        self.corpus = [self._tokenize(c[content_key]) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        tokens = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text.lower())
        return [t for t in tokens if len(t) > 1]
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Return (doc_index, score) pairs."""
        query_terms = self._tokenize(query)
        if not query_terms:
            return []
        
        scores = self.bm25.get_scores(query_terms)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]


# ============================================================================
# Hybrid Retriever with RRF and Query Expansion
# ============================================================================

class FastHybridRetriever:
    """Fast hybrid retrieval with Reciprocal Rank Fusion and query expansion."""
    
    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        embedder,
        config: Optional[FastVerifierConfig] = None,
        content_key: str = "content"
    ):
        self.chunks = chunks
        self.embedder = embedder
        self.config = config or DEFAULT_FAST_CONFIG
        self.content_key = content_key
        self.bm25 = FastBM25(chunks, content_key)
        self._chunk_id_to_idx = {c.get('chunk_id', i): i for i, c in enumerate(chunks)}
    
    def _extract_key_terms(self, query: str, character: Optional[str] = None) -> List[str]:
        """Extract key search terms from a query for multi-query retrieval."""
        queries = [query]  # Always include original
        
        # Add character name as separate query if provided
        if character:
            queries.append(character)
            # Add character + key action/attribute patterns
            born_match = re.search(r'born\s+(?:in|at)\s+(\w+)', query, re.I)
            if born_match:
                queries.append(f"{character} born")
            died_match = re.search(r'died\s+(?:in|at|on)', query, re.I)
            if died_match:
                queries.append(f"{character} died death")
        
        # Extract quoted phrases
        quoted = re.findall(r'"([^"]+)"', query)
        queries.extend(quoted)
        
        # Extract possessive relationships (e.g., "Noirtier's mother")
        possessive = re.findall(r"(\w+)'s\s+(\w+)", query)
        for owner, owned in possessive:
            queries.append(f"{owner} {owned}")
            queries.append(f"{owned} of {owner}")
        
        # Extract key entities (capitalized words)
        caps = re.findall(r'\b([A-Z][a-z]+)\b', query)
        for cap in caps:
            if len(cap) > 2 and cap.lower() not in ['the', 'and', 'was', 'his', 'her']:
                queries.append(cap)
        
        # Extract key biographical terms with character
        if character:
            bio_patterns = ['arrested', 'imprisoned', 'escaped', 'born', 'died', 'married', 'father', 'mother']
            query_lower = query.lower()
            for pattern in bio_patterns:
                if pattern in query_lower:
                    queries.append(f"{character} {pattern}")
        
        return list(set(queries))[:6]  # Limit to 6 queries

    def search(self, query: str, top_k: Optional[int] = None, character: Optional[str] = None) -> List[Dict[str, Any]]:
        """Hybrid search using RRF with optional query expansion."""
        k = top_k or self.config.top_k_retrieval
        rrf_k = 60
        
        # Get multiple query variants
        queries = self._extract_key_terms(query, character)
        
        # Aggregate results from all queries
        all_bm25_ranks = {}
        all_vector_ranks = {}
        
        for q_idx, q in enumerate(queries):
            weight = 1.0 if q_idx == 0 else 0.5  # Primary query gets more weight
            
            bm25_results = self.bm25.search(q, top_k=k * 2)
            for rank, (idx, _) in enumerate(bm25_results):
                # Lower rank (earlier) is better, so we want min
                if idx not in all_bm25_ranks:
                    all_bm25_ranks[idx] = rank * weight
                else:
                    all_bm25_ranks[idx] = min(all_bm25_ranks[idx], rank * weight)
            
            vector_results = self.embedder.search(q, top_k=k * 2)
            for rank, r in enumerate(vector_results):
                chunk_id = r.get('chunk_id')
                if chunk_id in self._chunk_id_to_idx:
                    idx = self._chunk_id_to_idx[chunk_id]
                    if idx not in all_vector_ranks:
                        all_vector_ranks[idx] = rank * weight
                    else:
                        all_vector_ranks[idx] = min(all_vector_ranks[idx], rank * weight)
        
        # RRF fusion
        all_indices = set(all_bm25_ranks.keys()) | set(all_vector_ranks.keys())
        rrf_scores = {}
        
        for idx in all_indices:
            score = 0.0
            if idx in all_bm25_ranks:
                score += self.config.bm25_weight / (rrf_k + all_bm25_ranks[idx])
            if idx in all_vector_ranks:
                score += self.config.vector_weight / (rrf_k + all_vector_ranks[idx])
            rrf_scores[idx] = score
        
        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        results = []
        for idx in sorted_indices[:k]:
            chunk = self.chunks[idx].copy()
            chunk['rrf_score'] = rrf_scores[idx]
            results.append(chunk)
        
        return results


# ============================================================================
# Claim Verifier with Citation Checking
# ============================================================================

class FastClaimVerifier:
    """Verify claims with citation-based evidence checking."""
    SYSTEM_PROMPT="""You verify claims about fictional characters against book evidence.

VERDICTS:
- SUPPORTS: Evidence confirms the claim is true
- CONTRADICTS: Evidence shows conflicting information
- UNCLEAR: Evidence doesn't address this topic AT ALL

VERIFICATION PRIORITY:
1. First, look for CONTRADICTIONS - any factual conflict with the evidence
2. If no contradiction exists, actively search for SUPPORTING statements
3. Only mark UNCLEAR if evidence truly doesn't address the topic

KEY RULES FOR CONTRADICTS:
1. Different dates = CONTRADICTS (claim says "1815", evidence says "1811" → CONTRADICTS)
2. Different facts = CONTRADICTS (claim says "royalist", evidence says "Jacobin" → CONTRADICTS)
3. Opposite statements = CONTRADICTS (claim says "escaped", evidence says "died there" → CONTRADICTS)

KEY RULES FOR SUPPORTS:
1. Evidence confirms the same fact stated in the claim → SUPPORTS
2. Evidence describes actions/events matching the claim → SUPPORTS
3. Character traits or roles confirmed by evidence → SUPPORTS

KEY RULES FOR UNCLEAR:
1. Topic not mentioned at all in evidence → UNCLEAR
2. Fabricated details (names/places not in the book) → UNCLEAR
3. Evidence is about different character/topic entirely → UNCLEAR

IMPORTANT: If the claim states a specific fact (date, name, event) and evidence shows a DIFFERENT specific fact about the SAME topic, that is CONTRADICTS - NOT unclear!

Example 1: Claim "arrested in 1815" vs evidence "arrested in 1811" → CONTRADICTS (different dates for same event)
Example 2: Claim "rescued Yurook" but evidence never mentions Yurook → UNCLEAR (fabricated name)
Example 3: Claim "father was royalist" vs evidence "father was Jacobin" → CONTRADICTS (opposite politics)

Output format:
VERDICT: [SUPPORTS/CONTRADICTS/UNCLEAR]
CONFIDENCE: [0.0-1.0]
CITATION: ["exact quote" or "NONE"]
REASONING: [brief explanation]"""

    USER_PROMPT = """CLAIM about {character}: "{claim}"

EVIDENCE FROM THE BOOK:
{evidence}

Analyze: Does the evidence SUPPORT, CONTRADICT, or is UNCLEAR?
If evidence shows DIFFERENT facts about the SAME topic → CONTRADICTS.
If evidence doesn't mention the topic at all → UNCLEAR."""

    def __init__(self, llm: GroqLLM, config: Optional[FastVerifierConfig] = None):
        self.llm = llm
        self.config = config or DEFAULT_FAST_CONFIG
    
    def verify(
        self,
        claim: str,
        evidence: List[Dict[str, Any]],
        character: str,
        book_name: str,
        content_key: str = "content"
    ) -> Dict[str, Any]:
        """Verify a claim against evidence with citation checking."""
        
        evidence_parts = []
        for i, e in enumerate(evidence[:self.config.top_k_final]):
            content = e.get(content_key, '')[:1000]
            chapter = e.get('chapter', 'Unknown')
            page = e.get('page', '?')
            
            evidence_parts.append(f"[PASSAGE {i+1}] Chapter: {chapter}, Page: {page}:\n{content}")
        
        evidence_text = "\n\n".join(evidence_parts)
        
        if not evidence_text.strip():
            return {
                'claim': claim,
                'verdict': 'unclear',
                'confidence': 0.3,
                'reasoning': 'No evidence found',
                'citation': None
            }
        
        prompt = self.USER_PROMPT.format(
            claim=claim,
            evidence=evidence_text,
            character=character
        )
        
        response = self.llm.generate(
            prompt,
            max_tokens=400,
            system_prompt=self.SYSTEM_PROMPT
        )
        
        # Parse the structured response
        result = self._parse_response(response)
        result['claim'] = claim
        
        return result
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response with multiple fallback strategies."""
        
        verdict = 'unclear'
        confidence = 0.5
        reasoning = ''
        citation = None
        
        # Strategy 1: Look for explicit VERDICT line
        verdict_match = re.search(r'VERDICT:\s*(SUPPORTS?|CONTRADICTS?|UNCLEAR)', response, re.IGNORECASE)
        if verdict_match:
            v = verdict_match.group(1).upper()
            if 'CONTRADICT' in v:
                verdict = 'contradicts'
            elif 'SUPPORT' in v:
                verdict = 'supports'
            else:
                verdict = 'unclear'
        
        # Strategy 2: Look for CONFIDENCE
        conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response, re.IGNORECASE)
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                pass
        
        # Strategy 3: Extract CITATION
        citation_match = re.search(r'CITATION:\s*["\']?(.+?)["\']?\s*(?:REASONING|$)', response, re.IGNORECASE | re.DOTALL)
        if citation_match:
            citation = citation_match.group(1).strip()[:300]
        
        # Strategy 4: Extract REASONING
        reasoning_match = re.search(r'REASONING:\s*(.+?)(?:$|\n\n)', response, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()[:300]
        
        # Fallback: Keyword-based detection if no explicit verdict
        if not verdict_match:
            response_lower = response.lower()
            
            # Strong contradiction indicators
            contradiction_phrases = [
                'contradict', 'conflict', 'inconsistent', 'different from',
                'does not match', 'incorrect', 'wrong', 'false', 'inaccurate',
                'not true', 'differs from', 'opposite', 'mismatch'
            ]
            support_phrases = [
                'support', 'confirm', 'consistent', 'matches', 'correct',
                'true', 'accurate', 'aligns', 'agrees with', 'verified'
            ]
            
            contradiction_count = sum(1 for p in contradiction_phrases if p in response_lower)
            support_count = sum(1 for p in support_phrases if p in response_lower)
            
            # Check for negations
            if 'not contradict' in response_lower or 'no contradiction' in response_lower:
                contradiction_count = 0
            if 'not support' in response_lower or 'no support' in response_lower:
                support_count = 0
            
            if contradiction_count > support_count and contradiction_count > 0:
                verdict = 'contradicts'
                confidence = 0.6 + (contradiction_count * 0.05)
            elif support_count > contradiction_count and support_count > 0:
                verdict = 'supports'
                confidence = 0.6 + (support_count * 0.05)
            
            confidence = min(0.9, confidence)
        
        # Boost confidence if citation was provided
        if citation and len(citation) > 20:
            confidence = min(1.0, confidence + 0.1)
        
        if not reasoning:
            reasoning = response[:200]
        
        return {
            'verdict': verdict,
            'confidence': confidence,
            'reasoning': reasoning,
            'citation': citation
        }


# ============================================================================
# Aggregator - One Error = Contradict
# ============================================================================

class FastAggregator:
    """Aggregate claim verdicts into final prediction."""
    
    # Phrases that indicate false positive contradictions
    FALSE_POSITIVE_PHRASES = [
        'same person', 'same individual', 'are the same',
        'does not directly', 'does not explicitly',
        'there is no mention of',
        'evidence does not mention',
        'not that he',  # Pattern: "not that he did X" (comparing unrelated events)
        'not that she',
        'no mention of',  # LLM says "no mention of X = contradict"
        'goal is to',  # Pattern: "his goal is to X" (comparing unrelated events)
        'his goal is',
        'not to rescue',  # Direct pattern from false positive
    ]
    
    # Phrases that indicate invalid citations (citation starts with these)
    BAD_CITATION_PHRASES = ['none', 'no evidence', 'no mention', 'not mentioned', 'does not address']
    
    def _is_strong_contradiction(self, r: Dict[str, Any]) -> bool:
        """Check if result is a strong, well-cited contradiction."""
        if r['verdict'] != 'contradicts' or r.get('confidence', 0) < 0.75:
            return False
        
        citation = r.get('citation', '')
        if not citation or len(citation) < 20:
            return False
        
        # Filter invalid citations
        if any(p in citation.lower()[:60] for p in self.BAD_CITATION_PHRASES):
            return False
        
        # Filter false positive reasoning
        reasoning = r.get('reasoning', '').lower()
        if any(p in reasoning for p in self.FALSE_POSITIVE_PHRASES):
            return False
        
        return True
    
    def aggregate(self, results: List[Dict[str, Any]], claims: Optional[List[str]] = None) -> Tuple[int, Dict[str, Any]]:
        """
        Aggregate verdicts: ONE factual error = CONTRADICT.
        
        Logic:
        1. Strong contradiction (high confidence + citation) → CONTRADICT
        2. Any contradiction with confidence >= 0.6 → CONTRADICT
        3. No decent contradictions → CONSISTENT
        """
        if not results:
            return 1, {
                'reason': 'No claims extracted',
                'verdicts': [],
                'explanation': "PREDICTION: CONSISTENT (Default)\nNo verifiable claims found."
            }
        
        # Count verdicts
        n_contradicts = sum(1 for r in results if r['verdict'] == 'contradicts')
        n_supports = sum(1 for r in results if r['verdict'] == 'supports')
        n_unclear = sum(1 for r in results if r['verdict'] == 'unclear')
        
        # Find strong contradiction (early stop)
        early_stop = None
        for i, r in enumerate(results):
            if self._is_strong_contradiction(r):
                claim_text = claims[i] if claims and i < len(claims) else f"Claim {i+1}"
                early_stop = (i, r, claim_text)
                break
        
        # Find any contradiction with decent confidence (0.6+)
        decent_contradiction = None
        for i, r in enumerate(results):
            if r['verdict'] == 'contradicts' and r.get('confidence', 0) >= 0.6:
                claim_text = claims[i] if claims and i < len(claims) else f"Claim {i+1}"
                decent_contradiction = (i, r, claim_text)
                break
        
        # Find weak contradictions (0.5-0.6 confidence) with citations
        weak_contradiction = None
        if not decent_contradiction:
            for i, r in enumerate(results):
                if r['verdict'] == 'contradicts' and 0.5 <= r.get('confidence', 0) < 0.6:
                    citation = r.get('citation', '')
                    # Only accept weak contradictions if they have a real citation
                    if citation and len(citation) > 30 and not any(p in citation.lower()[:50] for p in self.BAD_CITATION_PHRASES):
                        claim_text = claims[i] if claims and i < len(claims) else f"Claim {i+1}"
                        weak_contradiction = (i, r, claim_text)
                        break
        
        # DECISION: One factual error = CONTRADICT
        if early_stop:
            idx, r, claim_text = early_stop
            return 0, self._build_result(
                reason=f"Strong contradiction (claim {idx+1})",
                prediction="CONTRADICT",
                claim_info=(idx, r, claim_text),
                counts=(n_contradicts, n_supports, n_unclear)
            )
        
        if decent_contradiction:
            idx, r, claim_text = decent_contradiction
            return 0, self._build_result(
                reason=f"Contradiction found (claim {idx+1}, conf {r.get('confidence', 0):.2f})",
                prediction="CONTRADICT",
                claim_info=(idx, r, claim_text),
                counts=(n_contradicts, n_supports, n_unclear)
            )
        
        if weak_contradiction:
            idx, r, claim_text = weak_contradiction
            return 0, self._build_result(
                reason=f"Weak contradiction (claim {idx+1}, conf {r.get('confidence', 0):.2f})",
                prediction="CONTRADICT",
                claim_info=(idx, r, claim_text),
                counts=(n_contradicts, n_supports, n_unclear)
            )
        
        # No contradictions → CONSISTENT
        return 1, self._build_result(
            reason=f"Consistent ({n_supports} supports, {n_unclear} unclear)",
            prediction="CONSISTENT",
            counts=(n_contradicts, n_supports, n_unclear)
        )
    
    def _build_result(self, reason: str, prediction: str, counts: tuple,
                      claim_info: Optional[tuple] = None) -> Dict[str, Any]:
        """Build aggregation result dictionary."""
        n_contradicts, n_supports, n_unclear = counts
        
        parts = [f"PREDICTION: {prediction}", f"REASON: {reason}"]
        
        if claim_info:
            idx, r, claim_text = claim_info
            parts.append(f"\nClaim: \"{claim_text[:100]}\"")
            parts.append(f"Confidence: {r.get('confidence', 0):.2f}")
            if r.get('citation'):
                parts.append(f"Citation: \"{r['citation'][:120]}\"")
        else:
            parts.append(f"\nSupports: {n_supports} | Unclear: {n_unclear} | Contradicts: {n_contradicts}")
        
        return {
            'reason': reason,
            'explanation': "\n".join(parts),
            'counts': {'contradicts': n_contradicts, 'supports': n_supports, 'unclear': n_unclear}
        }


# ============================================================================
# Verification Pipeline
# ============================================================================

class FastVerificationPipeline:
    """Optimized verification pipeline with parallel processing and result logging."""
    
    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        embedder,
        config: Optional[FastVerifierConfig] = None,
        output_dir: str = "verification_results"
    ):
        self.config = config or DEFAULT_FAST_CONFIG
        self.chunks = chunks
        self.embedder = embedder
        
        print("Initializing FAST verification pipeline (Groq API)...")
        self.llm = GroqLLM(config)
        self.extractor = FastClaimExtractor(self.llm, config)
        self.retriever = FastHybridRetriever(chunks, embedder, config)
        self.verifier = FastClaimVerifier(self.llm, config)
        self.aggregator = FastAggregator()
        self.result_logger = ResultLogger(output_dir)
        print(f"Pipeline ready! Results will be saved to: {output_dir}/")
    
    def verify_backstory(
        self,
        backstory: str,
        character: str,
        book_name: str,
        sample_id: Any = None,
        true_label: Optional[str] = None,
        verbose: bool = True,
        save_results: bool = True
    ) -> Tuple[int, Dict[str, Any]]:
        """Verify a backstory against the book."""
        start_time = time.time()
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"Verifying: {character} ({book_name})")
        
        # 1. Extract claims
        if verbose:
            print("1. Extracting claims...")
        claims = self.extractor.extract(backstory, character, book_name)
        if verbose:
            print(f"   Found {len(claims)} claims")
            for i, c in enumerate(claims):
                print(f"   [{i+1}] {c[:80]}...")
        
        if not claims:
            elapsed = time.time() - start_time
            explanation = (
                "PREDICTION: CONSISTENT (Default)\n"
                "REASON: No verifiable claims could be extracted from the backstory.\n"
                "EXPLANATION: The backstory may be too vague or contain no specific facts."
            )
            details = {
                'reason': 'No claims extracted', 
                'explanation': explanation,
                'elapsed': elapsed,
                'claims': []
            }
            
            if save_results:
                self.result_logger.log_verification(
                    sample_id=sample_id or "unknown",
                    character=character,
                    book_name=book_name,
                    backstory=backstory,
                    claims=[],
                    claim_results=[],
                    prediction=1,
                    prediction_explanation=explanation,
                    true_label=true_label,
                    elapsed_time=elapsed
                )
            
            return 1, details
        
        # 2. Retrieve evidence for all claims (with character-aware query expansion)
        if verbose:
            print("\n2. Retrieving evidence...")
        claim_evidence = [(claim, self.retriever.search(claim, character=character)) for claim in claims]
        
        # 3. Verify claims in parallel
        if verbose:
            print("3. Verifying claims...")
        
        def verify_single(args):
            idx, claim, evidence = args
            result = self.verifier.verify(claim, evidence, character, book_name)
            # Add evidence metadata for logging
            result['evidence_metadata'] = [
                {
                    'chunk_id': e.get('chunk_id'),
                    'story': e.get('story'),
                    'chapter': e.get('chapter'),
                    'page': e.get('page'),
                    'score': e.get('rrf_score', e.get('score', 0))
                }
                for e in evidence[:self.config.top_k_final]
            ]
            return idx, result
        
        results: List[Optional[Dict[str, Any]]] = [None] * len(claims)
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            futures = {
                executor.submit(verify_single, (i, claim, evidence)): i
                for i, (claim, evidence) in enumerate(claim_evidence)
            }
            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    results[idx] = result
                    if verbose:
                        citation_note = " [CITED]" if result.get('citation') else ""
                        print(f"   Claim {idx+1}: {result['verdict']} (conf: {result['confidence']:.2f}){citation_note}")
                except Exception as e:
                    print(f"   Error verifying claim: {e}")
        
        # Filter out None results (failed verifications)
        valid_results: List[Dict[str, Any]] = [r for r in results if r is not None]
        
        # 4. Aggregate with claims for detailed explanation
        prediction, details = self.aggregator.aggregate(valid_results, claims)
        
        elapsed = time.time() - start_time
        details['elapsed'] = elapsed
        details['claims'] = claims
        details['llm_stats'] = self.llm.get_stats()
        
        # 5. Save detailed results to file
        if save_results:
            self.result_logger.log_verification(
                sample_id=sample_id or "unknown",
                character=character,
                book_name=book_name,
                backstory=backstory,
                claims=claims,
                claim_results=valid_results,
                prediction=prediction,
                prediction_explanation=details.get('explanation', details.get('reason', '')),
                true_label=true_label,
                elapsed_time=elapsed,
                additional_metadata={'llm_stats': self.llm.get_stats()}
            )
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"PREDICTION: {prediction} ({'CONSISTENT' if prediction == 1 else 'CONTRADICT'})")
            print(f"Reason: {details['reason']}")
            if details.get('explanation'):
                print(f"\n--- DETAILED EXPLANATION ---")
                print(details['explanation'])
                print(f"--- END EXPLANATION ---")
            print(f"\nTime: {elapsed:.1f}s | LLM calls: {self.llm.call_count}")
        
        return prediction, details


# ============================================================================
# Evaluator
# ============================================================================

class FastEvaluator:
    """Evaluate pipeline on dataset with detailed result logging."""
    
    def __init__(self, pipeline: FastVerificationPipeline):
        self.pipeline = pipeline
        self.results = []
    
    def evaluate(
        self,
        samples: List[Dict[str, Any]],
        max_samples: Optional[int] = None,
        verbose: bool = True,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Run evaluation with detailed logging to numbered files."""
        if max_samples:
            samples = samples[:max_samples]
        
        print(f"\n{'='*60}")
        print(f"KDSH Track A - Evaluating {len(samples)} samples")
        print(f"Results will be saved to: {self.pipeline.result_logger.output_dir}/")
        print(f"{'='*60}")
        
        label_map = {'consistent': 1, 'contradict': 0}
        
        for i, sample in enumerate(samples):
            print(f"\n[{i+1}/{len(samples)}] Sample {sample['id']} - {sample['char']}")
            
            true_label_num = label_map.get(sample['label'], -1)
            
            try:
                prediction, details = self.pipeline.verify_backstory(
                    backstory=sample['content'],
                    character=sample['char'],
                    book_name=sample['book_name'],
                    sample_id=sample['id'],
                    true_label=sample['label'],
                    verbose=verbose,
                    save_results=save_results
                )
            except Exception as e:
                print(f"Error processing sample: {e}")
                prediction = 1
                details = {'error': str(e), 'explanation': f"Error occurred: {str(e)}"}
            
            correct = prediction == true_label_num
            status = "✓" if correct else "✗"
            
            # Store result with explanation
            result_entry = {
                'id': sample['id'],
                'character': sample['char'],
                'book': sample['book_name'],
                'true_label': sample['label'],
                'prediction': 'consistent' if prediction == 1 else 'contradict',
                'correct': correct,
                'explanation': details.get('explanation', details.get('reason', 'No explanation')),
                'details': details
            }
            self.results.append(result_entry)
            
            # Print result with brief explanation
            pred_label = 'consistent' if prediction == 1 else 'contradict'
            print(f"\n{status} Result: True={sample['label']}, Pred={pred_label}")
            print(f"   Brief: {details.get('reason', 'No reason provided')}")
        
        # Summary
        correct_count = sum(1 for r in self.results if r['correct'])
        accuracy = correct_count / len(self.results) if self.results else 0
        
        consistent_samples = [r for r in self.results if r['true_label'] == 'consistent']
        contradict_samples = [r for r in self.results if r['true_label'] == 'contradict']
        
        consistent_acc = sum(1 for r in consistent_samples if r['correct']) / len(consistent_samples) if consistent_samples else 0
        contradict_acc = sum(1 for r in contradict_samples if r['correct']) / len(contradict_samples) if contradict_samples else 0
        
        llm_stats = self.pipeline.llm.get_stats()
        
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Overall Accuracy:    {correct_count}/{len(self.results)} ({accuracy:.1%})")
        print(f"Consistent Accuracy: {consistent_acc:.1%}")
        print(f"Contradict Accuracy: {contradict_acc:.1%}")
        print(f"{'='*60}")
        print(f"LLM calls: {llm_stats['call_count']} | Total time: {llm_stats['total_time']:.1f}s")
        print(f"Avg per call: {llm_stats['avg_time']:.2f}s | Errors: {llm_stats['errors']}")
        print(f"{'='*60}")
        print(f"📁 Detailed results saved to: {self.pipeline.result_logger.output_dir}/test_*.json")
        print(f"{'='*60}")
        
        return {
            'accuracy': accuracy,
            'consistent_accuracy': consistent_acc,
            'contradict_accuracy': contradict_acc,
            'correct': correct_count,
            'total': len(self.results),
            'results': self.results,
            'llm_stats': llm_stats,
            'output_dir': str(self.pipeline.result_logger.output_dir)
        }


if __name__ == "__main__":
    print("Fast verifier module. Run via: python -m pipeline.run_eval_fast")
