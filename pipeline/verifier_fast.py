"""
FAST Verifier module for KDSH Track A pipeline.
Uses Groq API for fast, free LLM inference.

Key optimizations:
1. Groq API - Free, fast cloud inference (~1s per call vs ~120s local)
2. NO LLM-based reranking (uses score-based fusion instead)
3. Simplified prompts for reliable parsing
4. Citation-based verification
5. Parallel claim verification
"""

import re
import json
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


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
    bm25_weight: float = 0.4
    vector_weight: float = 0.6
    
    # Verification settings
    max_claims: int = 5
    
    # Retry settings
    max_retries: int = 3


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
                
            except Exception as e:
                self.errors += 1
                if attempt < self.config.max_retries:
                    time.sleep(1)
                else:
                    print(f"Groq API error after {self.config.max_retries + 1} attempts: {e}")
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
    
    SYSTEM_PROMPT = """You extract SPECIFIC VERIFIABLE FACTS from character backstories.

Focus on facts that can be CHECKED against the book:
- Specific DATES, YEARS, or TIME PERIODS
- NAMES of people, places, ships, institutions
- RELATIONSHIPS (who is related to whom, how)
- SPECIFIC EVENTS (what happened, in what order)
- LOCATIONS where events occurred
- NUMBERS, QUANTITIES, DURATIONS

DO NOT extract:
- Vague emotional states
- General character descriptions
- Opinions or interpretations
- Meta-information about the novel itself"""

    USER_PROMPT = """Extract {max_claims} SPECIFIC VERIFIABLE FACTS about {character} from this backstory.

BACKSTORY:
{backstory}

For each fact, focus on:
- Dates/years mentioned
- Names of other people
- Specific events and their outcomes
- Locations and places
- Relationships between characters

Return exactly {max_claims} numbered facts. Each fact should be ONE specific, checkable claim:
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
# Hybrid Retriever with RRF
# ============================================================================

class FastHybridRetriever:
    """Fast hybrid retrieval with Reciprocal Rank Fusion."""
    
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
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Hybrid search using RRF."""
        k = top_k or self.config.top_k_retrieval
        rrf_k = 60
        
        bm25_results = self.bm25.search(query, top_k=k * 2)
        vector_results = self.embedder.search(query, top_k=k * 2)
        
        bm25_ranks = {idx: rank for rank, (idx, _) in enumerate(bm25_results)}
        
        vector_ranks = {}
        for rank, r in enumerate(vector_results):
            chunk_id = r.get('chunk_id')
            if chunk_id in self._chunk_id_to_idx:
                vector_ranks[self._chunk_id_to_idx[chunk_id]] = rank
        
        all_indices = set(bm25_ranks.keys()) | set(vector_ranks.keys())
        rrf_scores = {}
        
        for idx in all_indices:
            score = 0.0
            if idx in bm25_ranks:
                score += self.config.bm25_weight / (rrf_k + bm25_ranks[idx])
            if idx in vector_ranks:
                score += self.config.vector_weight / (rrf_k + vector_ranks[idx])
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
    
    SYSTEM_PROMPT = """You are a fact-checker verifying claims against book passages.

Your task: Determine if the CLAIM is SUPPORTED, CONTRADICTED, or UNCLEAR based on the EVIDENCE.

CRITICAL RULES FOR DETECTING CONTRADICTIONS:
1. DATES/TIMES: If claim says "1815" but evidence shows "1811" → CONTRADICTION
2. NAMES: If claim says "brother" but evidence shows "cousin" → CONTRADICTION  
3. EVENTS: If claim says "escaped" but evidence shows "captured" → CONTRADICTION
4. LOCATIONS: If claim says "Paris" but evidence shows "London" → CONTRADICTION
5. RELATIONSHIPS: If claim says "father" but evidence shows "uncle" → CONTRADICTION

CRITICAL RULES FOR SUPPORT:
- Evidence must EXPLICITLY state or strongly imply the claim
- Just mentioning the same character is NOT support

UNCLEAR means:
- Evidence doesn't address this specific claim
- Information is ambiguous or incomplete

OUTPUT FORMAT (you MUST follow this exactly):
VERDICT: [SUPPORTS/CONTRADICTS/UNCLEAR]
CONFIDENCE: [0.0-1.0]
CITATION: [Quote the specific passage that supports your verdict]
REASONING: [Explain why in 1-2 sentences]"""

    USER_PROMPT = """CLAIM TO VERIFY:
"{claim}"

EVIDENCE PASSAGES FROM THE BOOK:
{evidence}

Analyze the evidence carefully. Look for:
- Any FACTUAL CONFLICTS with the claim (different dates, names, events, relationships)
- Any passages that CONFIRM the claim
- Whether the evidence actually addresses the claim at all

VERDICT: [SUPPORTS/CONTRADICTS/UNCLEAR]
CONFIDENCE: [0.0-1.0]
CITATION: [Quote the relevant passage]
REASONING: [Why?]"""

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
        
        # Format evidence with passage numbers for citation
        evidence_parts = []
        for i, e in enumerate(evidence[:self.config.top_k_final]):
            content = e.get(content_key, '')[:800]
            chapter = e.get('chapter', 'Unknown')
            evidence_parts.append(f"[PASSAGE {i+1}] ({chapter}):\n{content}")
        
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
            evidence=evidence_text
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
# Aggregator with improved logic
# ============================================================================

class FastAggregator:
    """Aggregate verdicts into final prediction."""
    
    def aggregate(self, results: List[Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
        """
        Aggregation rules:
        1. Any high-confidence contradiction (>=0.6) → CONTRADICT
        2. Multiple contradictions (>=2) → CONTRADICT
        3. Strong contradiction score vs support score → CONTRADICT
        4. Otherwise → CONSISTENT
        """
        if not results:
            return 1, {'reason': 'No claims extracted', 'verdicts': []}
        
        # Calculate scores
        contradiction_score = 0.0
        support_score = 0.0
        
        for r in results:
            conf = r.get('confidence', 0.5)
            if r['verdict'] == 'contradicts':
                contradiction_score += conf
            elif r['verdict'] == 'supports':
                support_score += conf
        
        # Count verdicts
        verdicts = [r['verdict'] for r in results]
        n_contradicts = verdicts.count('contradicts')
        n_supports = verdicts.count('supports')
        n_unclear = verdicts.count('unclear')
        
        # Check for high-confidence contradictions
        high_conf_contradict = any(
            r['verdict'] == 'contradicts' and r.get('confidence', 0) >= 0.6
            for r in results
        )
        
        # Check for contradictions with citations (more reliable)
        cited_contradict = any(
            r['verdict'] == 'contradicts' and r.get('citation') and len(r.get('citation', '')) > 20
            for r in results
        )
        
        # Decision logic
        if high_conf_contradict or cited_contradict:
            prediction = 0
            reason = f"High-confidence contradiction found"
        elif n_contradicts >= 2:
            prediction = 0
            reason = f"Multiple contradictions ({n_contradicts})"
        elif contradiction_score > support_score * 1.2 and n_contradicts >= 1:
            prediction = 0
            reason = f"Contradiction score ({contradiction_score:.2f}) > support ({support_score:.2f})"
        else:
            prediction = 1
            reason = f"Consistent ({n_supports} supports, {n_unclear} unclear, {n_contradicts} contradicts)"
        
        return prediction, {
            'reason': reason,
            'verdicts': results,
            'counts': {'contradicts': n_contradicts, 'supports': n_supports, 'unclear': n_unclear},
            'scores': {'contradiction': contradiction_score, 'support': support_score}
        }


# ============================================================================
# Verification Pipeline
# ============================================================================

class FastVerificationPipeline:
    """Optimized verification pipeline with parallel processing."""
    
    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        embedder,
        config: Optional[FastVerifierConfig] = None
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
        print("Pipeline ready!")
    
    def verify_backstory(
        self,
        backstory: str,
        character: str,
        book_name: str,
        verbose: bool = True
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
            return 1, {
                'reason': 'No claims extracted', 
                'elapsed': time.time() - start_time,
                'claims': []
            }
        
        # 2. Retrieve evidence for all claims
        if verbose:
            print("\n2. Retrieving evidence...")
        claim_evidence = [(claim, self.retriever.search(claim)) for claim in claims]
        
        # 3. Verify claims in parallel
        if verbose:
            print("3. Verifying claims...")
        
        def verify_single(args):
            idx, claim, evidence = args
            return idx, self.verifier.verify(claim, evidence, character, book_name)
        
        results = [None] * len(claims)
        with ThreadPoolExecutor(max_workers=min(5, len(claims))) as executor:
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
        results = [r for r in results if r is not None]
        
        # 4. Aggregate
        prediction, details = self.aggregator.aggregate(results)
        
        elapsed = time.time() - start_time
        details['elapsed'] = elapsed
        details['claims'] = claims
        details['llm_stats'] = self.llm.get_stats()
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"PREDICTION: {prediction} ({'CONSISTENT' if prediction == 1 else 'CONTRADICT'})")
            print(f"Reason: {details['reason']}")
            print(f"Time: {elapsed:.1f}s | LLM calls: {self.llm.call_count}")
        
        return prediction, details


# ============================================================================
# Evaluator
# ============================================================================

class FastEvaluator:
    """Evaluate pipeline on dataset."""
    
    def __init__(self, pipeline: FastVerificationPipeline):
        self.pipeline = pipeline
        self.results = []
    
    def evaluate(
        self,
        samples: List[Dict[str, Any]],
        max_samples: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Run evaluation."""
        if max_samples:
            samples = samples[:max_samples]
        
        print(f"\n{'='*60}")
        print(f"KDSH Track A - Evaluating {len(samples)} samples")
        print(f"{'='*60}")
        
        label_map = {'consistent': 1, 'contradict': 0}
        
        for i, sample in enumerate(samples):
            print(f"\n[{i+1}/{len(samples)}] Sample {sample['id']} - {sample['char']}")
            
            true_label = label_map.get(sample['label'], -1)
            
            try:
                prediction, details = self.pipeline.verify_backstory(
                    backstory=sample['content'],
                    character=sample['char'],
                    book_name=sample['book_name'],
                    verbose=verbose
                )
            except Exception as e:
                print(f"Error processing sample: {e}")
                prediction = 1
                details = {'error': str(e)}
            
            correct = prediction == true_label
            status = "✓" if correct else "✗"
            
            self.results.append({
                'id': sample['id'],
                'character': sample['char'],
                'book': sample['book_name'],
                'true_label': sample['label'],
                'prediction': 'consistent' if prediction == 1 else 'contradict',
                'correct': correct,
                'details': details
            })
            
            print(f"\n{status} Result: True={sample['label']}, Pred={'consistent' if prediction == 1 else 'contradict'}")
        
        # Summary
        correct = sum(1 for r in self.results if r['correct'])
        accuracy = correct / len(self.results) if self.results else 0
        
        consistent_samples = [r for r in self.results if r['true_label'] == 'consistent']
        contradict_samples = [r for r in self.results if r['true_label'] == 'contradict']
        
        consistent_acc = sum(1 for r in consistent_samples if r['correct']) / len(consistent_samples) if consistent_samples else 0
        contradict_acc = sum(1 for r in contradict_samples if r['correct']) / len(contradict_samples) if contradict_samples else 0
        
        llm_stats = self.pipeline.llm.get_stats()
        
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Overall Accuracy:    {correct}/{len(self.results)} ({accuracy:.1%})")
        print(f"Consistent Accuracy: {consistent_acc:.1%}")
        print(f"Contradict Accuracy: {contradict_acc:.1%}")
        print(f"{'='*60}")
        print(f"LLM calls: {llm_stats['call_count']} | Total time: {llm_stats['total_time']:.1f}s")
        print(f"Avg per call: {llm_stats['avg_time']:.2f}s | Errors: {llm_stats['errors']}")
        print(f"{'='*60}")
        
        return {
            'accuracy': accuracy,
            'consistent_accuracy': consistent_acc,
            'contradict_accuracy': contradict_acc,
            'correct': correct,
            'total': len(self.results),
            'results': self.results,
            'llm_stats': llm_stats
        }


if __name__ == "__main__":
    print("Fast verifier module. Run via: python -m pipeline.run_eval_fast")
