"""
FAST Verifier module for KDSH Track A pipeline.
Uses Groq API for fast, free LLM inference.

Key optimizations:
1. Groq API - Free, fast cloud inference (~1s per call vs ~120s local)
2. NO LLM-based reranking (uses score-based fusion instead)
3. NO separate query generation (uses claim directly)
4. Improved prompt engineering for better accuracy
"""

import re
import json
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
import time


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class FastVerifierConfig:
    """Configuration for the fast verification pipeline."""
    # LLM settings (Groq API)
    llm_model: str = "llama-3.1-8b-instant"  # Groq model name
    llm_temperature: float = 0.0  # Deterministic for consistency
    llm_max_tokens: int = 512
    
    # Retrieval settings
    top_k_retrieval: int = 15     # More chunks for better recall
    top_k_final: int = 5          # Final chunks for verification
    bm25_weight: float = 0.3      # Slightly lower - semantic often better
    vector_weight: float = 0.7    # Higher weight for semantic search
    
    # Verification settings
    max_claims: int = 5           # 5 claims per backstory
    
    # Retry settings
    max_retries: int = 2          # Retry on API errors


DEFAULT_FAST_CONFIG = FastVerifierConfig()


# ============================================================================
# Groq LLM Wrapper (Fast, Free API) - with retry logic
# ============================================================================

class GroqLLM:
    """
    LLM wrapper using Groq API - fast and free.
    
    Get your API key at: https://console.groq.com/keys
    Set it as environment variable: GROQ_API_KEY
    """
    
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
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        result = "ERROR"
        
        # Retry loop
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
                    print(f"Groq API error (attempt {attempt + 1}): {e}, retrying...")
                    time.sleep(1)  # Brief pause before retry
                else:
                    print(f"Groq API error after {self.config.max_retries + 1} attempts: {e}")
                    result = "ERROR"
        
        elapsed = time.time() - start
        self.total_time += elapsed
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get LLM call statistics."""
        return {
            'call_count': self.call_count,
            'total_time': self.total_time,
            'avg_time': self.total_time / max(1, self.call_count),
            'errors': self.errors
        }


# ============================================================================
# Improved Claim Extractor with better prompting
# ============================================================================

class FastClaimExtractor:
    """Extract claims with a single LLM call using structured prompting."""
    
    SYSTEM_PROMPT = """You are extracting verifiable facts about a CHARACTER from their backstory.

ONLY extract facts about the CHARACTER mentioned, including:
- Events in their life
- Their relationships with other characters
- Their actions and decisions
- Dates/years related to their life events
- Places they lived, visited, or were imprisoned

Do NOT extract:
- Facts about the novel itself (publication date, author, etc.)
- General plot summary
- Facts about other characters unless related to this character"""

    USER_PROMPT = """Extract {max_claims} specific facts about {character} from this backstory.

BACKSTORY:
{backstory}

Extract ONLY facts about {character}'s life - their actions, relationships, events, dates, locations.
Do NOT extract meta-facts about the novel itself.

Return exactly {max_claims} numbered facts about {character}:
1. [Fact about {character}]
2. [Another fact about {character}]
...

Facts about {character}:"""

    def __init__(self, llm: GroqLLM, config: Optional[FastVerifierConfig] = None):
        self.llm = llm
        self.config = config or DEFAULT_FAST_CONFIG
    
    def extract(self, backstory: str, character: str, book_name: str) -> List[str]:
        """Extract claims from backstory."""
        prompt = self.USER_PROMPT.format(
            character=character,
            book_name=book_name,
            backstory=backstory[:2500],  # Slightly more context
            max_claims=self.config.max_claims
        )
        
        response = self.llm.generate(
            prompt, 
            max_tokens=600,
            system_prompt=self.SYSTEM_PROMPT
        )
        
        # Parse numbered claims
        claims = []
        for line in response.split('\n'):
            line = line.strip()
            # Match "1. claim" or "1) claim" or "1: claim"
            match = re.match(r'^[\d]+[.\):]\s*(.+)', line)
            if match:
                claim = match.group(1).strip()
                # Filter very short or meta claims
                if len(claim) > 15 and not claim.lower().startswith(('the claim', 'this claim', 'claim:')):
                    claims.append(claim)
        
        return claims[:self.config.max_claims]


# ============================================================================
# BM25 Retriever (using rank-bm25 library for speed)
# ============================================================================

class FastBM25:
    """Optimized BM25 using rank-bm25 library."""
    
    def __init__(self, chunks: List[Dict[str, Any]], content_key: str = "content"):
        from rank_bm25 import BM25Okapi
        
        self.chunks = chunks
        self.content_key = content_key
        
        # Tokenize corpus with improved tokenization
        self.corpus = [self._tokenize(c[content_key]) for c in chunks]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.corpus)
    
    def _tokenize(self, text: str) -> List[str]:
        """Improved tokenization - keeps meaningful tokens."""
        # Lowercase and extract words (including contractions)
        tokens = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text.lower())
        # Filter very short tokens
        return [t for t in tokens if len(t) > 1]
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Return (doc_index, score) pairs."""
        query_terms = self._tokenize(query)
        
        if not query_terms:
            return []
        
        # Get scores for all documents
        scores = self.bm25.get_scores(query_terms)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Return (index, score) pairs, filtering zero scores
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]


# ============================================================================
# Hybrid Retriever (Fast - No LLM) with improved fusion
# ============================================================================

class FastHybridRetriever:
    """Fast hybrid retrieval with Reciprocal Rank Fusion."""
    
    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        embedder,  # ChunkEmbedder
        config: Optional[FastVerifierConfig] = None,
        content_key: str = "content"
    ):
        self.chunks = chunks
        self.embedder = embedder
        self.config = config or DEFAULT_FAST_CONFIG
        self.content_key = content_key
        self.bm25 = FastBM25(chunks, content_key)
        
        # Pre-build chunk index for faster lookup
        self._chunk_id_to_idx = {
            c.get('chunk_id', i): i for i, c in enumerate(chunks)
        }
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Hybrid search using Reciprocal Rank Fusion (RRF)."""
        k = top_k or self.config.top_k_retrieval
        rrf_k = 60  # RRF constant (standard value)
        
        # BM25 search
        bm25_results = self.bm25.search(query, top_k=k * 2)
        
        # Vector search
        vector_results = self.embedder.search(query, top_k=k * 2)
        
        # Build rank maps
        bm25_ranks = {idx: rank for rank, (idx, _) in enumerate(bm25_results)}
        
        vector_ranks = {}
        for rank, r in enumerate(vector_results):
            chunk_id = r.get('chunk_id')
            if chunk_id in self._chunk_id_to_idx:
                idx = self._chunk_id_to_idx[chunk_id]
                vector_ranks[idx] = rank
        
        # Compute RRF scores
        all_indices = set(bm25_ranks.keys()) | set(vector_ranks.keys())
        rrf_scores = {}
        
        for idx in all_indices:
            score = 0.0
            if idx in bm25_ranks:
                score += self.config.bm25_weight / (rrf_k + bm25_ranks[idx])
            if idx in vector_ranks:
                score += self.config.vector_weight / (rrf_k + vector_ranks[idx])
            rrf_scores[idx] = score
        
        # Sort by RRF score
        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Build results
        results = []
        for idx in sorted_indices[:k]:
            chunk = self.chunks[idx].copy()
            chunk['rrf_score'] = rrf_scores[idx]
            chunk['bm25_rank'] = bm25_ranks.get(idx, -1)
            chunk['vector_rank'] = vector_ranks.get(idx, -1)
            results.append(chunk)
        
        return results


# ============================================================================
# Improved Claim Verifier with better prompting
# ============================================================================

class FastClaimVerifier:
    """Verify claims with improved prompting for better accuracy."""
    
    SYSTEM_PROMPT = """You are a meticulous fact-checker analyzing claims about fictional characters against evidence from their source novel.

Your task: Determine if each claim is SUPPORTED, CONTRADICTED, or UNCLEAR based on the evidence.

CRITICAL - How to identify CONTRADICTIONS:
- Dates/times differ (claim says "1815" but evidence shows "1811")
- Events differ (claim says "escaped" but evidence shows "was captured")
- Relationships differ (claim says "brother" but evidence shows "cousin")
- Actions differ (claim says "killed" but evidence shows "saved")
- Any factual inconsistency between claim and evidence

SUPPORTS: Evidence explicitly confirms or matches the claimed facts.

UNCLEAR: The evidence doesn't mention or address the claim at all.

Be AGGRESSIVE in detecting contradictions - if the evidence provides different facts than the claim, that's a CONTRADICTION."""

    USER_PROMPT = """Verify this claim about {character} from "{book_name}":

CLAIM: "{claim}"

EVIDENCE FROM BOOK:
{evidence}

INSTRUCTIONS:
1. Look for ANY factual inconsistency between the claim and evidence
2. Pay attention to: dates, names, locations, relationships, events, outcomes
3. If evidence shows DIFFERENT facts than the claim → CONTRADICTS
4. If evidence confirms the claim → SUPPORTS
5. If evidence doesn't address the claim → UNCLEAR

VERDICT: [SUPPORTS/CONTRADICTS/UNCLEAR]
REASON: [Brief explanation of why]"""

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
        """Verify a claim against evidence."""
        # Format evidence with clear labeling
        evidence_parts = []
        for i, e in enumerate(evidence[:self.config.top_k_final]):
            content = e.get(content_key, '')[:600]  # More context per chunk
            chapter = e.get('chapter', 'Unknown')
            evidence_parts.append(f"[Passage {i+1}, {chapter}]:\n{content}")
        
        evidence_text = "\n\n".join(evidence_parts)
        
        if not evidence_text.strip():
            return {
                'claim': claim,
                'verdict': 'unclear',
                'confidence': 0.3,
                'reasoning': 'No evidence found'
            }
        
        prompt = self.USER_PROMPT.format(
            character=character,
            book_name=book_name,
            claim=claim,
            evidence=evidence_text
        )
        
        response = self.llm.generate(
            prompt, 
            max_tokens=150,
            system_prompt=self.SYSTEM_PROMPT
        )
        
        # Parse structured response
        verdict, confidence, reasoning = self._parse_response(response)
        
        return {
            'claim': claim,
            'verdict': verdict,
            'confidence': confidence,
            'reasoning': reasoning
        }
    
    def _parse_response(self, response: str) -> Tuple[str, float, str]:
        """Parse the structured LLM response."""
        response_upper = response.upper()
        
        # Look for explicit verdict
        if 'VERDICT:' in response_upper:
            verdict_match = re.search(r'VERDICT:\s*(SUPPORTS?|CONTRADICTS?|UNCLEAR)', response_upper)
            if verdict_match:
                v = verdict_match.group(1)
                if 'CONTRADICT' in v:
                    verdict = 'contradicts'
                    confidence = 0.85
                elif 'SUPPORT' in v:
                    verdict = 'supports'
                    confidence = 0.85
                else:
                    verdict = 'unclear'
                    confidence = 0.5
                
                # Extract reasoning
                reason_match = re.search(r'REASON:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
                reasoning = reason_match.group(1).strip()[:200] if reason_match else response[:200]
                
                return verdict, confidence, reasoning
        
        # Fallback: keyword matching
        response_lower = response.lower()
        if 'contradict' in response_lower and 'not contradict' not in response_lower:
            return 'contradicts', 0.7, response[:200]
        elif 'support' in response_lower and 'not support' not in response_lower:
            return 'supports', 0.7, response[:200]
        else:
            return 'unclear', 0.5, response[:200]


# ============================================================================
# Improved Aggregator with weighted scoring
# ============================================================================

class FastAggregator:
    """Aggregate verdicts into final prediction with confidence weighting."""
    
    def aggregate(self, results: List[Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
        """
        Balanced aggregation:
        - ANY high-confidence contradiction → predict 0
        - Multiple contradictions (>=2) → predict 0
        - Otherwise → predict 1 (give benefit of doubt)
        """
        if not results:
            return 1, {'reason': 'No claims extracted', 'verdicts': []}
        
        # Calculate weighted scores
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
        
        # Decision logic
        # 1. Any contradiction (conf >= 0.7) → contradict
        has_contradict = any(
            r['verdict'] == 'contradicts' and r.get('confidence', 0) >= 0.65
            for r in results
        )
        
        # 2. Multiple contradictions regardless of confidence → contradict
        multiple_contradicts = n_contradicts >= 2
        
        if has_contradict or multiple_contradicts:
            prediction = 0
            reason = f"Contradiction detected ({n_contradicts} contradictions, {n_supports} supports)"
        else:
            prediction = 1
            reason = f"Consistent ({n_supports} supports, {n_unclear} unclear, {n_contradicts} contradicts)"
        
        return prediction, {
            'reason': reason,
            'verdicts': results,
            'counts': {
                'contradicts': n_contradicts, 
                'supports': n_supports, 
                'unclear': n_unclear
            },
            'scores': {
                'contradiction': contradiction_score,
                'support': support_score
            }
        }


# ============================================================================
# Fast Verification Pipeline
# ============================================================================

class FastVerificationPipeline:
    """
    Optimized verification pipeline with improved prompts.
    
    LLM calls per sample: ~6 (1 extraction + 5 verifications)
    """
    
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
        
        # 1. Extract claims (1 LLM call)
        if verbose:
            print("1. Extracting claims...")
        claims = self.extractor.extract(backstory, character, book_name)
        if verbose:
            print(f"   Found {len(claims)} claims")
            for i, c in enumerate(claims):
                print(f"   [{i+1}] {c[:70]}...")
        
        if not claims:
            return 1, {
                'reason': 'No claims extracted', 
                'elapsed': time.time() - start_time,
                'claims': []
            }
        
        # 2. For each claim: retrieve + verify
        results = []
        for i, claim in enumerate(claims):
            if verbose:
                print(f"\n2. Verifying claim {i+1}/{len(claims)}...")
            
            # Retrieve (no LLM)
            evidence = self.retriever.search(claim)
            if verbose:
                print(f"   Retrieved {len(evidence)} chunks")
            
            # Verify (1 LLM call)
            result = self.verifier.verify(claim, evidence, character, book_name)
            results.append(result)
            
            if verbose:
                print(f"   Verdict: {result['verdict']} (conf: {result['confidence']:.2f})")
                print(f"   Reason: {result['reasoning'][:80]}...")
        
        # 3. Aggregate
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
# Fast Evaluator
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
                prediction = 1  # Default to consistent on error
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
        
        # Per-class accuracy
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


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Fast verifier module. Run via: python -m pipeline.run_eval_fast")
