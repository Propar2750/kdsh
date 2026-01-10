"""
FAST Verifier module for KDSH Track A pipeline.
Uses Groq API for fast, free LLM inference.

Key optimizations:
1. Groq API - Free, fast cloud inference (~1s per call vs ~120s local)
2. NO LLM-based reranking (uses score-based fusion instead)
3. NO separate query generation (uses claim directly)
"""

import re
import json
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
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
    llm_temperature: float = 0.1
    llm_max_tokens: int = 512
    
    # Retrieval settings
    top_k_retrieval: int = 10     # Chunks from hybrid retrieval per query
    top_k_final: int = 5          # Final chunks for verification (no LLM rerank)
    bm25_weight: float = 0.4
    vector_weight: float = 0.6
    
    # Verification settings
    max_claims: int = 5           # Reduced from 10 for speed


DEFAULT_FAST_CONFIG = FastVerifierConfig()


# ============================================================================
# Groq LLM Wrapper (Fast, Free API)
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
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate text using Groq API."""
        client = self._get_client()
        
        start = time.time()
        self.call_count += 1
        
        max_tok = max_tokens or self.config.llm_max_tokens
        
        try:
            response = client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.llm_temperature,
                max_tokens=max_tok,
            )
            result = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Groq API error: {e}")
            result = "ERROR"
        
        elapsed = time.time() - start
        self.total_time += elapsed
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get LLM call statistics."""
        return {
            'call_count': self.call_count,
            'total_time': self.total_time,
            'avg_time': self.total_time / max(1, self.call_count)
        }


# ============================================================================
# Fast Claim Extractor
# ============================================================================

class FastClaimExtractor:
    """Extract claims with a single LLM call."""
    
    PROMPT = """Extract the key factual claims from this character backstory.
Focus on verifiable facts about events, relationships, actions, and timeline.

Character: {character}
Book: {book_name}

Backstory:
{backstory}

List up to {max_claims} specific, verifiable claims (one per line, numbered):"""

    def __init__(self, llm: GroqLLM, config: Optional[FastVerifierConfig] = None):
        self.llm = llm
        self.config = config or DEFAULT_FAST_CONFIG
    
    def extract(self, backstory: str, character: str, book_name: str) -> List[str]:
        """Extract claims from backstory."""
        prompt = self.PROMPT.format(
            character=character,
            book_name=book_name,
            backstory=backstory[:2000],  # Truncate for speed
            max_claims=self.config.max_claims
        )
        
        response = self.llm.generate(prompt, max_tokens=400)
        
        # Parse numbered claims
        claims = []
        for line in response.split('\n'):
            line = line.strip()
            # Match "1. claim" or "1) claim" or just numbered lines
            match = re.match(r'^[\d]+[.\)]\s*(.+)', line)
            if match:
                claim = match.group(1).strip()
                if len(claim) > 10:  # Filter very short claims
                    claims.append(claim)
        
        return claims[:self.config.max_claims]


# ============================================================================
# BM25 Retriever (no external dependencies)
# ============================================================================

class SimpleBM25:
    """Simple BM25 implementation."""
    
    def __init__(self, chunks: List[Dict[str, Any]], content_key: str = "content"):
        self.chunks = chunks
        self.content_key = content_key
        
        # Build corpus
        self.corpus = [self._tokenize(c[content_key]) for c in chunks]
        self.doc_len = [len(doc) for doc in self.corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 1
        self.N = len(self.corpus)
        
        # Build IDF
        self.idf = {}
        df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                df[term] += 1
        for term, freq in df.items():
            self.idf[term] = np.log((self.N - freq + 0.5) / (freq + 0.5) + 1)
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Return (doc_index, score) pairs."""
        query_terms = self._tokenize(query)
        scores = []
        
        k1, b = 1.5, 0.75
        
        for i, doc in enumerate(self.corpus):
            score = 0.0
            doc_len = self.doc_len[i]
            term_freq = Counter(doc)
            
            for term in query_terms:
                if term in term_freq:
                    tf = term_freq[term]
                    idf = self.idf.get(term, 0)
                    score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / self.avgdl))
            
            scores.append((i, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ============================================================================
# Hybrid Retriever (Fast - No LLM)
# ============================================================================

class FastHybridRetriever:
    """Fast hybrid retrieval without LLM reranking."""
    
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
        self.bm25 = SimpleBM25(chunks, content_key)
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Hybrid search with score fusion (no LLM)."""
        k = top_k or self.config.top_k_retrieval
        
        # BM25 search
        bm25_results = self.bm25.search(query, top_k=k * 2)
        bm25_scores = {idx: score for idx, score in bm25_results}
        
        # Vector search
        vector_results = self.embedder.search(query, top_k=k * 2)
        vector_scores = {}
        for r in vector_results:
            # Find chunk index
            for i, c in enumerate(self.chunks):
                if c.get('chunk_id') == r.get('chunk_id'):
                    vector_scores[i] = r['score']
                    break
        
        # Normalize and combine
        if bm25_scores:
            max_bm25 = max(bm25_scores.values()) or 1
            bm25_scores = {k: v / max_bm25 for k, v in bm25_scores.items()}
        
        if vector_scores:
            max_vec = max(vector_scores.values()) or 1
            vector_scores = {k: v / max_vec for k, v in vector_scores.items()}
        
        # Combine scores
        all_indices = set(bm25_scores.keys()) | set(vector_scores.keys())
        combined = {}
        for idx in all_indices:
            combined[idx] = (
                self.config.bm25_weight * bm25_scores.get(idx, 0) +
                self.config.vector_weight * vector_scores.get(idx, 0)
            )
        
        # Sort and return
        sorted_indices = sorted(combined.keys(), key=lambda x: combined[x], reverse=True)
        
        results = []
        for idx in sorted_indices[:k]:
            chunk = self.chunks[idx].copy()
            chunk['hybrid_score'] = combined[idx]
            chunk['bm25_score'] = bm25_scores.get(idx, 0)
            chunk['vector_score'] = vector_scores.get(idx, 0)
            results.append(chunk)
        
        return results


# ============================================================================
# Fast Claim Verifier
# ============================================================================

class FastClaimVerifier:
    """Verify claims with minimal LLM calls."""
    
    PROMPT = """Verify if this claim about a character is supported or contradicted by the evidence.

Character: {character}
Book: {book_name}

CLAIM: {claim}

EVIDENCE FROM THE BOOK:
{evidence}

Instructions:
- Output "CONTRADICTS" if evidence shows the claim is false
- Output "SUPPORTS" if evidence confirms the claim
- Output "UNCLEAR" if insufficient evidence

IMPORTANT: Look for factual conflicts - different events, dates, relationships, or details.

Your verdict (one word only):"""

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
        """Verify a claim."""
        # Format evidence (limit to top chunks)
        evidence_text = "\n\n".join([
            f"[{i+1}] {e[content_key][:500]}"
            for i, e in enumerate(evidence[:self.config.top_k_final])
        ])
        
        if not evidence_text.strip():
            return {'verdict': 'unclear', 'confidence': 0.3, 'reasoning': 'No evidence found'}
        
        prompt = self.PROMPT.format(
            character=character,
            book_name=book_name,
            claim=claim,
            evidence=evidence_text
        )
        
        response = self.llm.generate(prompt, max_tokens=50)
        
        # Parse verdict
        response_lower = response.lower()
        if 'contradict' in response_lower:
            verdict = 'contradicts'
            confidence = 0.8
        elif 'support' in response_lower:
            verdict = 'supports'
            confidence = 0.8
        else:
            verdict = 'unclear'
            confidence = 0.5
        
        return {
            'claim': claim,
            'verdict': verdict,
            'confidence': confidence,
            'reasoning': response[:200]
        }


# ============================================================================
# Fast Aggregator
# ============================================================================

class FastAggregator:
    """Aggregate verdicts into final prediction."""
    
    def aggregate(self, results: List[Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
        """
        Simple aggregation:
        - Any contradiction → predict 0
        - Otherwise → predict 1
        """
        if not results:
            return 1, {'reason': 'No claims', 'verdicts': []}
        
        verdicts = [r['verdict'] for r in results]
        n_contradicts = verdicts.count('contradicts')
        n_supports = verdicts.count('supports')
        
        if n_contradicts > 0:
            prediction = 0
            reason = f"Found {n_contradicts} contradiction(s)"
        else:
            prediction = 1
            reason = f"No contradictions found ({n_supports} supports, {verdicts.count('unclear')} unclear)"
        
        return prediction, {
            'reason': reason,
            'verdicts': results,
            'counts': {'contradicts': n_contradicts, 'supports': n_supports, 'unclear': verdicts.count('unclear')}
        }


# ============================================================================
# Fast Verification Pipeline
# ============================================================================

class FastVerificationPipeline:
    """
    Optimized verification pipeline.
    
    LLM calls per sample: ~6 (1 extraction + 5 verifications)
    vs original: ~65+ calls
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
        """Verify a backstory."""
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
        
        if not claims:
            return 1, {'reason': 'No claims extracted', 'elapsed': time.time() - start_time}
        
        # 2. For each claim: retrieve + verify (1 LLM call per claim)
        results = []
        for i, claim in enumerate(claims):
            if verbose:
                print(f"2. Claim {i+1}/{len(claims)}: {claim[:60]}...")
            
            # Retrieve (no LLM)
            evidence = self.retriever.search(claim)
            if verbose:
                print(f"   Retrieved {len(evidence)} chunks")
            
            # Verify (1 LLM call)
            result = self.verifier.verify(claim, evidence, character, book_name)
            results.append(result)
            
            if verbose:
                print(f"   Verdict: {result['verdict']}")
        
        # 3. Aggregate
        prediction, details = self.aggregator.aggregate(results)
        
        elapsed = time.time() - start_time
        details['elapsed'] = elapsed
        details['llm_stats'] = self.llm.get_stats()
        
        if verbose:
            print(f"\nPrediction: {prediction} ({'consistent' if prediction == 1 else 'contradict'})")
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
        
        print(f"\n{'='*50}")
        print(f"Evaluating {len(samples)} samples (FAST mode)")
        print(f"{'='*50}")
        
        label_map = {'consistent': 1, 'contradict': 0}
        
        for i, sample in enumerate(samples):
            print(f"\n[{i+1}/{len(samples)}] Sample {sample['id']}")
            
            true_label = label_map.get(sample['label'], -1)
            
            prediction, details = self.pipeline.verify_backstory(
                backstory=sample['content'],
                character=sample['char'],
                book_name=sample['book_name'],
                verbose=verbose
            )
            
            correct = prediction == true_label
            status = "✓" if correct else "✗"
            
            self.results.append({
                'id': sample['id'],
                'true_label': sample['label'],
                'prediction': 'consistent' if prediction == 1 else 'contradict',
                'correct': correct,
                'details': details
            })
            
            print(f"{status} True: {sample['label']}, Pred: {'consistent' if prediction == 1 else 'contradict'}")
        
        # Summary
        correct = sum(1 for r in self.results if r['correct'])
        accuracy = correct / len(self.results) if self.results else 0
        
        llm_stats = self.pipeline.llm.get_stats()
        
        print(f"\n{'='*50}")
        print(f"RESULTS: {correct}/{len(self.results)} correct ({accuracy:.1%})")
        print(f"LLM calls: {llm_stats['call_count']} | Total time: {llm_stats['total_time']:.1f}s")
        print(f"Avg per call: {llm_stats['avg_time']:.1f}s")
        print(f"{'='*50}")
        
        return {
            'accuracy': accuracy,
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
