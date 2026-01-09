"""
Verifier module for KDSH Track A pipeline.
Full verification pipeline: claim extraction → retrieval → verification → aggregation.

Optimizations:
- Batched LLM inference using HF datasets
- Parallel claim processing
- GPU memory-efficient model loading
"""

import re
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class VerifierConfig:
    """Configuration for the verification pipeline."""
    # LLM settings
    llm_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    llm_max_new_tokens: int = 512
    llm_temperature: float = 0.1  # Low temperature for deterministic output
    llm_batch_size: int = 4       # Batch size for LLM inference
    
    # Retrieval settings
    top_k_per_query: int = 5      # Chunks per query from each retriever
    top_k_reranked: int = 3       # Final chunks after reranking
    bm25_weight: float = 0.4      # Weight for BM25 in hybrid fusion
    vector_weight: float = 0.6   # Weight for vector search in hybrid fusion
    
    # Verification settings
    max_claims: int = 10          # Max claims to extract per backstory
    contradiction_threshold: float = 0.5  # Threshold for contradiction verdict
    
    # Parallelization settings
    parallel_claims: bool = False  # Parallel claim processing (careful with GPU memory)
    max_workers: int = 2           # Max parallel workers for claim processing
    
    # Memory optimization
    use_8bit: bool = False        # Use 8-bit quantization (saves ~50% VRAM)
    max_context_length: int = 4096 # Max context window


DEFAULT_VERIFIER_CONFIG = VerifierConfig()


# ============================================================================
# LLM Wrapper
# ============================================================================

class LlamaLLM:
    """
    Wrapper for Llama-3.1-8B-Instruct model.
    Lazy loading to avoid memory issues.
    
    Optimizations:
    - Uses `dtype` instead of deprecated `torch_dtype`
    - Supports batched inference via HF datasets
    - Optional 8-bit quantization for memory savings
    """
    
    def __init__(self, config: Optional[VerifierConfig] = None):
        self.config = config or DEFAULT_VERIFIER_CONFIG
        self._model = None
        self._tokenizer = None
        self._pipeline = None
    
    def _load_model(self):
        """Lazy load the model with memory optimizations."""
        if self._pipeline is not None:
            return
        
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
        
        print(f"Loading LLM: {self.config.llm_model}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Report GPU memory
        if device == "cuda":
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Device: {device} (GPU Memory: {gpu_mem:.1f} GB)")
        else:
            print(f"Device: {device}")
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model)
        
        # Set pad token if not set
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Model loading configuration
        model_kwargs = {
            "device_map": "auto",
        }
        
        # Use 8-bit quantization if configured (saves ~50% VRAM)
        if self.config.use_8bit and device == "cuda":
            try:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                model_kwargs["quantization_config"] = quantization_config
                print("Using 8-bit quantization for memory efficiency")
            except ImportError:
                print("bitsandbytes not available, using float16")
                model_kwargs["dtype"] = torch.float16
        else:
            # Use dtype instead of torch_dtype (fixes deprecation warning)
            model_kwargs["dtype"] = torch.float16
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.llm_model,
            **model_kwargs
        )
        
        # Create pipeline - use dtype instead of torch_dtype
        self._pipeline = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            batch_size=self.config.llm_batch_size,  # Enable batching
        )
        
        # Report memory usage
        if device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"LLM loaded! GPU Memory: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved")
        else:
            print("LLM loaded successfully!")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text (without the prompt)
        """
        self._load_model()
        
        max_tokens = max_new_tokens or self.config.llm_max_new_tokens
        temp = temperature or self.config.llm_temperature
        
        # Format as chat message
        messages = [{"role": "user", "content": prompt}]
        
        result = self._pipeline(
            messages,
            max_new_tokens=max_tokens,
            temperature=temp,
            do_sample=temp > 0,
            pad_token_id=self._tokenizer.eos_token_id
        )
        
        # Extract generated text
        generated = result[0]["generated_text"]
        if isinstance(generated, list):
            # Chat format returns list of messages
            return generated[-1]["content"]
        return generated[len(prompt):]
    
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> List[str]:
        """
        Generate text from multiple prompts in batch (more efficient on GPU).
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            
        Returns:
            List of generated texts
        """
        self._load_model()
        
        max_tokens = max_new_tokens or self.config.llm_max_new_tokens
        temp = temperature or self.config.llm_temperature
        
        # Format all as chat messages
        all_messages = [[{"role": "user", "content": p}] for p in prompts]
        
        # Use HuggingFace datasets for efficient batching
        try:
            from datasets import Dataset
            
            # Create dataset for batched inference
            dataset = Dataset.from_dict({"messages": all_messages})
            
            results = []
            for out in self._pipeline(
                dataset["messages"],
                max_new_tokens=max_tokens,
                temperature=temp,
                do_sample=temp > 0,
                pad_token_id=self._tokenizer.eos_token_id,
                batch_size=self.config.llm_batch_size
            ):
                generated = out[0]["generated_text"]
                if isinstance(generated, list):
                    results.append(generated[-1]["content"])
                else:
                    results.append(generated)
            
            return results
            
        except ImportError:
            # Fallback to sequential if datasets not available
            return [self.generate(p, max_new_tokens, temperature) for p in prompts]
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        if not torch.cuda.is_available():
            return {"device": "cpu"}
        
        return {
            "device": "cuda",
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9
        }
    
    def __del__(self):
        """Clean up model resources."""
        try:
            if self._model is not None:
                del self._model
                del self._tokenizer
                del self._pipeline
                if torch is not None and torch.cuda is not None:
                    torch.cuda.empty_cache()
        except Exception:
            pass  # Ignore cleanup errors during interpreter shutdown


# ============================================================================
# Claim Extractor
# ============================================================================

class ClaimExtractor:
    """
    Extracts atomic facts/claims from backstory text using LLM.
    """
    
    EXTRACTION_PROMPT = """Extract atomic facts from the following backstory about a character.
Each fact should be a single, verifiable statement that can be checked against the novel.
Focus on: events, relationships, character traits, locations, and timeline.

Backstory:
{backstory}

Character: {character}
Book: {book_name}

Instructions:
1. Extract up to {max_claims} atomic facts
2. Each fact should be independently verifiable
3. Be specific about names, places, and events
4. Output as a JSON array of strings

Output format (JSON array only, no explanation):
["fact 1", "fact 2", "fact 3", ...]

Atomic facts:"""

    def __init__(self, llm: LlamaLLM, config: Optional[VerifierConfig] = None):
        self.llm = llm
        self.config = config or DEFAULT_VERIFIER_CONFIG
    
    def extract_claims(
        self,
        backstory: str,
        character: str,
        book_name: str
    ) -> List[str]:
        """
        Extract atomic claims from a backstory.
        
        Args:
            backstory: The backstory text
            character: Character name
            book_name: Name of the book
            
        Returns:
            List of atomic claim strings
        """
        prompt = self.EXTRACTION_PROMPT.format(
            backstory=backstory,
            character=character,
            book_name=book_name,
            max_claims=self.config.max_claims
        )
        
        response = self.llm.generate(prompt, max_new_tokens=1024)
        
        # Parse JSON response
        claims = self._parse_claims(response)
        
        # Limit to max claims
        return claims[:self.config.max_claims]
    
    def _parse_claims(self, response: str) -> List[str]:
        """Parse claims from LLM response."""
        # Try to find JSON array in response
        try:
            # Look for JSON array pattern
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                claims = json.loads(match.group())
                if isinstance(claims, list):
                    return [str(c).strip() for c in claims if c]
        except json.JSONDecodeError:
            pass
        
        # Fallback: split by newlines and clean
        lines = response.strip().split('\n')
        claims = []
        for line in lines:
            # Remove numbering, bullets, quotes
            line = re.sub(r'^[\d\.\-\*\•]+\s*', '', line.strip())
            line = line.strip('"\'')
            if line and len(line) > 10:
                claims.append(line)
        
        return claims


# ============================================================================
# Query Generator
# ============================================================================

class QueryGenerator:
    """
    Generates search queries from atomic claims.
    Creates multiple query variations for better recall.
    """
    
    QUERY_PROMPT = """Given this claim about a character, generate 2-3 search queries to find evidence in the novel.
Make queries specific enough to find relevant passages.

Claim: {claim}
Character: {character}
Book: {book_name}

Generate queries as a JSON array:
["query 1", "query 2", "query 3"]

Queries:"""

    def __init__(self, llm: LlamaLLM):
        self.llm = llm
    
    def generate_queries(
        self,
        claim: str,
        character: str,
        book_name: str
    ) -> List[str]:
        """
        Generate search queries for a claim.
        
        Args:
            claim: The atomic claim
            character: Character name
            book_name: Book name
            
        Returns:
            List of search queries
        """
        prompt = self.QUERY_PROMPT.format(
            claim=claim,
            character=character,
            book_name=book_name
        )
        
        response = self.llm.generate(prompt, max_new_tokens=256)
        
        # Parse queries
        queries = self._parse_queries(response)
        
        # Always include the claim itself as a query
        if claim not in queries:
            queries.insert(0, claim)
        
        return queries[:3]  # Max 3 queries per claim
    
    def _parse_queries(self, response: str) -> List[str]:
        """Parse queries from LLM response."""
        try:
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                queries = json.loads(match.group())
                if isinstance(queries, list):
                    return [str(q).strip() for q in queries if q]
        except json.JSONDecodeError:
            pass
        
        # Fallback
        lines = response.strip().split('\n')
        queries = []
        for line in lines:
            line = re.sub(r'^[\d\.\-\*\•]+\s*', '', line.strip())
            line = line.strip('"\'')
            if line and len(line) > 5:
                queries.append(line)
        
        return queries


# ============================================================================
# BM25 Retriever
# ============================================================================

class BM25Retriever:
    """
    BM25 sparse retrieval for text search.
    """
    
    def __init__(self, chunks: List[Dict[str, Any]], content_key: str = "content"):
        """
        Initialize BM25 index.
        
        Args:
            chunks: List of chunk dictionaries
            content_key: Key for content in chunk dict
        """
        self.chunks = chunks
        self.content_key = content_key
        self._index = None
        self._tokenized_corpus = None
        self._build_index()
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def _build_index(self):
        """Build BM25 index."""
        from rank_bm25 import BM25Okapi
        
        self._tokenized_corpus = [
            self._tokenize(chunk[self.content_key])
            for chunk in self.chunks
        ]
        self._index = BM25Okapi(self._tokenized_corpus)
        print(f"BM25 index built with {len(self.chunks)} documents")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of (chunk, score) results
        """
        tokenized_query = self._tokenize(query)
        scores = self._index.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            result = self.chunks[idx].copy()
            result['bm25_score'] = float(scores[idx])
            results.append(result)
        
        return results


# ============================================================================
# Hybrid Retriever
# ============================================================================

class HybridRetriever:
    """
    Combines BM25 and vector search with score fusion.
    """
    
    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        embedder,  # ChunkEmbedder instance
        config: Optional[VerifierConfig] = None,
        content_key: str = "content"
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            chunks: List of chunk dictionaries
            embedder: ChunkEmbedder with pre-computed embeddings
            config: Verifier configuration
            content_key: Key for content in chunk dict
        """
        self.chunks = chunks
        self.embedder = embedder
        self.config = config or DEFAULT_VERIFIER_CONFIG
        self.content_key = content_key
        
        # Initialize BM25
        self.bm25 = BM25Retriever(chunks, content_key)
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining BM25 and vector search.
        
        Args:
            query: Search query
            top_k: Number of results (default from config)
            
        Returns:
            List of chunks with combined scores
        """
        k = top_k or self.config.top_k_per_query
        
        # Get more results from each retriever for fusion
        bm25_results = self.bm25.search(query, top_k=k * 2)
        vector_results = self.embedder.search(query, top_k=k * 2)
        
        # Create score maps
        bm25_scores = {}
        for r in bm25_results:
            chunk_id = r.get('chunk_id', r.get(self.content_key, '')[:50])
            bm25_scores[chunk_id] = r['bm25_score']
        
        vector_scores = {}
        for r in vector_results:
            chunk_id = r.get('chunk_id', r.get(self.content_key, '')[:50])
            vector_scores[chunk_id] = r['score']
        
        # Normalize scores
        if bm25_scores:
            max_bm25 = max(bm25_scores.values())
            if max_bm25 > 0:
                bm25_scores = {k: v / max_bm25 for k, v in bm25_scores.items()}
        
        if vector_scores:
            max_vec = max(vector_scores.values())
            if max_vec > 0:
                vector_scores = {k: v / max_vec for k, v in vector_scores.items()}
        
        # Combine scores using Reciprocal Rank Fusion
        all_chunk_ids = set(bm25_scores.keys()) | set(vector_scores.keys())
        combined_scores = {}
        
        for chunk_id in all_chunk_ids:
            bm25_s = bm25_scores.get(chunk_id, 0)
            vec_s = vector_scores.get(chunk_id, 0)
            combined_scores[chunk_id] = (
                self.config.bm25_weight * bm25_s +
                self.config.vector_weight * vec_s
            )
        
        # Sort by combined score
        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
        
        # Build results
        results = []
        seen_ids = set()
        
        # First add from vector results (have full chunk data)
        for r in vector_results:
            chunk_id = r.get('chunk_id', r.get(self.content_key, '')[:50])
            if chunk_id in sorted_ids[:k] and chunk_id not in seen_ids:
                r['hybrid_score'] = combined_scores[chunk_id]
                r['bm25_score'] = bm25_scores.get(chunk_id, 0)
                r['vector_score'] = vector_scores.get(chunk_id, 0)
                results.append(r)
                seen_ids.add(chunk_id)
        
        # Then add from BM25 results
        for r in bm25_results:
            chunk_id = r.get('chunk_id', r.get(self.content_key, '')[:50])
            if chunk_id in sorted_ids[:k] and chunk_id not in seen_ids:
                r['hybrid_score'] = combined_scores[chunk_id]
                r['vector_score'] = vector_scores.get(chunk_id, 0)
                results.append(r)
                seen_ids.add(chunk_id)
        
        # Sort by hybrid score
        results.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)
        
        return results[:k]


# ============================================================================
# Reranker
# ============================================================================

class Reranker:
    """
    Reranks retrieved chunks using cross-encoder or LLM scoring.
    Uses a simple LLM-based relevance scoring.
    """
    
    RERANK_PROMPT = """Rate how relevant this passage is for verifying the claim.
Score from 0 to 10: 0 = completely irrelevant, 10 = directly addresses the claim.

Claim: {claim}

Passage: {passage}

Output only a number from 0 to 10:"""

    def __init__(self, llm: LlamaLLM, config: Optional[VerifierConfig] = None):
        self.llm = llm
        self.config = config or DEFAULT_VERIFIER_CONFIG
    
    def rerank(
        self,
        claim: str,
        chunks: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        content_key: str = "content"
    ) -> List[Dict[str, Any]]:
        """
        Rerank chunks by relevance to claim.
        
        Args:
            claim: The claim being verified
            chunks: Retrieved chunks
            top_k: Number of top chunks to return
            content_key: Key for content in chunk dict
            
        Returns:
            Reranked chunks with relevance scores
        """
        k = top_k or self.config.top_k_reranked
        
        if not chunks:
            return []
        
        # Score each chunk
        scored_chunks = []
        for chunk in chunks:
            prompt = self.RERANK_PROMPT.format(
                claim=claim,
                passage=chunk[content_key][:500]  # Truncate for efficiency
            )
            
            response = self.llm.generate(prompt, max_new_tokens=16, temperature=0)
            score = self._parse_score(response)
            
            chunk_copy = chunk.copy()
            chunk_copy['relevance_score'] = score
            scored_chunks.append(chunk_copy)
        
        # Sort by relevance
        scored_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return scored_chunks[:k]
    
    def _parse_score(self, response: str) -> float:
        """Parse score from LLM response."""
        try:
            # Find first number in response
            match = re.search(r'\d+(?:\.\d+)?', response)
            if match:
                score = float(match.group())
                return min(10, max(0, score))  # Clamp to [0, 10]
        except:
            pass
        return 5.0  # Default middle score


# ============================================================================
# Claim Verifier
# ============================================================================

@dataclass
class VerificationResult:
    """Result of verifying a single claim."""
    claim: str
    verdict: str  # "supports", "contradicts", "unclear"
    confidence: float  # 0.0 to 1.0
    evidence: List[Dict[str, Any]]
    reasoning: str


class ClaimVerifier:
    """
    Verifies claims against retrieved evidence using LLM.
    """
    
    VERIFICATION_PROMPT = """You are a fact-checker verifying claims against novel text evidence.

Claim to verify: {claim}

Character: {character}
Book: {book_name}

Evidence passages from the novel:
{evidence}

Instructions:
1. Carefully analyze if the claim is supported or contradicted by the evidence
2. A claim CONTRADICTS if the evidence shows something incompatible with the claim
3. A claim is SUPPORTED if the evidence confirms the claim
4. Mark as UNCLEAR if there's insufficient evidence

Common contradiction patterns:
- Different events, dates, or sequences than claimed
- Different relationships or character traits
- Claims about events that evidence shows didn't happen
- Factual details that conflict with evidence

Output your analysis in this exact JSON format:
{{
    "verdict": "supports" or "contradicts" or "unclear",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation of your verdict"
}}

Analysis:"""

    def __init__(self, llm: LlamaLLM, config: Optional[VerifierConfig] = None):
        self.llm = llm
        self.config = config or DEFAULT_VERIFIER_CONFIG
    
    def verify_claim(
        self,
        claim: str,
        evidence: List[Dict[str, Any]],
        character: str,
        book_name: str,
        content_key: str = "content"
    ) -> VerificationResult:
        """
        Verify a claim against evidence.
        
        Args:
            claim: The claim to verify
            evidence: Retrieved evidence chunks
            character: Character name
            book_name: Book name
            content_key: Key for content in evidence dict
            
        Returns:
            VerificationResult with verdict and reasoning
        """
        # Format evidence
        evidence_text = "\n\n".join([
            f"[Passage {i+1}]: {e[content_key]}"
            for i, e in enumerate(evidence[:5])  # Max 5 passages
        ])
        
        if not evidence_text.strip():
            evidence_text = "[No relevant evidence found]"
        
        prompt = self.VERIFICATION_PROMPT.format(
            claim=claim,
            character=character,
            book_name=book_name,
            evidence=evidence_text
        )
        
        response = self.llm.generate(prompt, max_new_tokens=512)
        
        # Parse response
        result = self._parse_verification(response)
        
        return VerificationResult(
            claim=claim,
            verdict=result['verdict'],
            confidence=result['confidence'],
            evidence=evidence,
            reasoning=result['reasoning']
        )
    
    def _parse_verification(self, response: str) -> Dict[str, Any]:
        """Parse verification result from LLM response."""
        default = {
            'verdict': 'unclear',
            'confidence': 0.5,
            'reasoning': 'Could not parse response'
        }
        
        try:
            # Find JSON in response
            match = re.search(r'\{.*?\}', response, re.DOTALL)
            if match:
                result = json.loads(match.group())
                verdict = result.get('verdict', 'unclear').lower()
                if verdict not in ['supports', 'contradicts', 'unclear']:
                    verdict = 'unclear'
                
                confidence = float(result.get('confidence', 0.5))
                confidence = min(1.0, max(0.0, confidence))
                
                return {
                    'verdict': verdict,
                    'confidence': confidence,
                    'reasoning': result.get('reasoning', '')
                }
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback: look for keywords
        response_lower = response.lower()
        if 'contradict' in response_lower:
            default['verdict'] = 'contradicts'
            default['confidence'] = 0.7
        elif 'support' in response_lower:
            default['verdict'] = 'supports'
            default['confidence'] = 0.7
        
        default['reasoning'] = response[:200]
        return default


# ============================================================================
# Aggregator
# ============================================================================

class Aggregator:
    """
    Aggregates claim verification results into final prediction.
    """
    
    def __init__(self, config: Optional[VerifierConfig] = None):
        self.config = config or DEFAULT_VERIFIER_CONFIG
    
    def aggregate(
        self,
        results: List[VerificationResult]
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Aggregate verification results into final prediction.
        
        Strategy: 
        - If ANY claim contradicts with high confidence → predict 0 (contradict)
        - Otherwise → predict 1 (consistent)
        
        Args:
            results: List of VerificationResult
            
        Returns:
            (prediction, details) where prediction is 0 or 1
        """
        if not results:
            return 1, {'reason': 'No claims extracted', 'verdicts': []}
        
        verdicts = []
        contradiction_score = 0.0
        support_score = 0.0
        
        for r in results:
            verdict_info = {
                'claim': r.claim,
                'verdict': r.verdict,
                'confidence': r.confidence,
                'reasoning': r.reasoning
            }
            verdicts.append(verdict_info)
            
            if r.verdict == 'contradicts':
                contradiction_score += r.confidence
            elif r.verdict == 'supports':
                support_score += r.confidence
        
        # Count verdicts
        verdict_counts = Counter(r.verdict for r in results)
        
        # Decision logic:
        # 1. If any high-confidence contradiction exists → contradict
        # 2. If more contradictions than supports → contradict
        # 3. Otherwise → consistent
        
        n_contradicts = verdict_counts.get('contradicts', 0)
        n_supports = verdict_counts.get('supports', 0)
        
        # Check for any high-confidence contradiction
        high_conf_contradicts = any(
            r.verdict == 'contradicts' and r.confidence >= 0.7
            for r in results
        )
        
        if high_conf_contradicts:
            prediction = 0
            reason = f"High-confidence contradiction found"
        elif contradiction_score > support_score and n_contradicts > 0:
            prediction = 0
            reason = f"Contradictions ({n_contradicts}) outweigh supports ({n_supports})"
        else:
            prediction = 1
            reason = f"No strong contradictions found"
        
        details = {
            'reason': reason,
            'verdicts': verdicts,
            'counts': dict(verdict_counts),
            'contradiction_score': contradiction_score,
            'support_score': support_score
        }
        
        return prediction, details


# ============================================================================
# Full Verification Pipeline
# ============================================================================

class VerificationPipeline:
    """
    Complete verification pipeline combining all components.
    """
    
    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        embedder,  # ChunkEmbedder instance
        config: Optional[VerifierConfig] = None
    ):
        """
        Initialize the verification pipeline.
        
        Args:
            chunks: Pre-chunked book text
            embedder: ChunkEmbedder with pre-computed embeddings
            config: Pipeline configuration
        """
        self.config = config or DEFAULT_VERIFIER_CONFIG
        self.chunks = chunks
        self.embedder = embedder
        
        # Initialize LLM (shared across components)
        print("Initializing verification pipeline...")
        self.llm = LlamaLLM(config)
        
        # Initialize components
        self.claim_extractor = ClaimExtractor(self.llm, config)
        self.query_generator = QueryGenerator(self.llm)
        self.retriever = HybridRetriever(chunks, embedder, config)
        self.reranker = Reranker(self.llm, config)
        self.verifier = ClaimVerifier(self.llm, config)
        self.aggregator = Aggregator(config)
        
        print("Pipeline initialized!")
    
    def verify_backstory(
        self,
        backstory: str,
        character: str,
        book_name: str,
        verbose: bool = True
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Verify a backstory against the book.
        
        Args:
            backstory: The backstory text to verify
            character: Character name
            book_name: Book name
            verbose: Print progress
            
        Returns:
            (prediction, details) where prediction is 0 (contradict) or 1 (consistent)
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Verifying backstory for: {character} ({book_name})")
            print(f"{'='*60}")
        
        # Step 1: Extract claims
        if verbose:
            print("\n1. Extracting claims...")
        claims = self.claim_extractor.extract_claims(backstory, character, book_name)
        if verbose:
            print(f"   Extracted {len(claims)} claims")
            for i, c in enumerate(claims):
                print(f"   [{i+1}] {c[:80]}...")
        
        if not claims:
            return 1, {'reason': 'No claims could be extracted', 'claims': []}
        
        # Step 2-4: For each claim, retrieve and verify
        # Use parallel processing if configured (be careful with GPU memory)
        if self.config.parallel_claims and len(claims) > 1:
            verification_results = self._process_claims_parallel(
                claims, character, book_name, verbose
            )
        else:
            verification_results = self._process_claims_sequential(
                claims, character, book_name, verbose
            )
        
        # Step 5: Aggregate
        if verbose:
            print(f"\n5. Aggregating results...")
            mem = self.llm.get_memory_usage()
            if mem.get('device') == 'cuda':
                print(f"   GPU Memory: {mem['allocated_gb']:.1f} GB / {mem['total_gb']:.1f} GB")
        
        prediction, details = self.aggregator.aggregate(verification_results)
        
        if verbose:
            print(f"   Final prediction: {prediction} ({'consistent' if prediction == 1 else 'contradict'})")
            print(f"   Reason: {details['reason']}")
        
        # Add claims to details
        details['claims'] = claims
        details['backstory'] = backstory
        details['character'] = character
        details['book_name'] = book_name
        
        return prediction, details
    
    def _process_claims_sequential(
        self,
        claims: List[str],
        character: str,
        book_name: str,
        verbose: bool
    ) -> List[VerificationResult]:
        """Process claims one by one (safer for GPU memory)."""
        verification_results = []
        
        for i, claim in enumerate(claims):
            result = self._process_single_claim(
                claim, i, len(claims), character, book_name, verbose
            )
            verification_results.append(result)
        
        return verification_results
    
    def _process_claims_parallel(
        self,
        claims: List[str],
        character: str,
        book_name: str,
        verbose: bool
    ) -> List[VerificationResult]:
        """
        Process claims in parallel using ThreadPoolExecutor.
        Note: LLM calls are still sequential due to GPU, but retrieval can be parallel.
        """
        if verbose:
            print(f"\n   Using parallel processing with {self.config.max_workers} workers")
        
        # For now, parallel processing only helps with retrieval
        # LLM inference is still the bottleneck and must be sequential
        # True parallelization would need multiple GPUs or CPU offloading
        
        verification_results = []
        
        # Parallelize retrieval (CPU-bound BM25 and embedding search)
        all_evidence = {}
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {}
            for i, claim in enumerate(claims):
                # Generate queries first (needs LLM, so sequential)
                queries = self.query_generator.generate_queries(claim, character, book_name)
                
                # Submit retrieval tasks
                future = executor.submit(self._retrieve_evidence, claim, queries)
                futures[future] = (i, claim, queries)
            
            # Collect results
            for future in as_completed(futures):
                i, claim, queries = futures[future]
                evidence = future.result()
                all_evidence[i] = (claim, evidence)
        
        # Process in order (reranking and verification need LLM)
        for i in range(len(claims)):
            claim, evidence = all_evidence[i]
            
            if verbose:
                print(f"\n   Processing claim {i+1}/{len(claims)}: rerank + verify...")
            
            # Rerank (needs LLM)
            reranked = self.reranker.rerank(claim, evidence)
            
            # Verify (needs LLM)
            result = self.verifier.verify_claim(claim, reranked, character, book_name)
            verification_results.append(result)
            
            if verbose:
                print(f"   Verdict: {result.verdict} (confidence: {result.confidence:.2f})")
        
        return verification_results
    
    def _retrieve_evidence(
        self,
        claim: str,
        queries: List[str]
    ) -> List[Dict[str, Any]]:
        """Retrieve and deduplicate evidence for a claim."""
        all_evidence = []
        for query in queries:
            evidence = self.retriever.search(query)
            all_evidence.extend(evidence)
        
        # Deduplicate
        seen_contents = set()
        unique_evidence = []
        for e in all_evidence:
            content = e.get('content', '')[:100]
            if content not in seen_contents:
                seen_contents.add(content)
                unique_evidence.append(e)
        
        return unique_evidence
    
    def _process_single_claim(
        self,
        claim: str,
        index: int,
        total: int,
        character: str,
        book_name: str,
        verbose: bool
    ) -> VerificationResult:
        """Process a single claim through the pipeline."""
        if verbose:
            print(f"\n2-4. Processing claim {index+1}/{total}...")
        
        # Generate queries
        queries = self.query_generator.generate_queries(claim, character, book_name)
        if verbose:
            print(f"   Generated {len(queries)} queries")
        
        # Retrieve evidence
        unique_evidence = self._retrieve_evidence(claim, queries)
        if verbose:
            print(f"   Retrieved {len(unique_evidence)} unique evidence chunks")
        
        # Rerank
        reranked = self.reranker.rerank(claim, unique_evidence)
        if verbose:
            print(f"   Reranked to top {len(reranked)} chunks")
        
        # Verify
        result = self.verifier.verify_claim(claim, reranked, character, book_name)
        
        if verbose:
            print(f"   Verdict: {result.verdict} (confidence: {result.confidence:.2f})")
            print(f"   Reasoning: {result.reasoning[:100]}...")
        
        return result


# ============================================================================
# Evaluator
# ============================================================================

class Evaluator:
    """
    Evaluates pipeline accuracy on labeled data.
    """
    
    def __init__(self, pipeline: VerificationPipeline):
        self.pipeline = pipeline
        self.results = []
    
    def evaluate_sample(
        self,
        sample: Dict[str, Any],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a single sample.
        
        Args:
            sample: Dict with keys: id, book_name, char, content, label
            verbose: Print progress
            
        Returns:
            Evaluation result dict
        """
        # Map label
        label_map = {'consistent': 1, 'contradict': 0}
        true_label = label_map.get(sample['label'], -1)
        
        # Run prediction
        prediction, details = self.pipeline.verify_backstory(
            backstory=sample['content'],
            character=sample['char'],
            book_name=sample['book_name'],
            verbose=verbose
        )
        
        # Compare
        correct = prediction == true_label
        
        result = {
            'id': sample['id'],
            'character': sample['char'],
            'book_name': sample['book_name'],
            'true_label': true_label,
            'true_label_str': sample['label'],
            'prediction': prediction,
            'prediction_str': 'consistent' if prediction == 1 else 'contradict',
            'correct': correct,
            'details': details
        }
        
        self.results.append(result)
        
        if verbose:
            status = "✓" if correct else "✗"
            print(f"\n{status} Sample {sample['id']}: "
                  f"True={sample['label']}, Pred={'consistent' if prediction == 1 else 'contradict'}")
        
        return result
    
    def evaluate_dataset(
        self,
        samples: List[Dict[str, Any]],
        max_samples: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate multiple samples.
        
        Args:
            samples: List of sample dicts
            max_samples: Max samples to evaluate (None for all)
            verbose: Print progress
            
        Returns:
            Summary statistics
        """
        if max_samples:
            samples = samples[:max_samples]
        
        print(f"\n{'='*60}")
        print(f"Evaluating {len(samples)} samples")
        print(f"{'='*60}")
        
        for i, sample in enumerate(samples):
            print(f"\n[{i+1}/{len(samples)}] Processing sample {sample['id']}...")
            self.evaluate_sample(sample, verbose=verbose)
        
        # Calculate metrics
        correct = sum(1 for r in self.results if r['correct'])
        total = len(self.results)
        accuracy = correct / total if total > 0 else 0
        
        # Per-class accuracy
        consistent_samples = [r for r in self.results if r['true_label'] == 1]
        contradict_samples = [r for r in self.results if r['true_label'] == 0]
        
        consistent_acc = (
            sum(1 for r in consistent_samples if r['correct']) / len(consistent_samples)
            if consistent_samples else 0
        )
        contradict_acc = (
            sum(1 for r in contradict_samples if r['correct']) / len(contradict_samples)
            if contradict_samples else 0
        )
        
        summary = {
            'total_samples': total,
            'correct': correct,
            'accuracy': accuracy,
            'consistent_accuracy': consistent_acc,
            'contradict_accuracy': contradict_acc,
            'results': self.results
        }
        
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total samples: {total}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Consistent class accuracy: {consistent_acc:.2%}")
        print(f"Contradict class accuracy: {contradict_acc:.2%}")
        
        return summary


# ============================================================================
# Main / Test
# ============================================================================

if __name__ == "__main__":
    print("Verifier module loaded. Run tests via pytest.")
