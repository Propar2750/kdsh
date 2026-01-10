# KDSH Track A — Evaluation Report

## Overview

This report documents the **Fast Verification Pipeline** for the KDSH 2026 Track A competition. The goal is to verify whether a hypothetical character backstory is **consistent** or **contradicts** the source novel text.

**Test Run Results:** 2/2 samples correct (100% accuracy on validation subset)

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: (Novel, Backstory)                     │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. CLAIM EXTRACTION (FastClaimExtractor)                        │
│     • Extract 5 verifiable facts from backstory                  │
│     • Focus on: dates, names, relationships, events, locations   │
│     • LLM: Groq API (llama-3.1-8b-instant)                      │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. EVIDENCE RETRIEVAL (FastHybridRetriever)                     │
│     • Hybrid search: BM25 (40%) + Vector (60%)                   │
│     • Reciprocal Rank Fusion (RRF) for score combination         │
│     • Returns top-15 candidate chunks, uses top-8 for verify     │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. CLAIM VERIFICATION (FastClaimVerifier) — PARALLEL            │
│     • Each claim verified against retrieved evidence             │
│     • Citation-based checking with confidence scores             │
│     • Verdicts: SUPPORTS / CONTRADICTS / UNCLEAR                 │
│     • Parallel execution with ThreadPoolExecutor                 │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. AGGREGATION (FastAggregator)                                 │
│     • Any high-confidence contradiction (≥0.6) → CONTRADICT      │
│     • Citation-backed contradiction → CONTRADICT                 │
│     • Multiple contradictions (≥2) → CONTRADICT                  │
│     • Otherwise → CONSISTENT                                     │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                 OUTPUT: Prediction (0 or 1)                      │
│                 0 = Contradict, 1 = Consistent                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Components

### 1. Data Loading (`loader.py`)
- Loads CSV files with backstories and labels
- Loads full novel texts from `.txt` files
- Joins data into unified story table
- Label encoding: `consistent → 1`, `contradict → 0`

### 2. Chunking (`chunker.py`)
- Token-based chunking: **400 tokens** per chunk
- Overlap: **100 tokens** front + **100 tokens** back (200 total overlap)
- Ensures context preservation across chunk boundaries
- **1,916 chunks** generated from 4 novels

### 3. Embedding (`embedder.py`)
- Model: **nomic-ai/nomic-embed-text-v1.5** (768-dimensional)
- GPU-accelerated (CUDA 12.8) when available
- Matryoshka support for variable-dimension search
- Builds vector index for semantic search

### 4. Verification (`verifier_fast.py`)

#### 4.1 GroqLLM Wrapper
```python
model: "llama-3.1-8b-instant"
temperature: 0.0 (deterministic)
max_tokens: 800
max_retries: 3
```
- Free API tier (~1s per call)
- Automatic retry on rate limits

#### 4.2 Claim Extraction
Extracts **5 specific, verifiable facts** focusing on:
- Dates, years, time periods
- Names of people, places, ships
- Relationships (family, allies, enemies)
- Specific events and outcomes
- Locations where events occurred

#### 4.3 Hybrid Retrieval (BM25 + Vector)
| Component | Weight | Purpose |
|-----------|--------|---------|
| BM25 | 40% | Exact keyword matching |
| Vector | 60% | Semantic similarity |
| RRF k | 60 | Fusion smoothing parameter |

#### 4.4 Claim Verification
Structured output format:
```
VERDICT: [SUPPORTS/CONTRADICTS/UNCLEAR]
CONFIDENCE: [0.0-1.0]
CITATION: [Quote from evidence]
REASONING: [1-2 sentence explanation]
```

**Parsing Strategies:**
1. Regex extraction for structured fields
2. Keyword-based fallback with negation handling
3. Confidence boosting (+0.1) when citation provided

#### 4.5 Aggregation Rules
```
IF high_confidence_contradiction (≥0.6):
    → CONTRADICT
ELIF cited_contradiction (with quote > 20 chars):
    → CONTRADICT  
ELIF contradiction_count ≥ 2:
    → CONTRADICT
ELIF contradiction_score > support_score × 1.2:
    → CONTRADICT
ELSE:
    → CONSISTENT
```

---

## Test Results (2 Samples)

| Sample ID | Character | Book | True Label | Prediction | Result |
|-----------|-----------|------|------------|------------|--------|
| 46 | Thalcave | In Search of the Castaways | consistent | consistent | ✓ |
| 137 | Faria | The Count of Monte Cristo | contradict | contradict | ✓ |

### Sample 137 (Contradict) — Detailed Analysis
- **5 claims extracted** about Abbé Faria
- **1 CONTRADICTION detected** (confidence: 1.0, with citation)
  - Claim about year "1815" contradicted by evidence showing "1811"
- **1 SUPPORT** found
- **3 UNCLEAR** (evidence didn't address those specific claims)
- **Final verdict:** CONTRADICT ✓

### Sample 46 (Consistent) — Detailed Analysis
- **5 claims extracted** about Thalcave
- **All 5 UNCLEAR** (no direct support/contradiction found)
- No contradiction evidence → default to consistent
- **Final verdict:** CONSISTENT ✓

---

## Performance Statistics

| Metric | Value |
|--------|-------|
| Accuracy | 100% (2/2) |
| LLM Calls | 12 total |
| Total Time | 259.3 seconds |
| Avg Time/Call | 21.6 seconds |
| API Errors | 5 (all retried successfully) |

**Note:** High avg time due to network latency and Groq rate limiting. Individual successful calls typically take ~1-2s.

---

## Caching System

The pipeline implements caching for faster subsequent runs:

| Cache | File | Purpose |
|-------|------|---------|
| Chunks | `.cache/chunks.pkl` | Avoids re-chunking novels |
| Embeddings | `.cache/embeddings.npy` | Avoids re-embedding 1916 chunks |

First run: ~60s for chunking + embedding  
Cached run: <1s to load

---

## Configuration (`FastVerifierConfig`)

```python
@dataclass
class FastVerifierConfig:
    # LLM settings
    llm_model: str = "llama-3.1-8b-instant"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 800
    
    # Retrieval settings
    top_k_retrieval: int = 15
    top_k_final: int = 8
    bm25_weight: float = 0.4
    vector_weight: float = 0.6
    
    # Verification settings
    max_claims: int = 5
    max_retries: int = 3
```

---

## Strengths

1. **Citation-based verification** — More reliable than pure classification
2. **Hybrid retrieval** — Combines exact match + semantic understanding
3. **Parallel verification** — 5 claims verified concurrently
4. **Robust parsing** — Multiple fallback strategies for LLM output
5. **Free inference** — Groq API has generous free tier
6. **Caching** — Fast subsequent runs

## Limitations

1. **Small test set** — Only 2 samples evaluated (need full 80)
2. **Network dependency** — Groq API requires internet
3. **Rate limiting** — 5 errors in 12 calls suggests rate limit issues
4. **UNCLEAR bias** — Tends to mark claims as unclear when evidence is ambiguous
5. **No fine-tuning** — Uses base llama-3.1-8b model without domain adaptation

---

## Recommendations

### Immediate
1. **Run full evaluation** (80 test samples) once Docker is available
2. **Add exponential backoff** to reduce API errors
3. **Log all LLM inputs/outputs** for debugging

### Future Improvements
1. **Claim quality scoring** — Filter low-quality claims before verification
2. **Multi-hop retrieval** — For complex claims spanning multiple passages
3. **Ensemble models** — Combine multiple LLMs for robustness
4. **Active learning** — Use mistakes to improve prompts

---

## How to Run

```bash
# Set API key
$env:GROQ_API_KEY = "your-key-here"

# Run evaluation (Docker)
docker-compose run --rm -e GROQ_API_KEY=$env:GROQ_API_KEY pipeline python -m pipeline.run_eval_fast --verbose

# Run with sample limit
docker-compose run --rm -e GROQ_API_KEY=$env:GROQ_API_KEY pipeline python -m pipeline.run_eval_fast --max-samples 10 --verbose

# Clear cache and re-run
docker-compose run --rm -e GROQ_API_KEY=$env:GROQ_API_KEY pipeline python -m pipeline.run_eval_fast --no-cache
```

---

## Files Modified

| File | Changes |
|------|---------|
| `pipeline/verifier_fast.py` | Complete rewrite with citation checking, parallel verification |
| `pipeline/run_eval_fast.py` | Added caching for chunks/embeddings |
| `eval_results_fast.json` | Latest evaluation results |

---

## Conclusion

The Fast Verification Pipeline successfully identifies both consistent and contradictory backstories by:
1. Extracting specific, verifiable claims
2. Retrieving relevant evidence using hybrid search
3. Verifying each claim with citation-based LLM analysis
4. Aggregating results with contradiction-weighted decision logic

The 2-sample test shows 100% accuracy, but full evaluation on 80 samples is needed to validate the approach at scale.

---

*Report generated: January 10, 2026*  
*Pipeline version: verifier_fast.py (citation-based verification)*
