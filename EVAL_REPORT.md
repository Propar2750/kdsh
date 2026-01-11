# KDSH Track A — Team Status Report

## TL;DR

**Built a working verification pipeline. Current accuracy: 65% (52/80).**

We have a complete end-to-end pipeline that:
1. Extracts claims from backstories
2. Retrieves evidence from novels using hybrid search
3. Verifies claims via Groq API (free LLM)
4. Aggregates to final prediction

**Main issues to fix:** Too many false positives (saying "contradict" when it's actually "consistent").

---

## What I Built

### Pipeline Overview
```
Backstory → Extract 5 Claims → Retrieve Evidence → Verify Each Claim → Aggregate → Prediction
```

### Key Files
| File | What it does |
|------|--------------|
| `pipeline/loader.py` | Loads CSV + book text files |
| `pipeline/chunker.py` | Splits novels into 400-token chunks |
| `pipeline/embedder.py` | Creates embeddings using nomic-embed |
| `pipeline/verifier_fast.py` | **Main logic** - claim extraction, verification, aggregation |
| `pipeline/run_eval_fast.py` | Runs evaluation with caching |

### Tech Stack
- **LLM:** Groq API with `llama-3.1-8b-instant` (free tier)
- **Embeddings:** `nomic-ai/nomic-embed-text-v1.5` (768-dim)
- **Search:** Hybrid BM25 (40%) + Vector (60%) with RRF fusion
- **Runtime:** Docker with CUDA 12.8

---

## Full Evaluation Results (80 samples)

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **65% (52/80)** |
| Consistent samples | 51 total |
| Contradict samples | 29 total |
| LLM Calls | 461 |
| Total Runtime | ~17 hours |
| API Errors | 129 (retried with backoff) |

### Error Analysis

**28 wrong predictions:**
- ~18 **False Positives** (predicted contradict, was actually consistent)
- ~10 **False Negatives** (predicted consistent, was actually contradict)

**Main Problem:** Pipeline is too aggressive at finding "contradictions" when there aren't any.

---

## How it Works

### Step 1: Claim Extraction
LLM extracts 5 specific, verifiable facts from the backstory:
```
"Faria was arrested in 1815"
"Thalcave's father was a tribal guide"
```

### Step 2: Evidence Retrieval
For each claim, retrieves top-8 relevant chunks from the novel using:
- BM25 (keyword matching)
- Vector search (semantic similarity)
- Combined with Reciprocal Rank Fusion

### Step 3: Claim Verification
LLM checks each claim against evidence:
```
VERDICT: SUPPORTS / CONTRADICTS / UNCLEAR
CONFIDENCE: 0.0 - 1.0
CITATION: "quote from the book"
REASONING: why
```

### Step 4: Aggregation
```python
if any_high_confidence_contradiction >= 0.6:
    return CONTRADICT
elif contradiction_count >= 2:
    return CONTRADICT
else:
    return CONSISTENT
```

---

## How to Run

```powershell
# Set API key (get free key at https://console.groq.com/keys)
$env:GROQ_API_KEY = "your-key-here"

# Run on 5 samples (quick test)
docker-compose run --rm -e GROQ_API_KEY=$env:GROQ_API_KEY pipeline python -m pipeline.run_eval_fast --max-samples 5 --verbose

# Run full evaluation (takes ~17 hours)
docker-compose run --rm -e GROQ_API_KEY=$env:GROQ_API_KEY pipeline python -m pipeline.run_eval_fast --max-samples 80 --verbose
```

---

## Known Issues

1. **Rate Limiting** - Groq free tier has limits. Added exponential backoff (up to 128s) but still get errors.
2. **False Positives** - Often finds "contradictions" that don't exist.
3. **Slow** - ~17 hours for 80 samples due to rate limiting.
4. **UNCLEAR bias** - Many claims marked "unclear" when evidence is ambiguous.

---

## What Needs Work

### High Priority
1. **Reduce false positives** - Tune aggregation thresholds or improve prompts
2. **Improve claim quality** - Extract more specific, verifiable claims
3. **Better evidence retrieval** - Some contradictions missed due to poor chunk selection

### Nice to Have
- Switch to a local LLM to avoid rate limits
- Add claim filtering (discard vague claims)
- Ensemble multiple models

---

## Caching

Pipeline caches chunks and embeddings in `.cache/`:
- `chunks.pkl` - 1916 chunks from 4 novels
- `embeddings.npy` - Pre-computed vectors

First run: ~60s setup | Cached run: <1s

---

## Questions for Team

1. Should we prioritize precision (fewer false contradictions) or recall (catch more real contradictions)?
2. Is 65% accuracy good enough for submission, or do we need to improve?
3. Should we try a different LLM (local Ollama, or paid API)?

---

*Last updated: January 11, 2026*  
*Full results in: `eval_results_fast.json`*
