# CLAUDE.md - Project Reference for KDSH 2026 Track A

## Project Goal
Given (novel text, hypothetical backstory), predict label y ∈ {0, 1}:
- **y=1**: Backstory is CONSISTENT with the novel
- **y=0**: Backstory CONTRADICTS the novel

Output: `results.csv` with columns `[id, prediction]`

## Critical Constraints
- **MUST use Docker** - All code runs in Docker environment, never on host machine
- **MUST use Pathway** - Required for ingestion + indexing + retrieval orchestration
- **Reproducible**: `python -m pipeline.run_eval_fast --input_dir ... --out results.csv`
- **GPU**: NVIDIA GeForce RTX 5060 Laptop GPU if available

---

## Pipeline Architecture

```
Backstory → Claim Extraction → Hybrid Retrieval → LLM Verification → Aggregation → Prediction
     ↓              ↓                  ↓                  ↓                ↓
  (input)     (5 claims)      (BM25 + Vector)      (Groq API)      (One error = 0)
```

### Flow
1. **Ingest**: Load novels/backstories via Pathway tables
2. **Chunk**: Split novel into 400-token overlapping chunks
3. **Embed**: Generate vector embeddings (nomic-embed-text-v1.5)
4. **Extract**: LLM extracts 5 verifiable claims from backstory
5. **Retrieve**: Hybrid search (BM25 + vector) with RRF fusion per claim
6. **Verify**: LLM checks each claim → SUPPORTS/CONTRADICTS/UNCLEAR
7. **Aggregate**: Any high-confidence contradiction → predict 0
8. **Output**: Save results + logs

---

## Key Files

| File | Purpose |
|------|---------|
| `pipeline/__init__.py` | Package exports |
| `pipeline/loader.py` | Load CSV + book texts into Pathway tables |
| `pipeline/chunker.py` | Chunk text (400 tokens, 100 overlap front+back) |
| `pipeline/embedder.py` | Nomic embeddings + vector search |
| `pipeline/verifier_fast.py` | **Main pipeline**: Groq LLM, retrieval, verification, aggregation |
| `pipeline/run_eval_fast.py` | Evaluation runner with caching |
| `Dockerfile` | Docker config (CUDA 12.8, PyTorch nightly) |
| `docker-compose.yml` | Docker Compose config |

---

## Configuration (FastVerifierConfig)

```python
@dataclass
class FastVerifierConfig:
    # LLM settings (Groq API)
    llm_model: str = "llama-3.1-8b-instant"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 800
    
    # Retrieval settings
    top_k_retrieval: int = 15      # Chunks retrieved per query
    top_k_final: int = 8           # Chunks sent to LLM
    bm25_weight: float = 0.5       # BM25 weight in RRF fusion
    vector_weight: float = 0.5     # Vector weight in RRF fusion
    
    # Verification settings
    max_claims: int = 5            # Claims per backstory
    
    # Retry settings
    max_retries: int = 3
    parallel_workers: int = 10     # Concurrent Groq requests
```

---

## Key Design Decisions

### Chunking
- **Size**: 400 tokens per chunk
- **Overlap**: 100 front + 100 back (200 total)
- **Chapter detection**: Extracts chapter metadata for context

### Embedding
- **Model**: `nomic-ai/nomic-embed-text-v1.5` (768-dim)
- **Matryoshka**: Can truncate to 64/128/256/512 dims
- **Task prefixes**: `search_document:` for chunks, `search_query:` for queries

### Retrieval
- **Hybrid**: BM25 + Vector search
- **Fusion**: Reciprocal Rank Fusion (RRF) with k=60
- **Query expansion**: Extracts character names, relationships, biographical terms
- **Multi-query**: Primary query (weight 1.0) + expanded queries (weight 0.5)

### Verification
- **LLM**: Groq API with Llama 3.1 8B Instant (~1s per call, free tier)
- **Verdicts**: SUPPORTS, CONTRADICTS, UNCLEAR
- **Priority**: Look for contradictions first, then supports, then unclear
- **Detective approach**: "Precise but not assumptional"

### Aggregation
- **Policy**: One factual error = CONTRADICT (hard-kill)
- **Strong contradiction**: confidence >= 0.75 + valid citation
- **Decent contradiction**: confidence >= 0.6
- **Weak contradiction**: confidence 0.5-0.6 with citation > 30 chars
- **False positive filtering**: Filters patterns like "same person", "no mention of"

---

## Data Locations

```
Dataset/
├── train.csv          # Training data (id, book_name, char, caption, content, label)
├── test.csv           # Test data (no label column)
└── Books/
    ├── In search of the castaways.txt
    └── The Count of Monte Cristo.txt
```

### CSV Schema
- `id`: Sample identifier
- `book_name`: Source novel name
- `char`: Character name
- `caption`: Brief description
- `content`: Backstory text to verify
- `label`: "consistent" or "contradict" (train only)

---

## Environment Variables

```bash
GROQ_API_KEY=your-api-key-here  # Required - get from https://console.groq.com/keys
```

---

## Running Commands

```bash
# Build and run Docker (no GPU required)
docker-compose build
docker-compose run --rm pipeline python -m pipeline.run_eval_fast --max-samples 5

# With GPU
docker-compose run --rm pipeline-gpu python -m pipeline.run_eval_fast --max-samples 5

# Inside container - quick test
python -m pipeline.run_eval_fast --max-samples 5

# Inside container - full eval
python -m pipeline.run_eval_fast --verbose

# Generate submission file (test.csv)
python -m pipeline.run_eval_fast --test --out results.csv

# Skip cache (regenerate embeddings)
python -m pipeline.run_eval_fast --no-cache
```

---

## Performance Metrics

**Best achieved**: 66.2% overall accuracy
- Consistent accuracy: 68.6%
- Contradict accuracy: 62.1%

**Goal**: 60%+ on BOTH consistent and contradict classes ✅ ACHIEVED

---

## Caching

Chunks and embeddings are cached in `.cache/`:
- `chunks.pkl` - Pickled chunk data
- `embeddings.npy` - NumPy embedding matrix

Use `--no-cache` to regenerate.

---

## Key Classes

### FastClaimExtractor
Extracts 5 verifiable factual claims from backstory focusing on:
- Dates, years, time periods
- Names of people mentioned
- Relationships
- Specific events
- Locations

### FastHybridRetriever
Hybrid search combining:
- BM25 (exact term matching)
- Vector search (semantic similarity)
- Query expansion for character names and biographical terms

### FastClaimVerifier
Verifies claims with prompts designed for:
- Detecting contradictions (different dates, facts, opposite statements)
- Confirming supports (same facts confirmed)
- Marking unclear (topic not addressed)

### FastAggregator
Aggregates verdicts with false positive filtering:
- Filters "same person" comparisons
- Filters "no mention of" patterns
- Requires citations for weak contradictions

---

## Testing

```bash
# Run tests
pytest tests/ -v

# Run specific test file
pytest tests/test_chunker.py -v
```

---

## Common Issues

1. **Pathway warning on Windows**: Expected - Pathway runs properly only in Docker (Linux)
2. **CUDA errors**: Ensure `--gpus all` flag in Docker run
3. **Groq rate limits**: Pipeline has retry logic with backoff
4. **Embedding cache mismatch**: Use `--no-cache` if chunks changed

---

## Future Improvements

- [ ] Tune aggregation thresholds based on more data
- [ ] Add more books to dataset
- [ ] Experiment with different embedding dimensions
- [ ] Try different LLM models (Llama 3.1 70B for better accuracy)
- [ ] Implement confidence calibration
