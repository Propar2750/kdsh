# Project: KDSH 2026 — Track A (Pathway pipeline)

## Goal
Given (novel text, hypothetical backstory), predict label y ∈ {0,1}:
y=1 if backstory is consistent with the novel, else 0.
Also produce an evidence rationale internally.

## Constraints
- Must use Pathway meaningfully (ingestion + indexing + retrieval orchestration).
- **ALWAYS use Docker** - All Pathway code must run in Docker environment, never on host machine
- Output: results.csv with columns: [id, prediction]
- Must be reproducible: `python -m pipeline.run --input_dir ... --out results.csv`
- Must use NVIDIA GeForce RTX 5060 Laptop GPU if possible

## Pipeline (high level)
1) Ingest novels/backstories into tables (Pathway) ✅
2) Chunk novel -> chunk_table ✅
3) Embed + vector index ✅
4) Backstory -> atomic claims
5) Retrieve top-k evidence per claim
6) Verify (supports/contradicts/unclear)
7) Aggregate to final label
8) Save results + logs/rationale

## Design decisions
- **Docker**: CUDA 12.8 + PyTorch nightly (cu128) for RTX 5060 Blackwell support
- **Chunk size**: 400 tokens ; overlap: 100 front + 100 back (200 total overlap)
- **Embedding model**: nomic-ai/nomic-embed-text-v1.5 (768-dim, matryoshka support)
- **Reasoning LLM**: Groq API with llama-3.1-8b-instant (free, ~1s per call)
- #claims per backstory: 5
- Contradiction policy: hard-kill (any contradiction → predict 0)
- Retriever: top-k=10 hybrid (BM25 + vector), no LLM rerank
- Verifier: LLM rubric via Groq API
- Aggregator rule: If any claim contradicts → 0, else → 1

## Data contracts
- story_id: string
- novel_text: string
- backstory_text: string
- output schema: id, prediction (0/1)
- Data is in /Dataset/train.csv and Dataset/Books/book_name.txt

## "Definition of done"
- Smoke test runs on 2 samples end-to-end < 2 minutes
- Deterministic output given fixed seed
- Clear logging: per-claim retrieved passages + verdict

---

## Development Setup

### Prerequisites
- Docker with NVIDIA Container Toolkit
- NVIDIA GPU (tested on RTX 5060 Laptop GPU)

**CRITICAL**: All development and testing MUST be done inside Docker containers. Never run Pathway code directly on the host machine.

### Commands
```bash
# Build Docker image (GPU-enabled)
docker-compose build

# Run all tests
docker-compose run --rm pipeline python -m pytest tests/ -v

# Run tests excluding slow ones
docker-compose run --rm pipeline python -m pytest tests/ -v -k "not slow"

# Verify GPU access
docker-compose run --rm pipeline python -c "import torch; print(torch.cuda.get_device_name(0))"

# Interactive shell
docker-compose run --rm pipeline bash
```

### Project Structure
```
pipeline/
├── __init__.py      # Exports all modules
├── loader.py        # CSV/book loading → Pathway tables
├── chunker.py       # Token-based chunking with overlap
├── embedder.py      # nomic-embed + vector index
tests/
├── test_loader.py   # 7 tests
├── test_chunker.py  # 20 tests  
├── test_embedder.py # 18 tests (2 slow)
├── test_integration.py # 1 test
Dataset/
├── train.csv        # Training data with labels
├── test.csv         # Test data
├── Books/           # Novel text files
```

### Implemented Modules

**loader.py** - Data ingestion
- `load_csv_to_pathway(csv_path)` → Pathway table
- `load_books(dataset_dir)` → dict[book_name, text]
- `create_story_table(csv_path, books)` → Pathway table with novel_text joined

**chunker.py** - Text chunking
- `chunk_text(text, config)` → list of chunk dicts
- `chunk_books(books, config)` → all chunks from all books
- `BookChunker` class for pipeline integration
- Chapter detection for metadata

**embedder.py** - Embeddings & retrieval
- `NomicEmbedder` - Wraps sentence-transformers model
- `ChunkEmbedder` - Embeds chunks, provides search()
- `PathwayVectorIndex` - Pathway table with embeddings

**verifier.py** - Full verification pipeline (SLOW - ~65 LLM calls per sample)
- `LlamaLLM` - Llama wrapper with batching + memory optimization
- `ClaimExtractor` - Extract atomic facts from backstories
- `QueryGenerator` - Generate search queries per claim
- `BM25Retriever` - Sparse lexical search
- `HybridRetriever` - Combines BM25 + vector search with score fusion
- `Reranker` - LLM-based relevance reranking (SLOW!)
- `ClaimVerifier` - Verify claims against evidence
- `Aggregator` - Combine verdicts into final prediction
- `VerificationPipeline` - Full end-to-end pipeline
- `Evaluator` - Evaluate accuracy on labeled data

**verifier_fast.py** - FAST verification pipeline (~6 LLM calls per sample) ⭐
- `FastLlamaLLM` - Optimized LLM wrapper with timing stats
- `FastClaimExtractor` - Single LLM call for claim extraction
- `SimpleBM25` - Pure Python BM25 (no external deps)
- `FastHybridRetriever` - Score-based fusion (NO LLM reranking)
- `FastClaimVerifier` - Simple verdict extraction
- `FastAggregator` - Any-contradiction policy
- `FastVerificationPipeline` - 10x faster end-to-end pipeline
- `FastEvaluator` - Quick evaluation with stats

**run_eval.py** - Original evaluation runner (SLOW)
**run_eval_fast.py** - FAST evaluation runner ⭐
- `python -m pipeline.run_eval_fast --max-samples 5`

### Tech Stack
- **Pathway**: 0.28.0 (real tables, not stubs)
- **PyTorch**: 2.11.0.dev+cu128 (Blackwell support)
- **CUDA**: 12.8
- **Embedding**: nomic-ai/nomic-embed-text-v1.5
- **LLM**: Groq API with llama-3.1-8b-instant (free, fast cloud inference)

### Environment Variables
```bash
GROQ_API_KEY=your-key-here  # Get free key at https://console.groq.com/keys
```

---

## Performance & Optimization Notes

### Docker Build Speed
- **Problem**: Full rebuild takes ~8+ hours (downloads PyTorch nightly + all packages)
- **Solution**: Layer caching with `requirements.txt`
  - requirements.txt is copied first and cached
  - Subsequent builds only rebuild if requirements change
  - Code changes don't trigger full package reinstall
- **Tip**: Use `docker-compose build --no-cache` only when absolutely needed

### GPU Memory (8GB VRAM Analysis)
- **Llama-3.1-8B @ float16**: ~16GB → needs offloading to shared memory
- **With device_map="auto"**: Automatically splits between GPU + CPU RAM
- **8-bit quantization**: Reduces to ~8GB (enable with `use_8bit=True`)
- **Current setup**: Works with 8GB dedicated + 16GB shared memory
- **Recommendation**: 8-bit quantization recommended for faster inference

### Pipeline Efficiency - WHY IT WAS SLOW

**Original pipeline (verifier.py) - ~65+ LLM calls per sample:**
1. Claim extraction: 1 LLM call
2. Per claim (x5 claims):
   - Query generation: 1 LLM call
   - Reranking: ~10 LLM calls (one per retrieved chunk!)
   - Verification: 1 LLM call
   - Subtotal: ~12 LLM calls per claim
3. Total: 1 + (5 × 12) = **~61 LLM calls per sample**
4. At ~10s per call = **~10 minutes per sample** = **12 hours for 72 samples**

**FAST pipeline (verifier_fast.py) - ~6 LLM calls per sample:**
1. Claim extraction: 1 LLM call
2. Per claim (x5 claims):
   - Query: Use claim directly (0 LLM calls)
   - Reranking: Score fusion only (0 LLM calls)
   - Verification: 1 LLM call
3. Total: 1 + 5 = **6 LLM calls per sample**
4. At ~10s per call = **~1 minute per sample** = **~1 hour for 72 samples**

### Config Options (VerifierConfig)
```python
VerifierConfig(
    llm_batch_size=4,       # Increase for faster batching
    use_8bit=True,          # Enable 8-bit quantization (saves ~50% VRAM)
    parallel_claims=True,   # Parallel retrieval (CPU-bound)
    max_workers=2,          # Thread workers for retrieval
    top_k_per_query=5,      # Reduce for speed, increase for recall
    top_k_reranked=3,       # Chunks after reranking
)
```

---

## TODO (Next Steps)
- [x] Implement claim extraction from backstories ✅
- [x] Add retrieval pipeline (top-k per claim) ✅
- [x] Add verification module (NLI or LLM-based) ✅
- [x] Add aggregation logic ✅
- [x] Create `pipeline/run_eval.py` evaluation runner ✅
- [ ] Run end-to-end smoke test on 2-5 samples
- [ ] Tune aggregation thresholds based on results
- [ ] Create final `pipeline/run.py` entry point for submission
