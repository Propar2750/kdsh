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
- #claims per backstory: ___
- Contradiction policy: (hard-kill / weighted / threshold) -> ___
- Retriever: (top-k=___) ; rerank: (none / LLM / cross-encoder)
- Verifier: (NLI model / LLM rubric) -> ___
- Aggregator rule: ___ (describe in 2 lines)

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

### Tech Stack
- **Pathway**: 0.28.0 (real tables, not stubs)
- **PyTorch**: 2.11.0.dev+cu128 (Blackwell support)
- **CUDA**: 12.8
- **Embedding**: nomic-ai/nomic-embed-text-v1.5

---

## TODO (Next Steps)
- [ ] Implement claim extraction from backstories
- [ ] Add retrieval pipeline (top-k per claim)
- [ ] Add verification module (NLI or LLM-based)
- [ ] Add aggregation logic
- [ ] Create `pipeline/run.py` entry point
- [ ] End-to-end smoke test

## General info
- **ALWAYS use Docker** for all Pathway operations - use `docker-compose run --rm pipeline` prefix for all commands
- Generally make all changes in /pipeline or /tests
- Test immediately after implementing any feature: `docker-compose run --rm pipeline python -m pytest tests/ -v`
