# KDSH 2026 Track A - Fictional Character Backstory Verification

A RAG-based pipeline to verify fictional character backstories against their source novels using hybrid retrieval and LLM verification.

## 🚀 Quick Start for Judges

### Prerequisites
- **Docker** & Docker Compose installed
- **Groq API key** (free at https://console.groq.com/keys)

### Reproduce Results in 3 Steps

```bash
# Step 1: Set your Groq API key
export GROQ_API_KEY='your-api-key-here'   # Linux/Mac
$env:GROQ_API_KEY='your-api-key-here'     # Windows PowerShell

# Step 2: Build the Docker image (first time only, ~5-10 min)
docker-compose build

# Step 3: Generate submission on test.csv
docker-compose run --rm pipeline python -m pipeline.run_eval_fast --test --out results.csv
```

This produces `results.csv` with columns: `id`, `prediction`, `rationale`

### Quick Validation (Optional)
```bash
# Test on 5 samples from train.csv to verify setup (~1 min)
docker-compose run --rm pipeline python -m pipeline.run_eval_fast --max-samples 5
```

---

## Problem Statement

Given a character's name, book title, and hypothetical backstory, predict whether the backstory is:
- **Consistent (1)**: The backstory aligns with the book's content
- **Contradict (0)**: The backstory contains factual errors

## Pipeline Architecture

```
Backstory → Claim Extraction → Hybrid Retrieval → LLM Verification → Aggregation → Prediction
```

### Components

1. **Chunker** (`pipeline/chunker.py`): Splits book text into overlapping 400-token chunks with chapter detection
2. **Embedder** (`pipeline/embedder.py`): Uses `nomic-ai/nomic-embed-text-v1.5` for semantic embeddings
3. **Retriever** (`pipeline/verifier_fast.py`): Hybrid BM25 + vector search with RRF fusion and query expansion
4. **Verifier** (`pipeline/verifier_fast.py`): Groq API (Llama 3.1 8B) for claim verification
5. **Aggregator** (`pipeline/verifier_fast.py`): One factual error = Contradict

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Groq API key (free at https://console.groq.com/keys)
- **Optional**: NVIDIA GPU with Docker GPU support (for faster embeddings, but works on CPU)

> **Quick Setup Check**: Run `bash test_setup.sh` to verify all prerequisites are met.

### Running with Docker

**Step 1: Set your Groq API key**

```bash
# Linux/Mac
export GROQ_API_KEY='your-api-key-here'

# Windows PowerShell
$env:GROQ_API_KEY='your-api-key-here'

# Windows CMD
set GROQ_API_KEY=your-api-key-here
```

**Step 2: Build the Docker image**

```bash
docker-compose build
```

**Step 3: Run evaluation**

```bash
# Quick test (5 samples)
docker-compose run --rm pipeline python -m pipeline.run_eval_fast --max-samples 5

# Full evaluation on train.csv
docker-compose run --rm pipeline python -m pipeline.run_eval_fast --verbose

# Or get an interactive shell inside container
docker-compose run --rm pipeline bash
# Then inside container:
python -m pipeline.run_eval_fast --max-samples 5
```

**With GPU (if available):**

```bash
# Use the GPU-enabled service
docker-compose run --rm pipeline-gpu python -m pipeline.run_eval_fast --max-samples 5
```

**Alternative: Manual Docker run**

```bash
# Build
docker build -t kdsh-pipeline .

# Run with GPU (if available)
docker run --gpus all -e GROQ_API_KEY=$GROQ_API_KEY kdsh-pipeline python -m pipeline.run_eval_fast --max-samples 5

# Run without GPU
docker run -e GROQ_API_KEY=$GROQ_API_KEY kdsh-pipeline python -m pipeline.run_eval_fast --max-samples 5
```

### Command Options

```bash
--max-samples N     # Number of samples to evaluate (default: all)
--verbose, -v       # Detailed output for each sample
--output FILE       # Output JSON file (default: eval_results_fast.json)
--out FILE          # Output CSV for submission (default: results.csv)
--input-dir DIR     # Custom dataset directory
--test              # Run on test.csv instead of train.csv
--no-cache          # Regenerate chunks and embeddings
```

### Submission (Final Prediction)

To generate the final `results.csv` submission file:

```bash
# Run on test.csv to generate results.csv
docker-compose run --rm pipeline python -m pipeline.run_eval_fast --test --out results.csv

# Or with custom input directory
docker-compose run --rm pipeline python -m pipeline.run_eval_fast --test --input-dir /app/Dataset --out results.csv
```

The output `results.csv` contains:

| Column | Description |
|--------|-------------|
| `id` | Sample ID from test.csv |
| `prediction` | 1 (consistent) or 0 (contradict) |
| `rationale` | A 3-sentence explanation with cited evidence |

**Example output:**
```csv
id,prediction,rationale
95,1,"The backstory for Edmond Dantès is consistent with the source text. Analysis of 4 claims found 3 supported, 0 contradicted, and 1 unclear based on retrieved passages. No significant contradictions were found in the verified claims."
96,0,"The backstory for Villefort is contradictory with the source text. Analysis of 3 claims found 1 supported, 1 contradicted, and 1 unclear based on retrieved passages. Key evidence: ""The prisoner was taken to the Chateau d'If, not to Paris as claimed."""
```

> **Note:** This pipeline requires Docker to run. Pathway and other dependencies are configured specifically for the Docker environment.

## Project Structure

```
├── pipeline/
│   ├── __init__.py         # Package exports
│   ├── loader.py           # Data loading (CSV + books)
│   ├── chunker.py          # Text chunking with overlap
│   ├── embedder.py         # Nomic embedding model
│   ├── verifier_fast.py    # Main verification pipeline
│   └── run_eval_fast.py    # Evaluation runner
├── Dataset/
│   ├── train.csv           # Training data
│   ├── test.csv            # Test data
│   └── Books/              # Source novel texts
├── tests/                  # Unit tests
├── requirements.txt        # Dependencies
├── Dockerfile             # Docker configuration
└── docker-compose.yml     # Docker Compose config
```

## Configuration

Key parameters in `FastVerifierConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_claims` | 5 | Claims extracted per backstory |
| `top_k_retrieval` | 15 | Chunks retrieved per query |
| `top_k_final` | 8 | Chunks sent to LLM |
| `bm25_weight` | 0.5 | BM25 weight in RRF fusion |
| `vector_weight` | 0.5 | Vector weight in RRF fusion |

## Key Design Decisions

- **Chunking**: 400 tokens with 100-token overlap (front + back) for context preservation
- **Embedding**: Nomic v1.5 (768-dim) with matryoshka support for efficiency
- **Retrieval**: Hybrid (BM25 + vector) with RRF fusion for robust retrieval
- **Verification**: Groq API for fast, free LLM inference (~1s/call)
- **Aggregation**: Single high-confidence contradiction triggers "Contradict"

## Results

Achieves **66%+ accuracy** on both consistent and contradict classes:
- Consistent accuracy: ~68%
- Contradict accuracy: ~62%

### Expected Runtime
- **Build time**: ~5-10 minutes (first time only, dependencies cached)
- **Test set (60 samples)**: ~5-8 minutes on CPU
- **Per sample**: ~5-8 seconds (includes LLM API calls)

## Troubleshooting

### GROQ_API_KEY not set
**Error**: `ValueError: GROQ_API_KEY environment variable not set!`

**Solution**: Make sure to export/set the API key before running Docker:
```bash
# Linux/Mac
export GROQ_API_KEY='your-key-here'

# Windows PowerShell
$env:GROQ_API_KEY='your-key-here'
```

### GPU not available
**Warning**: `CUDA not available, using CPU`

**Solution**: This is OK! The pipeline works on CPU, just slower for embeddings. If you have an NVIDIA GPU:
1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
2. Make sure `--gpus all` flag is used in docker run command

### Pathway warnings on Windows
**Warning**: `This is not the real Pathway package...`

**Solution**: This is expected - Pathway is designed for Linux/Docker. The pipeline uses workarounds for Windows development but runs properly in Docker.

### Dataset not found
**Error**: `FileNotFoundError: Dataset directory not found`

**Solution**: Make sure the `Dataset/` folder with `train.csv`, `test.csv`, and `Books/` is in the project root directory.

## License

MIT License
