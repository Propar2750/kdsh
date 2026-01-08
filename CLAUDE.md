# Project: KDSH 2026 — Track A (Pathway pipeline)

## Goal
Given (novel text, hypothetical backstory), predict label y ∈ {0,1}:
y=1 if backstory is consistent with the novel, else 0.
Also produce an evidence rationale internally.

## Constraints
- Must use Pathway meaningfully (ingestion + indexing + retrieval orchestration).
- Output: results.csv with columns: [id, prediction]
- Must be reproducible: `python -m pipeline.run --input_dir ... --out results.csv`
- Must use NVIDIA GeForce RTX 5060 Laptop GPU if possible

## Pipeline (high level)
1) Ingest novels/backstories into tables (Pathway)
2) Chunk novel -> chunk_table
3) Embed + vector index
4) Backstory -> atomic claims
5) Retrieve top-k evidence per claim
6) Verify (supports/contradicts/unclear)
7) Aggregate to final label
8) Save results + logs/rationale

## Design decisions (YOU decide)
- We are using Docker
- Chunk size: 400 tokens ; overlap: 100 front + 100 back (200 total overlap)
- Embedding model: nomic-ai/nomic-embed-text-v1.5 (768-dim, matryoshka support)
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

## “Definition of done”
- Smoke test runs on 2 samples end-to-end < 2 minutes
- Deterministic output given fixed seed
- Clear logging: per-claim retrieved passages + verdict

## General info,generally make all changes in /pipeline or/tests 
