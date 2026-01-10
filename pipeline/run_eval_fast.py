#!/usr/bin/env python
"""
KDSH Track A - FAST Verification Pipeline Runner

10x faster than original by minimizing LLM calls.

Usage (inside Docker):
    python -m pipeline.run_eval_fast --max-samples 5
    python -m pipeline.run_eval_fast --max-samples 20 --verbose
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

from pipeline.loader import load_books
from pipeline.chunker import BookChunker, ChunkConfig
from pipeline.embedder import ChunkEmbedder
from pipeline.verifier_fast import FastVerificationPipeline, FastEvaluator, FastVerifierConfig


def main():
    parser = argparse.ArgumentParser(description="KDSH Track A - FAST Pipeline")
    parser.add_argument('--max-samples', type=int, default=5,
                        help='Max samples to evaluate (default: 5)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--output', '-o', type=str, default='eval_results_fast.json',
                        help='Output file')
    
    args = parser.parse_args()
    
    dataset_dir = Path("/app/Dataset") if Path("/app/Dataset").exists() else Path(__file__).parent.parent / "Dataset"
    
    print("="*60)
    print("KDSH Track A - FAST Verification Pipeline")
    print("="*60)
    print(f"Max samples: {args.max_samples}")
    
    # Step 1: Load books
    print("\n[1/4] Loading books...")
    books = load_books(str(dataset_dir))
    print(f"      Loaded {len(books)} books")
    
    # Step 2: Chunk books
    print("\n[2/4] Chunking books...")
    chunk_config = ChunkConfig(chunk_size=400, overlap_front=100, overlap_back=100)
    chunker = BookChunker(chunk_config)
    chunker.chunk_books(books)
    print(f"      Created {len(chunker.chunks)} chunks")
    
    # Step 3: Embed chunks
    print("\n[3/4] Embedding chunks...")
    embedder = ChunkEmbedder()
    embedder.embed_chunks(chunker.chunks)
    print(f"      Embeddings: {embedder.embeddings.shape}")
    
    # Step 4: Create fast pipeline
    print("\n[4/4] Creating FAST pipeline...")
    config = FastVerifierConfig(max_claims=5, top_k_retrieval=10, top_k_final=5)
    pipeline = FastVerificationPipeline(chunker.chunks, embedder, config)
    
    # Load samples
    df = pd.read_csv(dataset_dir / "train.csv")
    samples = df.to_dict('records')
    print(f"\nTotal samples: {len(df)} ({len(df[df['label']=='consistent'])} consistent, {len(df[df['label']=='contradict'])} contradict)")
    
    # Evaluate
    evaluator = FastEvaluator(pipeline)
    summary = evaluator.evaluate(samples, max_samples=args.max_samples, verbose=args.verbose)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'accuracy': summary['accuracy'],
            'correct': summary['correct'],
            'total': summary['total'],
            'llm_stats': summary['llm_stats'],
            'results': [
                {'id': r['id'], 'true': r['true_label'], 'pred': r['prediction'], 'correct': r['correct']}
                for r in summary['results']
            ]
        }, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")
    return summary['accuracy']


if __name__ == "__main__":
    main()
