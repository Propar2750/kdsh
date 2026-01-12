#!/usr/bin/env python
"""
KDSH Track A - FAST Verification Pipeline Runner

10x faster than original by minimizing LLM calls.
Includes embedding caching for faster subsequent runs.

Usage (inside Docker):
    python -m pipeline.run_eval_fast --max-samples 5
    python -m pipeline.run_eval_fast --max-samples 20 --verbose
"""

import argparse
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from typing import Any, Dict, List

from pipeline.loader import load_books
from pipeline.chunker import BookChunker, ChunkConfig
from pipeline.embedder import ChunkEmbedder
from pipeline.verifier_fast import FastVerificationPipeline, FastEvaluator, FastVerifierConfig

# Cache paths
CACHE_DIR = Path("/app/.cache") if Path("/app").exists() else Path(__file__).parent.parent / ".cache"
CHUNKS_CACHE = CACHE_DIR / "chunks.pkl"
EMBEDDINGS_CACHE = CACHE_DIR / "embeddings.npy"


def load_or_create_chunks(dataset_dir: Path, force_reload: bool = False):
    """Load chunks from cache or create new ones."""
    CACHE_DIR.mkdir(exist_ok=True)
    
    if not force_reload and CHUNKS_CACHE.exists():
        print("      Loading chunks from cache...")
        with open(CHUNKS_CACHE, 'rb') as f:
            chunks = pickle.load(f)
        print(f"      Loaded {len(chunks)} cached chunks")
        return chunks
    
    # Create new chunks
    books = load_books(str(dataset_dir))
    print(f"      Loaded {len(books)} books")
    
    chunk_config = ChunkConfig(chunk_size=400, overlap_front=100, overlap_back=100)
    chunker = BookChunker(chunk_config)
    chunker.chunk_books(books)
    
    # Cache chunks
    with open(CHUNKS_CACHE, 'wb') as f:
        pickle.dump(chunker.chunks, f)
    print(f"      Created and cached {len(chunker.chunks)} chunks")
    
    return chunker.chunks


def load_or_create_embeddings(chunks: list, force_reload: bool = False):
    """Load embeddings from cache or create new ones."""
    CACHE_DIR.mkdir(exist_ok=True)
    
    embedder = ChunkEmbedder()
    
    if not force_reload and EMBEDDINGS_CACHE.exists():
        print("      Loading embeddings from cache...")
        embeddings = np.load(EMBEDDINGS_CACHE)
        if len(embeddings) == len(chunks):
            embedder._chunks = chunks
            embedder._embeddings = embeddings
            print(f"      Loaded cached embeddings: {embeddings.shape}")
            return embedder
        else:
            print("      Cache size mismatch, regenerating...")
    
    # Create new embeddings
    embedder.embed_chunks(chunks)
    
    # Cache embeddings
    np.save(EMBEDDINGS_CACHE, embedder.embeddings)
    print(f"      Created and cached embeddings: {embedder.embeddings.shape}")
    
    return embedder


def main():
    parser = argparse.ArgumentParser(description="KDSH Track A - FAST Pipeline")
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples to evaluate (default: all)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--output', '-o', type=str, default='eval_results_fast.json',
                        help='Output JSON file (default: eval_results_fast.json)')
    parser.add_argument('--out', type=str, default='results.csv',
                        help='Output CSV file for submission (default: results.csv)')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Input directory containing train.csv/test.csv and Books/')
    parser.add_argument('--test', action='store_true',
                        help='Run on test.csv instead of train.csv (for submission)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Force regenerate chunks and embeddings')
    
    args = parser.parse_args()
    
    # Determine dataset directory
    if args.input_dir:
        dataset_dir = Path(args.input_dir)
    elif Path("/app/Dataset").exists():
        dataset_dir = Path("/app/Dataset")
    else:
        dataset_dir = Path(__file__).parent.parent / "Dataset"
    
    print("="*60)
    print("KDSH Track A - FAST Verification Pipeline")
    print("="*60)
    print(f"Dataset dir: {dataset_dir}")
    print(f"Mode: {'TEST (submission)' if args.test else 'TRAIN (evaluation)'}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples}")
    
    # Step 1-2: Load/create chunks (with caching)
    print("\n[1/3] Loading chunks...")
    chunks = load_or_create_chunks(dataset_dir, force_reload=args.no_cache)
    
    # Step 3: Load/create embeddings (with caching)
    print("\n[2/3] Loading embeddings...")
    embedder = load_or_create_embeddings(chunks, force_reload=args.no_cache)
    
    # Step 4: Create fast pipeline
    print("\n[3/3] Creating FAST pipeline...")
    config = FastVerifierConfig(max_claims=5, top_k_retrieval=10, top_k_final=5)
    pipeline = FastVerificationPipeline(chunks, embedder, config)
    
    # Load samples
    csv_file = "test.csv" if args.test else "train.csv"
    df = pd.read_csv(dataset_dir / csv_file)
    samples: List[Dict[str, Any]] = df.to_dict('records')  # type: ignore
    
    if args.test:
        print(f"\nTest samples: {len(df)}")
    else:
        print(f"\nTotal samples: {len(df)} ({len(df[df['label']=='consistent'])} consistent, {len(df[df['label']=='contradict'])} contradict)")
    
    # For test mode, run predictions without evaluation
    if args.test:
        results = []
        for i, sample in enumerate(samples):
            if args.max_samples and i >= args.max_samples:
                break
            print(f"\n[{i+1}/{len(samples) if not args.max_samples else min(args.max_samples, len(samples))}] Sample {sample['id']} - {sample['char']}")
            prediction, _ = pipeline.verify_backstory(
                backstory=sample['content'],
                character=sample['char'],
                book_name=sample['book_name'],
                sample_id=sample['id'],
                verbose=args.verbose,
                save_results=False
            )
            results.append({'id': sample['id'], 'prediction': prediction})
        
        # Save submission CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(args.out, index=False)
        print(f"\n{'='*60}")
        print(f"Submission saved to: {args.out}")
        print(f"{'='*60}")
        return 0
    
    # Evaluate on train
    evaluator = FastEvaluator(pipeline)
    summary = evaluator.evaluate(samples, max_samples=args.max_samples, verbose=args.verbose)
    
    # Save JSON results
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
    
    # Also save CSV for consistency
    results_df = pd.DataFrame([
        {'id': r['id'], 'prediction': 1 if r['prediction'] == 'consistent' else 0}
        for r in summary['results']
    ])
    results_df.to_csv(args.out, index=False)
    
    print(f"\nResults saved to: {args.output}")
    print(f"Submission CSV: {args.out}")
    return summary['accuracy']


if __name__ == "__main__":
    main()
