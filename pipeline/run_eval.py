#!/usr/bin/env python
"""
KDSH Track A - Verification Pipeline Runner

This script runs the full verification pipeline on train.csv
and evaluates accuracy.

Usage (inside Docker):
    python -m pipeline.run_eval --max-samples 10 --verbose
    python -m pipeline.run_eval --all
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

from pipeline.loader import load_books
from pipeline.chunker import BookChunker, ChunkConfig
from pipeline.embedder import ChunkEmbedder
from pipeline.verifier import VerificationPipeline, Evaluator, VerifierConfig


def main():
    parser = argparse.ArgumentParser(description="KDSH Track A - Verification Pipeline")
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to evaluate (default: all)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed progress')
    parser.add_argument('--output', '-o', type=str, default='eval_results.json',
                        help='Output file for results')
    parser.add_argument('--chunk-size', type=int, default=400,
                        help='Chunk size in tokens (default: 400)')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Top-k chunks per query (default: 5)')
    parser.add_argument('--top-k-reranked', type=int, default=3,
                        help='Top-k after reranking (default: 3)')
    
    args = parser.parse_args()
    
    dataset_dir = Path("/app/Dataset") if Path("/app/Dataset").exists() else Path(__file__).parent.parent / "Dataset"
    
    print("="*60)
    print("KDSH Track A - Verification Pipeline")
    print("="*60)
    print(f"Dataset directory: {dataset_dir}")
    print(f"Max samples: {args.max_samples or 'all'}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Top-k per query: {args.top_k}")
    print(f"Top-k reranked: {args.top_k_reranked}")
    print("="*60)
    
    # Step 1: Load books
    print("\n[1/5] Loading books...")
    books = load_books(str(dataset_dir))
    print(f"      Loaded {len(books)} books:")
    for name in books:
        print(f"        - {name}: {len(books[name]):,} characters")
    
    # Step 2: Chunk books
    print("\n[2/5] Chunking books...")
    chunk_config = ChunkConfig(
        chunk_size=args.chunk_size,
        overlap_front=100,
        overlap_back=100
    )
    chunker = BookChunker(chunk_config)
    chunker.chunk_books(books)
    print(f"      Created {len(chunker.chunks):,} chunks")
    
    # Step 3: Embed chunks
    print("\n[3/5] Embedding chunks...")
    embedder = ChunkEmbedder()
    embedder.embed_chunks(chunker.chunks)
    print(f"      Embeddings shape: {embedder.embeddings.shape}")
    
    # Step 4: Create pipeline
    print("\n[4/5] Creating verification pipeline...")
    verifier_config = VerifierConfig(
        top_k_per_query=args.top_k,
        top_k_reranked=args.top_k_reranked
    )
    pipeline = VerificationPipeline(chunker.chunks, embedder, verifier_config)
    
    # Step 5: Load and evaluate samples
    print("\n[5/5] Loading and evaluating samples...")
    df = pd.read_csv(dataset_dir / "train.csv")
    print(f"      Total samples in train.csv: {len(df)}")
    print(f"      Consistent: {len(df[df['label'] == 'consistent'])}")
    print(f"      Contradict: {len(df[df['label'] == 'contradict'])}")
    
    samples = df.to_dict('records')
    
    # Evaluate
    evaluator = Evaluator(pipeline)
    summary = evaluator.evaluate_dataset(
        samples,
        max_samples=args.max_samples,
        verbose=args.verbose
    )
    
    # Save results
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'chunk_size': args.chunk_size,
            'top_k_per_query': args.top_k,
            'top_k_reranked': args.top_k_reranked,
            'max_samples': args.max_samples
        },
        'summary': {
            'total_samples': summary['total_samples'],
            'correct': summary['correct'],
            'accuracy': summary['accuracy'],
            'consistent_accuracy': summary['consistent_accuracy'],
            'contradict_accuracy': summary['contradict_accuracy']
        },
        'results': [
            {
                'id': r['id'],
                'character': r['character'],
                'book_name': r['book_name'],
                'true_label': r['true_label_str'],
                'prediction': r['prediction_str'],
                'correct': r['correct'],
                'reason': r['details']['reason']
            }
            for r in summary['results']
        ]
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n      Results saved to: {args.output}")
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Accuracy:            {summary['accuracy']:.2%}")
    print(f"Consistent accuracy: {summary['consistent_accuracy']:.2%}")
    print(f"Contradict accuracy: {summary['contradict_accuracy']:.2%}")
    print("="*60)
    
    # Print misclassifications
    misclassified = [r for r in summary['results'] if not r['correct']]
    if misclassified:
        print(f"\nMisclassified samples ({len(misclassified)}):")
        for r in misclassified[:10]:  # Show first 10
            print(f"  ID {r['id']}: {r['character']} - True: {r['true_label_str']}, Pred: {r['prediction_str']}")
    
    return summary['accuracy']


if __name__ == "__main__":
    main()
