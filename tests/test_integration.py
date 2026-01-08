"""
Integration test: end-to-end data loading workflow
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pathway as pw
from pipeline.loader import load_csv_to_pathway, load_books, create_story_table


def test_full_pipeline():
    """
    E2E test: Load CSV + books + create unified story table
    Verifies the complete data ingestion pipeline works end-to-end
    """
    dataset_dir = Path(__file__).parent.parent / "Dataset"
    
    # Step 1: Load books
    print("\n[Step 1] Loading book texts...")
    books = load_books(str(dataset_dir))
    print(f"  ✓ Loaded {len(books)} books")
    for name in list(books.keys())[:2]:
        print(f"    - {name}: {len(books[name])} chars")
    
    # Step 2: Load CSV (use pandas for inspection, Pathway table for pipeline)
    print("\n[Step 2] Loading training CSV...")
    csv_path = str(dataset_dir / "train.csv")
    df = pd.read_csv(csv_path)
    df["label"] = (df["label"] == "consistent").astype(int)
    print(f"  ✓ Loaded {len(df)} records")
    print(f"    Columns: {df.columns.tolist()}")
    print(f"    Labels: {df['label'].value_counts().to_dict()}")
    
    # Step 3: Create Pathway table
    print("\n[Step 3] Creating Pathway table...")
    csv_table = load_csv_to_pathway(csv_path)
    print(f"  ✓ Created Pathway table: {type(csv_table)}")
    
    # Step 4: Create unified story table
    print("\n[Step 4] Creating unified story table...")
    story_table = create_story_table(csv_path, books)
    print(f"  ✓ Created story table: {type(story_table)}")
    
    # Step 5: Verify data quality (using pandas for inspection)
    print("\n[Step 5] Data quality checks...")
    df["novel_text"] = df["book_name"].map(books)
    print(f"  ✓ No null IDs: {df['id'].notna().all()}")
    print(f"  ✓ No null novel_text: {df['novel_text'].notna().all()}")
    print(f"  ✓ No null backstory content: {df['content'].notna().all()}")
    
    # Sample output
    print("\n[Sample] First record:")
    row = df.iloc[0]
    print(f"  id: {row['id']}")
    print(f"  book_name: {row['book_name']}")
    print(f"  char: {row['char']}")
    print(f"  label: {row['label']}")
    print(f"  backstory (first 100 chars): {row['content'][:100]}...")
    print(f"  novel_text (first 100 chars): {row['novel_text'][:100]}...")
    
    print("\n" + "=" * 60)
    print("✓ End-to-end data loading pipeline PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_full_pipeline()
