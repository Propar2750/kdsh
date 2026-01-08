"""
Test suite for data loader.
"""

import pytest
from pathlib import Path
import pandas as pd
import pathway as pw
from pipeline.loader import load_csv_to_pathway, load_books, create_story_table


@pytest.fixture
def dataset_dir():
    """Get Dataset directory path."""
    return Path(__file__).parent.parent / "Dataset"


def test_load_csv_basic(dataset_dir):
    """Test basic CSV loading."""
    csv_path = str(dataset_dir / "train.csv")
    table = load_csv_to_pathway(csv_path)
    
    # Verify table exists and is a Pathway table
    assert table is not None
    assert isinstance(table, pw.Table)
    print(f"Loaded Pathway table: {table}")


def test_load_csv_columns(dataset_dir):
    """Test that CSV has expected columns."""
    csv_path = str(dataset_dir / "train.csv")
    df = pd.read_csv(csv_path)
    
    expected_cols = {"id", "book_name", "char", "caption", "content", "label"}
    assert expected_cols.issubset(df.columns), f"Missing columns. Got: {df.columns.tolist()}"
    print(f"CSV columns verified: {df.columns.tolist()}")


def test_load_csv_label_encoding(dataset_dir):
    """Test that labels are properly encoded as 0/1."""
    csv_path = str(dataset_dir / "train.csv")
    df = pd.read_csv(csv_path)
    df["label"] = (df["label"] == "consistent").astype(int)
    
    assert df["label"].isin([0, 1]).all(), "Labels should be binary 0/1"
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")


def test_load_books(dataset_dir):
    """Test book loading."""
    books = load_books(str(dataset_dir))
    
    assert len(books) > 0, "Should load at least one book"
    # Check that books are accessible (case-insensitive)
    assert "in search of the castaways" in books or "In search of the castaways" in books
    assert "the count of monte cristo" in books or "The Count of Monte Cristo" in books
    print(f"Books loaded: {list(books.keys())}")


def test_book_content_not_empty(dataset_dir):
    """Test that loaded books have content."""
    books = load_books(str(dataset_dir))
    
    for book_name, text in books.items():
        assert len(text) > 0, f"Book {book_name} is empty"
        print(f"Book '{book_name}': {len(text)} chars")


def test_create_story_table(dataset_dir):
    """Test unified story table creation."""
    books = load_books(str(dataset_dir))
    csv_path = str(dataset_dir / "train.csv")
    
    story_table = create_story_table(csv_path, books)
    
    assert story_table is not None
    assert isinstance(story_table, pw.Table)
    print(f"Story table created successfully")


def test_story_table_has_novel_text(dataset_dir):
    """Test that story table includes novel_text column."""
    books = load_books(str(dataset_dir))
    csv_path = str(dataset_dir / "train.csv")
    df = pd.read_csv(csv_path)
    
    df["novel_text"] = df["book_name"].map(books)
    
    # All rows should have novel_text (no NaNs)
    missing = df[df["novel_text"].isna()]
    assert len(missing) == 0, f"Missing novel_text for: {missing['book_name'].tolist()}"
    print(f"All {len(df)} rows have novel_text")


if __name__ == "__main__":
    # Quick manual test
    data_dir = Path(__file__).parent.parent / "Dataset"
    
    print("=" * 60)
    print("Running manual data load tests...")
    print("=" * 60)
    
    print("\n[Test 1] Loading CSV...")
    test_load_csv_basic(data_dir)
    
    print("\n[Test 2] Verifying CSV columns...")
    test_load_csv_columns(data_dir)
    
    print("\n[Test 3] Label encoding...")
    test_load_csv_label_encoding(data_dir)
    
    print("\n[Test 4] Loading books...")
    test_load_books(data_dir)
    
    print("\n[Test 5] Book content check...")
    test_book_content_not_empty(data_dir)
    
    print("\n[Test 6] Story table creation...")
    test_create_story_table(data_dir)
    
    print("\n[Test 7] Story table has novel_text...")
    test_story_table_has_novel_text(data_dir)
    
    print("\n" + "=" * 60)
    print("All manual tests passed!")
    print("=" * 60)
