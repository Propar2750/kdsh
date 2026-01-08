"""
Data loader for KDSH Track A pipeline.
Loads train/test CSV and book texts into Pathway tables.
"""

import pandas as pd
import pathway as pw
from pathlib import Path
from typing import Optional, Union, Any


def load_csv_to_pathway(csv_path: str) -> Any:
    """
    Load CSV file into Pathway table.
    
    Args:
        csv_path: Path to CSV file (train.csv or test.csv)
    
    Returns:
        Pathway table with columns: id, book_name, char, caption, content, label (if present)
    """
    df = pd.read_csv(csv_path)
    
    # Convert label to 0/1 if present (consistent -> 1, contradict -> 0)
    if "label" in df.columns:
        # Only convert string labels, preserve existing numeric values
        mask = df["label"].astype(str).isin(["consistent", "contradict"])
        df.loc[mask & (df["label"] == "consistent"), "label"] = 1
        df.loc[mask & (df["label"] == "contradict"), "label"] = 0
    
    table = pw.debug.table_from_pandas(df)
    return table


def load_books(dataset_dir: str) -> dict:
    """
    Load book texts from Dataset/Books/ directory.
    
    Args:
        dataset_dir: Path to Dataset directory
    
    Returns:
        Dict mapping book_name -> book_text (case-insensitive lookup)
    """
    books_dir = Path(dataset_dir) / "Books"
    books = {}
    books_lower = {}  # For case-insensitive lookup
    
    if not books_dir.exists():
        print(f"Warning: Books directory not found at {books_dir}")
        return books
    
    for book_file in books_dir.glob("*.txt"):
        book_name = book_file.stem  # filename without .txt
        with open(book_file, "r", encoding="utf-8") as f:
            text = f.read()
            books[book_name] = text
            books_lower[book_name.lower()] = text  # Store lowercase key
    
    # Return a dict-like object that handles case-insensitive lookup
    class CaseInsensitiveDict(dict):
        def __missing__(self, key):
            # Try lowercase lookup
            if key.lower() in books_lower:
                return books_lower[key.lower()]
            return None
    
    result = CaseInsensitiveDict(books)
    result.update(books_lower)
    return result


def create_story_table(
    csv_path: str,
    books: dict,
    dataset_dir: Optional[str] = None
) -> Any:
    """
    Create unified story table with novel text joined from books.
    
    Args:
        csv_path: Path to train/test CSV
        books: Dict of book_name -> text
        dataset_dir: Optional path to Dataset dir (used to load books if None provided)
    
    Returns:
        Pathway table with columns: id, book_name, char, caption, content, label (if present), novel_text
    """
    df = pd.read_csv(csv_path)
    
    # Convert label to 0/1 if present
    if "label" in df.columns:
        # Only convert string labels, preserve existing numeric values
        mask = df["label"].astype(str).isin(["consistent", "contradict"])
        df.loc[mask & (df["label"] == "consistent"), "label"] = 1
        df.loc[mask & (df["label"] == "contradict"), "label"] = 0
    
    # Join with book texts
    df["novel_text"] = df["book_name"].map(books)
    
    # Check for missing books
    missing_books = df[df["novel_text"].isna()]["book_name"].unique()
    if len(missing_books) > 0:
        print(f"Warning: Missing books: {missing_books}")
    
    table = pw.debug.table_from_pandas(df)
    return table


if __name__ == "__main__":
    # Quick test
    dataset_dir = Path(__file__).parent.parent / "Dataset"
    
    print("Loading CSV...")
    csv_table = load_csv_to_pathway(str(dataset_dir / "train.csv"))
    print(f"CSV loaded: {csv_table}")
    
    print("\nLoading books...")
    books = load_books(str(dataset_dir))
    print(f"Books loaded: {list(books.keys())}")
    
    print("\nCreating story table...")
    story_table = create_story_table(str(dataset_dir / "train.csv"), books)
    print(f"Story table created: {story_table}")
