"""
Chunker module for KDSH Track A pipeline.
Handles chunking of book texts with metadata extraction.
"""

import re
import pandas as pd
import pathway as pw
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ChunkConfig:
    """Configuration for text chunking."""
    chunk_size: int = 400       # Number of tokens per chunk
    overlap_front: int = 100    # Overlap tokens at start
    overlap_back: int = 100     # Overlap tokens at end


DEFAULT_CONFIG = ChunkConfig()


# ============================================================================
# Chapter Detection
# ============================================================================

# Common chapter patterns
CHAPTER_PATTERNS = [
    r'^CHAPTER\s+([IVXLCDM]+|\d+)[.\s]*(.*)$',      # CHAPTER I, CHAPTER 1
    r'^Chapter\s+([IVXLCDM]+|\d+)[.\s]*(.*)$',      # Chapter I, Chapter 1
    r'^([IVXLCDM]+)\.\s*(.*)$',                      # I. Title or IV. Title
    r'^(\d+)\.\s+([A-Z].*)$',                        # 1. Title
    r'^PART\s+([IVXLCDM]+|\d+)[.\s]*(.*)$',         # PART I
    r'^Part\s+([IVXLCDM]+|\d+)[.\s]*(.*)$',         # Part I
    r'^BOOK\s+([IVXLCDM]+|\d+)[.\s]*(.*)$',         # BOOK I
]


def detect_chapters(text: str) -> List[Dict[str, Any]]:
    """
    Detect chapter boundaries in text.
    
    Args:
        text: Full book text
        
    Returns:
        List of chapter info: {'chapter': name, 'start': char_idx, 'end': char_idx}
    """
    chapters = []
    lines = text.split('\n')
    current_pos = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        for pattern in CHAPTER_PATTERNS:
            match = re.match(pattern, stripped, re.IGNORECASE)
            if match:
                chapter_num = match.group(1)
                chapter_title = match.group(2).strip() if len(match.groups()) > 1 else ""
                chapter_name = f"Chapter {chapter_num}"
                if chapter_title:
                    chapter_name += f": {chapter_title}"
                
                chapters.append({
                    'chapter': chapter_name,
                    'start': current_pos,
                    'line': i
                })
                break
        
        current_pos += len(line) + 1  # +1 for newline
    
    # Set end positions
    for i, ch in enumerate(chapters):
        if i + 1 < len(chapters):
            ch['end'] = chapters[i + 1]['start']
        else:
            ch['end'] = len(text)
    
    return chapters


def get_chapter_for_position(chapters: List[Dict], char_pos: int) -> Optional[str]:
    """
    Get chapter name for a character position.
    
    Args:
        chapters: List of chapter info dicts
        char_pos: Character position in text
        
    Returns:
        Chapter name or None
    """
    for ch in chapters:
        if ch['start'] <= char_pos < ch['end']:
            return ch['chapter']
    return None


# ============================================================================
# Tokenization
# ============================================================================

def tokenize_text(text: str) -> List[str]:
    """
    Simple tokenizer that splits on whitespace and punctuation.
    
    Args:
        text: Text to tokenize
    
    Returns:
        List of tokens
    """
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
    return tokens


def get_token_char_positions(text: str) -> List[tuple]:
    """
    Get character start/end positions for each token.
    
    Args:
        text: Text to tokenize
        
    Returns:
        List of (token, start_char, end_char) tuples
    """
    positions = []
    for match in re.finditer(r'\b\w+\b|[^\w\s]', text):
        positions.append((match.group(), match.start(), match.end()))
    return positions


# ============================================================================
# Core Chunking
# ============================================================================

def chunk_text(
    text: str,
    config: ChunkConfig = DEFAULT_CONFIG,
    story: Optional[str] = None,
    extract_chapters: bool = True
) -> List[Dict[str, Any]]:
    """
    Chunk text into overlapping chunks with token-based boundaries.
    
    Args:
        text: Text to chunk
        config: Chunking configuration
        story: Story/book name metadata
        extract_chapters: Whether to detect and add chapter metadata
    
    Returns:
        List of chunks with metadata
    """
    token_positions = get_token_char_positions(text)
    if not token_positions:
        return []
    
    # Detect chapters if requested
    chapters = detect_chapters(text) if extract_chapters else []
    
    chunks = []
    pos = 0
    chunk_id = 0
    
    while pos < len(token_positions):
        # Calculate chunk boundaries in token space
        chunk_start = max(0, pos - config.overlap_front)
        chunk_end = min(len(token_positions), pos + config.chunk_size + config.overlap_back)
        
        # Extract chunk tokens
        chunk_token_info = token_positions[chunk_start:chunk_end]
        chunk_tokens = [t[0] for t in chunk_token_info]
        chunk_content = ' '.join(chunk_tokens)
        
        # Get character positions for metadata
        char_start = chunk_token_info[0][1] if chunk_token_info else 0
        char_end = chunk_token_info[-1][2] if chunk_token_info else 0
        
        # Get chapter info based on start position
        chapter = get_chapter_for_position(chapters, char_start) if chapters else None
        
        # Estimate page (assuming ~3000 chars per page)
        page = char_start // 3000 + 1
        
        # Create chunk record with metadata
        chunk_record = {
            'chunk_id': f"{story}_{chunk_id}" if story else str(chunk_id),
            'content': chunk_content,
            'start_idx': chunk_start,          # Token index
            'end_idx': chunk_end,              # Token index
            'char_start': char_start,          # Character index
            'char_end': char_end,              # Character index
            'chapter': chapter,
            'page': page,
            'story': story,
            'token_count': len(chunk_tokens)
        }
        chunks.append(chunk_record)
        
        # Move position forward
        pos += config.chunk_size
        chunk_id += 1
    
    return chunks


def chunk_books(
    books: Dict[str, str],
    config: ChunkConfig = DEFAULT_CONFIG,
    story_names: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Chunk all loaded books into overlapping chunks.
    
    Args:
        books: Dict mapping book_name -> book_text
        config: Chunking configuration
        story_names: Optional list of book names to chunk (if None, chunks all)
    
    Returns:
        List of all chunks with metadata from all books
    """
    all_chunks = []
    processed_stories = set()  # Avoid duplicates from case-insensitive dict
    
    for book_name, text in books.items():
        # Skip if we've already processed this story (case variations)
        normalized_name = book_name.lower()
        if normalized_name in processed_stories:
            continue
        
        if story_names is not None and book_name not in story_names:
            continue
        
        chunks = chunk_text(
            text,
            config=config,
            story=book_name,
            extract_chapters=True
        )
        all_chunks.extend(chunks)
        processed_stories.add(normalized_name)
    
    return all_chunks


def chunks_to_dataframe(chunks: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert chunks to pandas DataFrame.
    
    Args:
        chunks: List of chunk dicts
        
    Returns:
        DataFrame with chunk data
    """
    return pd.DataFrame(chunks)


def create_chunk_table(chunks: List[Dict[str, Any]]) -> Any:
    """
    Create Pathway table from chunks.
    
    Args:
        chunks: List of chunk dicts
        
    Returns:
        Pathway table with chunk data
    """
    df = chunks_to_dataframe(chunks)
    return pw.debug.table_from_pandas(df)


# ============================================================================
# Pipeline Integration
# ============================================================================

class BookChunker:
    """
    Main chunker class for pipeline integration.
    Handles book loading, chunking, and table creation.
    """
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        """
        Initialize chunker.
        
        Args:
            config: Optional chunking configuration
        """
        self.config = config or DEFAULT_CONFIG
        self.chunks: List[Dict[str, Any]] = []
        self._chunk_df: Optional[pd.DataFrame] = None
    
    def chunk_books(self, books: Dict[str, str]) -> 'BookChunker':
        """
        Chunk all books.
        
        Args:
            books: Dict mapping book_name -> book_text
            
        Returns:
            self for chaining
        """
        self.chunks = chunk_books(books, self.config)
        self._chunk_df = None  # Reset cached df
        return self
    
    @property
    def dataframe(self) -> pd.DataFrame:
        """Get chunks as DataFrame."""
        if self._chunk_df is None:
            self._chunk_df = chunks_to_dataframe(self.chunks)
        return self._chunk_df
    
    @property
    def num_chunks(self) -> int:
        """Get number of chunks."""
        return len(self.chunks)
    
    def get_chunks_for_story(self, story: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific story.
        
        Args:
            story: Story/book name
            
        Returns:
            List of chunks for that story
        """
        return [c for c in self.chunks if c['story'] == story]
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get chunk by its ID.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Chunk dict or None
        """
        for c in self.chunks:
            if c['chunk_id'] == chunk_id:
                return c
        return None
    
    def summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.
        
        Returns:
            Dict with summary stats
        """
        if not self.chunks:
            return {'total_chunks': 0}
        
        stories = set(c['story'] for c in self.chunks)
        chapters_per_story = {}
        for story in stories:
            story_chunks = self.get_chunks_for_story(story)
            chapters = set(c['chapter'] for c in story_chunks if c['chapter'])
            chapters_per_story[story] = len(chapters)
        
        return {
            'total_chunks': len(self.chunks),
            'stories': list(stories),
            'chunks_per_story': {s: len(self.get_chunks_for_story(s)) for s in stories},
            'chapters_per_story': chapters_per_story,
            'config': {
                'chunk_size': self.config.chunk_size,
                'overlap_front': self.config.overlap_front,
                'overlap_back': self.config.overlap_back
            }
        }


# ============================================================================
# Main / Test
# ============================================================================

if __name__ == "__main__":
    from pathlib import Path
    from loader import load_books
    
    dataset_dir = Path(__file__).parent.parent / "Dataset"
    
    print("=" * 60)
    print("KDSH Track A - Book Chunker Test")
    print("=" * 60)
    
    # Load books
    print("\n1. Loading books...")
    books = load_books(str(dataset_dir))
    print(f"   Loaded {len(books)} book entries")
    
    # Initialize chunker
    print("\n2. Initializing chunker...")
    config = ChunkConfig(chunk_size=400, overlap_front=100, overlap_back=100)
    chunker = BookChunker(config)
    
    # Chunk books
    print("\n3. Chunking books...")
    chunker.chunk_books(books)
    
    # Summary
    print("\n4. Summary:")
    summary = chunker.summary()
    print(f"   Total chunks: {summary['total_chunks']}")
    print(f"   Stories: {summary['stories']}")
    for story, count in summary['chunks_per_story'].items():
        chapters = summary['chapters_per_story'].get(story, 0)
        print(f"   - {story}: {count} chunks, {chapters} chapters detected")
    
    # Sample chunks
    print("\n5. Sample chunks:")
    for i, chunk in enumerate(chunker.chunks[:3]):
        print(f"\n   Chunk {i+1}:")
        print(f"   - ID: {chunk['chunk_id']}")
        print(f"   - Story: {chunk['story']}")
        print(f"   - Chapter: {chunk['chapter']}")
        print(f"   - Page: {chunk['page']}")
        print(f"   - Tokens: {chunk['start_idx']} - {chunk['end_idx']} ({chunk['token_count']} tokens)")
        print(f"   - Content: {chunk['content'][:80]}...")
    
    # DataFrame
    print("\n6. DataFrame shape:", chunker.dataframe.shape)
    print("   Columns:", list(chunker.dataframe.columns))
