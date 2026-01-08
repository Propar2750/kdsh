"""
Test suite for chunker module.
"""

import pytest
from pathlib import Path
from pipeline.chunker import (
    BookChunker, 
    ChunkConfig, 
    chunk_text, 
    chunk_books,
    tokenize_text,
    detect_chapters
)
from pipeline.loader import load_books


@pytest.fixture
def dataset_dir():
    """Get Dataset directory path."""
    return Path(__file__).parent.parent / "Dataset"


@pytest.fixture
def books(dataset_dir):
    """Load books fixture."""
    return load_books(str(dataset_dir))


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return "The quick brown fox jumps over the lazy dog. " * 100


# ============================================================================
# Tokenizer Tests
# ============================================================================

def test_tokenize_basic():
    """Test basic tokenization."""
    text = "Hello, world! This is a test."
    tokens = tokenize_text(text)
    
    assert len(tokens) > 0
    assert "Hello" in tokens
    assert "world" in tokens
    assert "," in tokens
    assert "!" in tokens


def test_tokenize_empty():
    """Test tokenization of empty string."""
    tokens = tokenize_text("")
    assert tokens == []


def test_tokenize_preserves_words():
    """Test that tokenizer preserves all words."""
    text = "one two three four five"
    tokens = tokenize_text(text)
    
    assert tokens == ["one", "two", "three", "four", "five"]


# ============================================================================
# Chapter Detection Tests
# ============================================================================

def test_detect_chapters_roman():
    """Test chapter detection with Roman numerals."""
    text = """Some intro text.

CHAPTER I

First chapter content.

CHAPTER II

Second chapter content.
"""
    chapters = detect_chapters(text)
    
    assert len(chapters) >= 2
    assert any("Chapter I" in ch['chapter'] for ch in chapters)


def test_detect_chapters_arabic():
    """Test chapter detection with Arabic numbers."""
    text = """Intro.

Chapter 1. The Beginning

Content here.

Chapter 2. The Middle

More content.
"""
    chapters = detect_chapters(text)
    
    assert len(chapters) >= 2


def test_detect_chapters_empty():
    """Test chapter detection with no chapters."""
    text = "Just some text without any chapter markers."
    chapters = detect_chapters(text)
    
    assert chapters == []


# ============================================================================
# Chunking Tests
# ============================================================================

def test_chunk_text_basic(sample_text):
    """Test basic text chunking."""
    config = ChunkConfig(chunk_size=50, overlap_front=10, overlap_back=10)
    chunks = chunk_text(sample_text, config=config, story="test_story")
    
    assert len(chunks) > 0
    assert all('content' in c for c in chunks)
    assert all('story' in c for c in chunks)
    assert all(c['story'] == 'test_story' for c in chunks)


def test_chunk_text_metadata():
    """Test chunk metadata is present."""
    text = "word " * 500  # Simple repeated text
    config = ChunkConfig(chunk_size=100, overlap_front=20, overlap_back=20)
    chunks = chunk_text(text, config=config, story="test")
    
    required_keys = ['chunk_id', 'content', 'start_idx', 'end_idx', 
                     'char_start', 'char_end', 'chapter', 'page', 
                     'story', 'token_count']
    
    for chunk in chunks:
        for key in required_keys:
            assert key in chunk, f"Missing key: {key}"


def test_chunk_text_overlap():
    """Test that chunks properly overlap."""
    text = " ".join([f"word{i}" for i in range(1000)])
    config = ChunkConfig(chunk_size=100, overlap_front=20, overlap_back=20)
    chunks = chunk_text(text, config=config)
    
    if len(chunks) >= 2:
        # First chunk should end at ~120 (100 + 20 back)
        # Second chunk should start at ~80 (100 - 20 front)
        chunk1 = chunks[0]
        chunk2 = chunks[1]
        
        # Check overlap exists
        assert chunk2['start_idx'] < chunk1['end_idx'], "Chunks should overlap"


def test_chunk_text_empty():
    """Test chunking empty text."""
    chunks = chunk_text("")
    assert chunks == []


def test_chunk_config_defaults():
    """Test default chunk configuration."""
    config = ChunkConfig()
    
    assert config.chunk_size == 400
    assert config.overlap_front == 100
    assert config.overlap_back == 100


# ============================================================================
# Book Chunking Tests
# ============================================================================

def test_chunk_books_basic(books):
    """Test chunking loaded books."""
    config = ChunkConfig(chunk_size=400, overlap_front=100, overlap_back=100)
    chunks = chunk_books(books, config=config)
    
    assert len(chunks) > 0
    # Should have processed 2 unique books (not the lowercase duplicates)
    stories = set(c['story'] for c in chunks)
    assert len(stories) == 2


def test_chunk_books_filter(books):
    """Test chunking with story filter."""
    config = ChunkConfig(chunk_size=400, overlap_front=100, overlap_back=100)
    chunks = chunk_books(books, config=config, story_names=["In search of the castaways"])
    
    stories = set(c['story'] for c in chunks)
    assert len(stories) == 1
    assert "In search of the castaways" in stories


# ============================================================================
# BookChunker Class Tests
# ============================================================================

def test_book_chunker_init():
    """Test BookChunker initialization."""
    chunker = BookChunker()
    
    assert chunker.config.chunk_size == 400
    assert chunker.config.overlap_front == 100
    assert chunker.config.overlap_back == 100
    assert chunker.chunks == []


def test_book_chunker_custom_config():
    """Test BookChunker with custom config."""
    config = ChunkConfig(chunk_size=200, overlap_front=50, overlap_back=50)
    chunker = BookChunker(config)
    
    assert chunker.config.chunk_size == 200


def test_book_chunker_workflow(books):
    """Test full BookChunker workflow."""
    chunker = BookChunker()
    chunker.chunk_books(books)
    
    # Check chunks created
    assert chunker.num_chunks > 0
    
    # Check dataframe
    df = chunker.dataframe
    assert len(df) == chunker.num_chunks
    assert 'content' in df.columns
    assert 'story' in df.columns
    
    # Check summary
    summary = chunker.summary()
    assert 'total_chunks' in summary
    assert summary['total_chunks'] == chunker.num_chunks


def test_book_chunker_get_chunks_for_story(books):
    """Test getting chunks for specific story."""
    chunker = BookChunker()
    chunker.chunk_books(books)
    
    story_chunks = chunker.get_chunks_for_story("In search of the castaways")
    
    assert len(story_chunks) > 0
    assert all(c['story'] == "In search of the castaways" for c in story_chunks)


def test_book_chunker_chaining(books):
    """Test method chaining."""
    chunker = BookChunker().chunk_books(books)
    
    assert chunker.num_chunks > 0


# ============================================================================
# Integration Tests
# ============================================================================

def test_chunking_produces_valid_output(books):
    """Integration test: verify chunking output is valid for next pipeline stage."""
    chunker = BookChunker()
    chunker.chunk_books(books)
    
    df = chunker.dataframe
    
    # Check no empty content
    assert not df['content'].isna().any(), "No chunks should have empty content"
    assert (df['content'].str.len() > 0).all(), "All chunks should have content"
    
    # Check token counts are reasonable
    assert (df['token_count'] > 0).all(), "All chunks should have tokens"
    
    # Check indices are valid
    assert (df['start_idx'] >= 0).all(), "Start indices should be non-negative"
    assert (df['end_idx'] > df['start_idx']).all(), "End should be > start"


def test_chunk_content_not_truncated(books):
    """Test that chunk content matches expected token count."""
    config = ChunkConfig(chunk_size=100, overlap_front=20, overlap_back=20)
    chunker = BookChunker(config)
    chunker.chunk_books(books)
    
    # Check first few chunks
    for chunk in chunker.chunks[:5]:
        tokens = tokenize_text(chunk['content'])
        # Allow some variance due to punctuation
        assert len(tokens) <= chunk['token_count'] + 10, \
            f"Token count mismatch: {len(tokens)} vs reported {chunk['token_count']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
