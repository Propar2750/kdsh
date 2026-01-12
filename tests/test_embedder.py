"""
Test suite for embedder module.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


# ============================================================================
# Test fixtures
# ============================================================================

@pytest.fixture
def dataset_dir():
    """Get Dataset directory path."""
    return Path(__file__).parent.parent / "Dataset"


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    return [
        {
            'chunk_id': 'test_0',
            'content': 'The quick brown fox jumps over the lazy dog.',
            'story': 'Test Story',
            'chapter': 'Chapter 1',
            'page': 1,
            'start_idx': 0,
            'end_idx': 10
        },
        {
            'chunk_id': 'test_1',
            'content': 'A sailor went to sea to see what he could see.',
            'story': 'Test Story',
            'chapter': 'Chapter 2',
            'page': 2,
            'start_idx': 10,
            'end_idx': 20
        },
        {
            'chunk_id': 'test_2',
            'content': 'The captain navigated through the stormy waters.',
            'story': 'Test Story',
            'chapter': 'Chapter 3',
            'page': 3,
            'start_idx': 20,
            'end_idx': 30
        },
    ]


# ============================================================================
# Configuration Tests
# ============================================================================

def test_embedder_config_defaults():
    """Test default embedder configuration."""
    from pipeline.embedder import EmbedderConfig
    
    config = EmbedderConfig()
    
    assert config.model_name == "nomic-ai/nomic-embed-text-v1.5"
    assert config.embedding_dim == 768
    assert config.batch_size == 32
    assert config.normalize is True


def test_embedder_config_custom():
    """Test custom embedder configuration."""
    from pipeline.embedder import EmbedderConfig
    
    config = EmbedderConfig(
        matryoshka_dim=256,
        batch_size=16,
        device='cpu'
    )
    
    assert config.matryoshka_dim == 256
    assert config.batch_size == 16
    assert config.device == 'cpu'


# ============================================================================
# Task Type Tests
# ============================================================================

def test_task_types():
    """Test task type constants."""
    from pipeline.embedder import TaskType
    
    assert TaskType.SEARCH_DOCUMENT == "search_document"
    assert TaskType.SEARCH_QUERY == "search_query"


# ============================================================================
# Embedder Tests (with mocking for fast tests)
# ============================================================================

def test_nomic_embedder_init():
    """Test NomicEmbedder initialization."""
    from pipeline.embedder import NomicEmbedder, EmbedderConfig
    
    config = EmbedderConfig(device='cpu')
    embedder = NomicEmbedder(config)
    
    assert embedder.config.device == 'cpu'
    assert embedder._model is None  # Model not loaded yet


def test_nomic_embedder_embedding_dimension():
    """Test embedding dimension calculation."""
    from pipeline.embedder import NomicEmbedder, EmbedderConfig
    
    # Default dimension
    embedder = NomicEmbedder(EmbedderConfig())
    assert embedder.embedding_dimension == 768
    
    # Matryoshka dimension
    embedder = NomicEmbedder(EmbedderConfig(matryoshka_dim=256))
    assert embedder.embedding_dimension == 256


@pytest.fixture
def mock_embedder():
    """Create a mock embedder for testing without loading real model."""
    from pipeline.embedder import NomicEmbedder, EmbedderConfig
    
    embedder = NomicEmbedder(EmbedderConfig(device='cpu', matryoshka_dim=64))
    
    # Mock the model
    mock_model = MagicMock()
    
    def mock_encode(texts, **kwargs):
        import torch
        n = len(texts)
        # Return random embeddings
        return torch.randn(n, 768)
    
    mock_model.encode = mock_encode
    embedder._model = mock_model
    
    return embedder


def test_embed_texts_mock(mock_embedder):
    """Test text embedding with mock model."""
    texts = ['Hello world', 'Test text']
    
    embeddings = mock_embedder.embed_texts(texts, show_progress=False)
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 2
    # Should be matryoshka_dim (64) since we set it in fixture
    assert embeddings.shape[1] == 64


def test_embed_documents_mock(mock_embedder):
    """Test document embedding with mock model."""
    texts = ['Document 1', 'Document 2']
    
    embeddings = mock_embedder.embed_documents(texts, show_progress=False)
    
    assert embeddings.shape[0] == 2


def test_embed_queries_mock(mock_embedder):
    """Test query embedding with mock model."""
    texts = ['Query 1', 'Query 2']
    
    embeddings = mock_embedder.embed_queries(texts, show_progress=False)
    
    assert embeddings.shape[0] == 2


# ============================================================================
# ChunkEmbedder Tests
# ============================================================================

def test_chunk_embedder_init():
    """Test ChunkEmbedder initialization."""
    from pipeline.embedder import ChunkEmbedder, EmbedderConfig
    
    config = EmbedderConfig(device='cpu')
    embedder = ChunkEmbedder(config)
    
    assert embedder.config.device == 'cpu'
    assert len(embedder._embeddings) == 0
    assert embedder._chunks == []


@pytest.fixture
def mock_chunk_embedder(sample_chunks):
    """Create a ChunkEmbedder with mock embeddings."""
    from pipeline.embedder import ChunkEmbedder, EmbedderConfig
    
    embedder = ChunkEmbedder(EmbedderConfig(device='cpu'))
    embedder._chunks = sample_chunks
    # Create fake normalized embeddings
    embeddings = np.random.randn(len(sample_chunks), 768)
    embedder._embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    return embedder


def test_chunk_embedder_num_chunks(mock_chunk_embedder, sample_chunks):
    """Test chunk count."""
    assert mock_chunk_embedder.num_chunks == len(sample_chunks)


def test_chunk_embedder_embeddings_property(mock_chunk_embedder):
    """Test embeddings property."""
    embeddings = mock_chunk_embedder.embeddings
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 768)


def test_chunk_embedder_search(mock_chunk_embedder):
    """Test search functionality."""
    # Mock the embedder's embed_queries method
    def mock_embed_queries(texts, show_progress=False):
        return np.array([[0.5, 0.5] + [0.0] * 766])  # Simple query vector
    
    mock_chunk_embedder.embedder.embed_queries = mock_embed_queries
    
    results = mock_chunk_embedder.search("test query", top_k=2)
    
    assert len(results) == 2
    assert all('score' in r for r in results)
    assert all('content' in r for r in results)


def test_chunk_embedder_search_with_threshold(mock_chunk_embedder):
    """Test search with similarity threshold."""
    def mock_embed_queries(texts, show_progress=False):
        return np.array([[0.5, 0.5] + [0.0] * 766])
    
    mock_chunk_embedder.embedder.embed_queries = mock_embed_queries
    
    # Very high threshold should filter most results
    results = mock_chunk_embedder.search("test query", top_k=10, threshold=0.99)
    
    # Results should be filtered by threshold
    assert all(r['score'] >= 0.99 for r in results)


def test_chunk_embedder_batch_search(mock_chunk_embedder):
    """Test batch search functionality."""
    def mock_embed_queries(texts, show_progress=False):
        return np.random.randn(len(texts), 768)
    
    mock_chunk_embedder.embedder.embed_queries = mock_embed_queries
    
    results = mock_chunk_embedder.batch_search(["query 1", "query 2"], top_k=2)
    
    assert len(results) == 2  # Two queries
    assert all(len(r) == 2 for r in results)  # Top 2 results each


# ============================================================================
# Integration Tests (uses real model - mark as slow)
# ============================================================================

@pytest.mark.slow
def test_real_embedding_integration():
    """Integration test with real embedding model."""
    from pipeline.embedder import NomicEmbedder, EmbedderConfig
    
    config = EmbedderConfig(
        device='cpu',
        matryoshka_dim=64,  # Small for speed
        batch_size=2
    )
    embedder = NomicEmbedder(config)
    
    texts = ['Hello world', 'Test text']
    embeddings = embedder.embed_texts(texts, show_progress=False)
    
    assert embeddings.shape == (2, 64)
    # Check normalized
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_array_almost_equal(norms, [1.0, 1.0], decimal=5)


@pytest.mark.slow
def test_real_search_integration(sample_chunks):
    """Integration test with real search."""
    from pipeline.embedder import ChunkEmbedder, EmbedderConfig
    
    config = EmbedderConfig(
        device='cpu',
        matryoshka_dim=64,
        batch_size=2
    )
    embedder = ChunkEmbedder(config)
    embedder.embed_chunks(sample_chunks, show_progress=False)
    
    # Search for sailor-related content
    results = embedder.search("sailor at sea", top_k=1)
    
    assert len(results) == 1
    # The sailor chunk should be most relevant
    assert 'sailor' in results[0]['content'].lower()


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
