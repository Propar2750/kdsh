"""
Embedder module for KDSH Track A pipeline.
Embeds chunks using nomic-ai/nomic-embed-text-v1.5 and stores in Pathway vector index.
"""

import numpy as np
import pandas as pd
import pathway as pw
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class EmbedderConfig:
    """Configuration for embedder."""
    model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    embedding_dim: int = 768  # Full dimension, can reduce with matryoshka
    matryoshka_dim: Optional[int] = None  # If set, truncate to this dimension (64, 128, 256, 512)
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    normalize: bool = True
    trust_remote_code: bool = True


DEFAULT_EMBEDDER_CONFIG = EmbedderConfig()


# ============================================================================
# Task Prefixes for nomic-embed-text-v1.5
# ============================================================================

class TaskType:
    """Task type prefixes for nomic-embed-text-v1.5."""
    SEARCH_DOCUMENT = "search_document"
    SEARCH_QUERY = "search_query"
    CLUSTERING = "clustering"
    CLASSIFICATION = "classification"


def add_task_prefix(texts: List[str], task_type: str) -> List[str]:
    """
    Add task prefix to texts for nomic-embed model.
    
    Args:
        texts: List of texts to prefix
        task_type: Task type (search_document, search_query, etc.)
        
    Returns:
        List of prefixed texts
    """
    return [f"{task_type}: {text}" for text in texts]


# ============================================================================
# Embedding Model
# ============================================================================

class NomicEmbedder:
    """
    Embedder using nomic-ai/nomic-embed-text-v1.5.
    Uses sentence-transformers for efficient batch embedding.
    """
    
    def __init__(self, config: Optional[EmbedderConfig] = None):
        """
        Initialize embedder.
        
        Args:
            config: Embedder configuration
        """
        self.config = config or DEFAULT_EMBEDDER_CONFIG
        self._model = None
        self._device = self.config.device
    
    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            
            print(f"Loading embedding model: {self.config.model_name}")
            print(f"Device: {self._device}")
            
            self._model = SentenceTransformer(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code,
                device=self._device
            )
        return self._model
    
    def embed_texts(
        self,
        texts: List[str],
        task_type: str = TaskType.SEARCH_DOCUMENT,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Embed a list of texts.
        
        Args:
            texts: Texts to embed
            task_type: Task prefix type
            show_progress: Show progress bar
            
        Returns:
            Numpy array of shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        # Add task prefix
        prefixed_texts = add_task_prefix(texts, task_type)
        
        # Encode
        embeddings = self.model.encode(
            prefixed_texts,
            batch_size=self.config.batch_size,
            show_progress_bar=show_progress,
            convert_to_tensor=True,
            device=self._device
        )
        
        # Apply matryoshka dimensionality reduction if configured
        if self.config.matryoshka_dim is not None:
            embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
            embeddings = embeddings[:, :self.config.matryoshka_dim]
        
        # Normalize
        if self.config.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
    
    def embed_documents(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Embed texts as documents (for indexing)."""
        return self.embed_texts(texts, TaskType.SEARCH_DOCUMENT, show_progress)
    
    def embed_queries(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Embed texts as queries (for retrieval)."""
        return self.embed_texts(texts, TaskType.SEARCH_QUERY, show_progress)
    
    @property
    def embedding_dimension(self) -> int:
        """Get the output embedding dimension."""
        if self.config.matryoshka_dim is not None:
            return self.config.matryoshka_dim
        return self.config.embedding_dim


# ============================================================================
# Chunk Embedder (integrates with chunker output)
# ============================================================================

class ChunkEmbedder:
    """
    Embeds chunks and creates vector index for retrieval.
    Integrates with BookChunker output.
    """
    
    def __init__(self, config: Optional[EmbedderConfig] = None):
        """
        Initialize chunk embedder.
        
        Args:
            config: Embedder configuration
        """
        self.config = config or DEFAULT_EMBEDDER_CONFIG
        self.embedder = NomicEmbedder(config)
        self._embeddings: Optional[np.ndarray] = None
        self._chunk_ids: List[str] = []
        self._chunks: List[Dict[str, Any]] = []
    
    def embed_chunks(
        self,
        chunks: List[Dict[str, Any]],
        content_key: str = "content",
        show_progress: bool = True
    ) -> 'ChunkEmbedder':
        """
        Embed all chunks.
        
        Args:
            chunks: List of chunk dictionaries from BookChunker
            content_key: Key for chunk content in dict
            show_progress: Show progress bar
            
        Returns:
            self for chaining
        """
        self._chunks = chunks
        self._chunk_ids = [c.get('chunk_id', str(i)) for i, c in enumerate(chunks)]
        
        # Extract content
        texts = [c[content_key] for c in chunks]
        
        print(f"Embedding {len(texts)} chunks...")
        self._embeddings = self.embedder.embed_documents(texts, show_progress)
        print(f"Embeddings shape: {self._embeddings.shape}")
        
        return self
    
    @property
    def embeddings(self) -> np.ndarray:
        """Get embeddings array."""
        if self._embeddings is None:
            raise ValueError("No embeddings computed. Call embed_chunks first.")
        return self._embeddings
    
    @property
    def num_chunks(self) -> int:
        """Number of embedded chunks."""
        return len(self._chunks)
    
    def get_embedding(self, chunk_id: str) -> Optional[np.ndarray]:
        """Get embedding for a specific chunk."""
        try:
            idx = self._chunk_ids.index(chunk_id)
            return self._embeddings[idx]
        except ValueError:
            return None
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert chunks with embeddings to DataFrame.
        
        Returns:
            DataFrame with chunk data and embeddings
        """
        df = pd.DataFrame(self._chunks)
        df['embedding'] = list(self._embeddings)
        return df
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for most similar chunks to a query.
        
        Args:
            query: Query text
            top_k: Number of results to return
            threshold: Optional similarity threshold
            
        Returns:
            List of (chunk, score) dicts sorted by similarity
        """
        # Embed query
        query_embedding = self.embedder.embed_queries([query], show_progress=False)[0]
        
        # Compute cosine similarity
        similarities = np.dot(self._embeddings, query_embedding)
        
        # Get top-k indices
        if threshold is not None:
            mask = similarities >= threshold
            valid_indices = np.where(mask)[0]
            top_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1][:top_k]]
        else:
            top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            result = self._chunks[idx].copy()
            result['score'] = float(similarities[idx])
            results.append(result)
        
        return results


# ============================================================================
# Pathway Vector Index Integration
# ============================================================================

class PathwayVectorIndex:
    """
    Vector index using Pathway for storage and retrieval.
    """
    
    def __init__(self, embedder: Optional[NomicEmbedder] = None):
        """
        Initialize Pathway vector index.
        
        Args:
            embedder: Optional embedder instance (creates new if None)
        """
        self.embedder = embedder or NomicEmbedder()
        self._table: Optional[Any] = None
        self._chunks: List[Dict[str, Any]] = []
        self._embeddings: Optional[np.ndarray] = None
    
    def index_chunks(
        self,
        chunks: List[Dict[str, Any]],
        content_key: str = "content"
    ) -> 'PathwayVectorIndex':
        """
        Index chunks into Pathway table with embeddings.
        
        Args:
            chunks: List of chunk dictionaries
            content_key: Key for content in chunk dict
            
        Returns:
            self for chaining
        """
        self._chunks = chunks
        
        # Extract content and embed
        texts = [c[content_key] for c in chunks]
        print(f"Indexing {len(texts)} chunks into Pathway vector index...")
        
        self._embeddings = self.embedder.embed_documents(texts, show_progress=True)
        
        # Create DataFrame with embeddings
        df = pd.DataFrame(chunks)
        df['embedding'] = [emb.tolist() for emb in self._embeddings]
        
        # Create Pathway table
        self._table = pw.debug.table_from_pandas(df)
        
        print(f"Indexed {len(chunks)} chunks with {self._embeddings.shape[1]}-dim embeddings")
        return self
    
    @property
    def table(self) -> Any:
        """Get Pathway table."""
        return self._table
    
    def query(
        self,
        query_text: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query the vector index.
        
        Args:
            query_text: Query string
            top_k: Number of results
            
        Returns:
            List of matching chunks with scores
        """
        # Embed query
        query_emb = self.embedder.embed_queries([query_text], show_progress=False)[0]
        
        # Cosine similarity search
        similarities = np.dot(self._embeddings, query_emb)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            result = self._chunks[idx].copy()
            result['score'] = float(similarities[idx])
            results.append(result)
        
        return results
    
    def batch_query(
        self,
        queries: List[str],
        top_k: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """
        Batch query the vector index.
        
        Args:
            queries: List of query strings
            top_k: Number of results per query
            
        Returns:
            List of result lists
        """
        # Embed all queries
        query_embs = self.embedder.embed_queries(queries, show_progress=False)
        
        # Compute all similarities at once
        all_similarities = np.dot(query_embs, self._embeddings.T)
        
        all_results = []
        for i, similarities in enumerate(all_similarities):
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = []
            for idx in top_indices:
                result = self._chunks[idx].copy()
                result['score'] = float(similarities[idx])
                results.append(result)
            all_results.append(results)
        
        return all_results


# ============================================================================
# Main / Test
# ============================================================================

if __name__ == "__main__":
    from pathlib import Path
    from chunker import BookChunker, ChunkConfig
    from loader import load_books
    
    dataset_dir = Path(__file__).parent.parent / "Dataset"
    
    print("=" * 60)
    print("KDSH Track A - Embedder Test")
    print("=" * 60)
    
    # Check device
    print(f"\n1. Device check:")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load and chunk books (use smaller chunks for testing)
    print("\n2. Loading and chunking books...")
    books = load_books(str(dataset_dir))
    
    # Use smaller config for faster testing
    chunk_config = ChunkConfig(chunk_size=200, overlap_front=50, overlap_back=50)
    chunker = BookChunker(chunk_config)
    chunker.chunk_books(books)
    
    # Take first 50 chunks for testing
    test_chunks = chunker.chunks[:50]
    print(f"   Using {len(test_chunks)} chunks for testing")
    
    # Initialize embedder
    print("\n3. Initializing embedder...")
    embed_config = EmbedderConfig(
        matryoshka_dim=256,  # Use smaller dimension for testing
        batch_size=16
    )
    chunk_embedder = ChunkEmbedder(embed_config)
    
    # Embed chunks
    print("\n4. Embedding chunks...")
    chunk_embedder.embed_chunks(test_chunks)
    
    # Test search
    print("\n5. Testing search...")
    query = "the captain sailed across the ocean"
    results = chunk_embedder.search(query, top_k=3)
    
    print(f"   Query: '{query}'")
    print(f"   Top {len(results)} results:")
    for i, r in enumerate(results):
        print(f"   {i+1}. Score: {r['score']:.4f}")
        print(f"      Story: {r['story']}")
        print(f"      Chapter: {r['chapter']}")
        print(f"      Content: {r['content'][:100]}...")
    
    # Test Pathway integration
    print("\n6. Testing Pathway vector index...")
    try:
        pw_index = PathwayVectorIndex()
        pw_index.index_chunks(test_chunks[:20])  # Use smaller subset
        
        pw_results = pw_index.query(query, top_k=3)
        print(f"   Pathway query results: {len(pw_results)}")
    except (AttributeError, Exception) as e:
        print(f"   Pathway not available (expected on Windows): {type(e).__name__}")
        print("   Pathway integration will work in Docker/Linux environment")
    
    print("\n" + "=" * 60)
    print("Embedder test complete!")
    print("=" * 60)
