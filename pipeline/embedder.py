"""
Embedder module for KDSH Track A pipeline.
Embeds chunks using nomic-ai/nomic-embed-text-v1.5 for vector search.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class EmbedderConfig:
    """Configuration for embedder."""
    model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    embedding_dim: int = 768
    matryoshka_dim: Optional[int] = None  # Truncate to 64, 128, 256, or 512
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    normalize: bool = True
    trust_remote_code: bool = True


DEFAULT_CONFIG = EmbedderConfig()


class TaskType:
    """Task prefixes for nomic-embed-text-v1.5."""
    SEARCH_DOCUMENT = "search_document"
    SEARCH_QUERY = "search_query"


class NomicEmbedder:
    """Embedder using nomic-ai/nomic-embed-text-v1.5."""
    
    def __init__(self, config: Optional[EmbedderConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self._model = None
    
    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {self.config.model_name}")
            print(f"Device: {self.config.device}")
            self._model = SentenceTransformer(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code,
                device=self.config.device
            )
        return self._model
    
    def embed_texts(
        self,
        texts: List[str],
        task_type: str = TaskType.SEARCH_DOCUMENT,
        show_progress: bool = True
    ) -> np.ndarray:
        """Embed texts with task prefix."""
        if not texts:
            return np.array([])
        
        prefixed = [f"{task_type}: {t}" for t in texts]
        
        embeddings = self.model.encode(
            prefixed,
            batch_size=self.config.batch_size,
            show_progress_bar=show_progress,
            convert_to_tensor=True,
            device=self.config.device
        )
        
        # Matryoshka dimensionality reduction
        if self.config.matryoshka_dim:
            embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
            embeddings = embeddings[:, :self.config.matryoshka_dim]
        
        if self.config.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
    
    def embed_documents(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Embed texts for indexing."""
        return self.embed_texts(texts, TaskType.SEARCH_DOCUMENT, show_progress)
    
    def embed_queries(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Embed texts for search."""
        return self.embed_texts(texts, TaskType.SEARCH_QUERY, show_progress)
    
    @property
    def embedding_dimension(self) -> int:
        """Output embedding dimension."""
        return self.config.matryoshka_dim or self.config.embedding_dim


class ChunkEmbedder:
    """Embeds chunks and provides vector search."""
    
    def __init__(self, config: Optional[EmbedderConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.embedder = NomicEmbedder(config)
        self._embeddings: np.ndarray = np.array([])
        self._chunks: List[Dict[str, Any]] = []
    
    def embed_chunks(
        self,
        chunks: List[Dict[str, Any]],
        content_key: str = "content",
        show_progress: bool = True
    ) -> 'ChunkEmbedder':
        """Embed all chunks."""
        self._chunks = chunks
        texts = [c[content_key] for c in chunks]
        
        print(f"Embedding {len(texts)} chunks...")
        self._embeddings = self.embedder.embed_documents(texts, show_progress)
        print(f"Embeddings shape: {self._embeddings.shape}")
        
        return self
    
    @property
    def embeddings(self) -> np.ndarray:
        """Get embeddings array."""
        return self._embeddings
    
    @property
    def num_chunks(self) -> int:
        """Number of embedded chunks."""
        return len(self._chunks)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks."""
        if len(self._embeddings) == 0:
            return []
        
        query_emb = self.embedder.embed_queries([query], show_progress=False)[0]
        similarities = np.dot(self._embeddings, query_emb)
        
        if threshold is not None:
            valid_idx = np.where(similarities >= threshold)[0]
            top_idx = valid_idx[np.argsort(similarities[valid_idx])[::-1][:top_k]]
        else:
            top_idx = np.argsort(similarities)[::-1][:top_k]
        
        return [
            {**self._chunks[i], 'score': float(similarities[i])}
            for i in top_idx
        ]
    
    def batch_search(
        self,
        queries: List[str],
        top_k: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """Batch search for multiple queries."""
        if len(self._embeddings) == 0:
            return [[] for _ in queries]
        
        query_embs = self.embedder.embed_queries(queries, show_progress=False)
        all_sims = np.dot(query_embs, self._embeddings.T)
        
        results = []
        for sims in all_sims:
            top_idx = np.argsort(sims)[::-1][:top_k]
            results.append([
                {**self._chunks[i], 'score': float(sims[i])}
                for i in top_idx
            ])
        return results


if __name__ == "__main__":
    from pathlib import Path
    from chunker import BookChunker, ChunkConfig
    from loader import load_books
    
    dataset_dir = Path(__file__).parent.parent / "Dataset"
    
    print("=" * 60)
    print("Embedder Test")
    print("=" * 60)
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load and chunk
    books = load_books(str(dataset_dir))
    chunker = BookChunker(ChunkConfig(chunk_size=200, overlap_front=50, overlap_back=50))
    chunker.chunk_books(books)
    test_chunks = chunker.chunks[:50]
    
    # Embed
    embedder = ChunkEmbedder(EmbedderConfig(matryoshka_dim=256, batch_size=16))
    embedder.embed_chunks(test_chunks)
    
    # Search
    query = "the captain sailed across the ocean"
    results = embedder.search(query, top_k=3)
    
    print(f"\nQuery: '{query}'")
    for i, r in enumerate(results):
        print(f"{i+1}. Score: {r['score']:.4f} | {r['content'][:80]}...")
    
    print("\nTest complete!")
