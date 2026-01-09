"""
Tests for the verification pipeline.
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


# ============================================================================
# Unit Tests (no LLM required)
# ============================================================================

class TestBM25Retriever:
    """Tests for BM25 retriever."""
    
    def test_build_index(self):
        """Test BM25 index building."""
        from pipeline.verifier import BM25Retriever
        
        chunks = [
            {"content": "The captain sailed across the ocean."},
            {"content": "Mary went to the garden to pick flowers."},
            {"content": "The ship encountered a terrible storm."},
        ]
        
        retriever = BM25Retriever(chunks)
        assert retriever._index is not None
        assert len(retriever.chunks) == 3
    
    def test_search_basic(self):
        """Test BM25 search."""
        from pipeline.verifier import BM25Retriever
        
        chunks = [
            {"content": "The captain sailed across the ocean.", "id": 1},
            {"content": "Mary went to the garden to pick flowers.", "id": 2},
            {"content": "The ship encountered a terrible storm at sea.", "id": 3},
        ]
        
        retriever = BM25Retriever(chunks)
        results = retriever.search("ship storm ocean", top_k=2)
        
        assert len(results) == 2
        # Ship/storm/ocean should rank higher
        assert results[0]['id'] in [1, 3]
    
    def test_search_empty_query(self):
        """Test search with empty query."""
        from pipeline.verifier import BM25Retriever
        
        chunks = [{"content": "Some text", "id": 1}]
        retriever = BM25Retriever(chunks)
        results = retriever.search("", top_k=1)
        
        assert len(results) == 1


class TestAggregator:
    """Tests for the aggregator."""
    
    def test_all_supports(self):
        """Test aggregation when all claims support."""
        from pipeline.verifier import Aggregator, VerificationResult
        
        results = [
            VerificationResult("claim1", "supports", 0.9, [], "reason1"),
            VerificationResult("claim2", "supports", 0.8, [], "reason2"),
        ]
        
        agg = Aggregator()
        prediction, details = agg.aggregate(results)
        
        assert prediction == 1  # consistent
        assert details['counts']['supports'] == 2
    
    def test_high_confidence_contradiction(self):
        """Test that high-confidence contradiction triggers contradict."""
        from pipeline.verifier import Aggregator, VerificationResult
        
        results = [
            VerificationResult("claim1", "supports", 0.9, [], "reason1"),
            VerificationResult("claim2", "contradicts", 0.8, [], "reason2"),
        ]
        
        agg = Aggregator()
        prediction, details = agg.aggregate(results)
        
        assert prediction == 0  # contradict
    
    def test_low_confidence_contradiction(self):
        """Test that low-confidence contradictions may not trigger."""
        from pipeline.verifier import Aggregator, VerificationResult
        
        results = [
            VerificationResult("claim1", "supports", 0.9, [], "reason1"),
            VerificationResult("claim2", "supports", 0.8, [], "reason2"),
            VerificationResult("claim3", "contradicts", 0.3, [], "reason3"),  # Low confidence
        ]
        
        agg = Aggregator()
        prediction, details = agg.aggregate(results)
        
        # With 2 supports and 1 low-conf contradict, should still be consistent
        # Support score = 1.7, Contradiction score = 0.3
        assert prediction == 1
    
    def test_empty_results(self):
        """Test aggregation with no results."""
        from pipeline.verifier import Aggregator
        
        agg = Aggregator()
        prediction, details = agg.aggregate([])
        
        assert prediction == 1  # Default to consistent
        assert 'No claims' in details['reason']


class TestClaimExtractor:
    """Tests for claim extraction parsing."""
    
    def test_parse_json_array(self):
        """Test parsing JSON array response."""
        from pipeline.verifier import ClaimExtractor
        
        mock_llm = Mock()
        extractor = ClaimExtractor(mock_llm)
        
        response = '["fact one", "fact two", "fact three"]'
        claims = extractor._parse_claims(response)
        
        assert len(claims) == 3
        assert claims[0] == "fact one"
    
    def test_parse_json_with_noise(self):
        """Test parsing JSON with surrounding text."""
        from pipeline.verifier import ClaimExtractor
        
        mock_llm = Mock()
        extractor = ClaimExtractor(mock_llm)
        
        response = 'Here are the facts:\n["fact one", "fact two"]\nDone.'
        claims = extractor._parse_claims(response)
        
        assert len(claims) == 2
    
    def test_parse_fallback(self):
        """Test fallback parsing when JSON fails."""
        from pipeline.verifier import ClaimExtractor
        
        mock_llm = Mock()
        extractor = ClaimExtractor(mock_llm)
        
        response = """1. First fact about the character
2. Second fact about the story
3. Third fact here"""
        claims = extractor._parse_claims(response)
        
        assert len(claims) >= 2


class TestQueryGenerator:
    """Tests for query generation parsing."""
    
    def test_parse_queries(self):
        """Test parsing query response."""
        from pipeline.verifier import QueryGenerator
        
        mock_llm = Mock()
        gen = QueryGenerator(mock_llm)
        
        response = '["query one", "query two"]'
        queries = gen._parse_queries(response)
        
        assert len(queries) == 2


class TestClaimVerifier:
    """Tests for verification result parsing."""
    
    def test_parse_json_result(self):
        """Test parsing JSON verification result."""
        from pipeline.verifier import ClaimVerifier
        
        mock_llm = Mock()
        verifier = ClaimVerifier(mock_llm)
        
        response = '{"verdict": "contradicts", "confidence": 0.85, "reasoning": "The evidence shows..."}'
        result = verifier._parse_verification(response)
        
        assert result['verdict'] == 'contradicts'
        assert result['confidence'] == 0.85
    
    def test_parse_fallback_contradict(self):
        """Test fallback when JSON fails but keywords present."""
        from pipeline.verifier import ClaimVerifier
        
        mock_llm = Mock()
        verifier = ClaimVerifier(mock_llm)
        
        response = "This claim clearly contradicts the evidence in the novel..."
        result = verifier._parse_verification(response)
        
        assert result['verdict'] == 'contradicts'
    
    def test_parse_fallback_support(self):
        """Test fallback for support keyword."""
        from pipeline.verifier import ClaimVerifier
        
        mock_llm = Mock()
        verifier = ClaimVerifier(mock_llm)
        
        response = "The evidence supports this claim strongly."
        result = verifier._parse_verification(response)
        
        assert result['verdict'] == 'supports'


class TestReranker:
    """Tests for reranker score parsing."""
    
    def test_parse_score(self):
        """Test parsing score from response."""
        from pipeline.verifier import Reranker
        
        mock_llm = Mock()
        reranker = Reranker(mock_llm)
        
        assert reranker._parse_score("8") == 8.0
        assert reranker._parse_score("Score: 7.5") == 7.5
        assert reranker._parse_score("The relevance is 9 out of 10") == 9.0
    
    def test_parse_score_clamping(self):
        """Test score clamping to [0, 10]."""
        from pipeline.verifier import Reranker
        
        mock_llm = Mock()
        reranker = Reranker(mock_llm)
        
        assert reranker._parse_score("15") == 10.0
        assert reranker._parse_score("-5") == 5.0  # Fallback


# ============================================================================
# Integration Tests (require Docker + GPU)
# ============================================================================

@pytest.mark.slow
class TestHybridRetriever:
    """Integration tests for hybrid retrieval."""
    
    def test_hybrid_search(self):
        """Test hybrid BM25 + vector search."""
        from pipeline.verifier import HybridRetriever
        from pipeline.embedder import ChunkEmbedder
        
        chunks = [
            {"content": "The captain commanded the ship through storms.", "chunk_id": "1"},
            {"content": "Mary loved to read books in the garden.", "chunk_id": "2"},
            {"content": "The vessel sailed across the Atlantic ocean.", "chunk_id": "3"},
        ]
        
        # Create embedder and embed chunks
        embedder = ChunkEmbedder()
        embedder.embed_chunks(chunks)
        
        # Create hybrid retriever
        retriever = HybridRetriever(chunks, embedder)
        
        results = retriever.search("ship sailing ocean", top_k=2)
        
        assert len(results) <= 2
        assert all('hybrid_score' in r for r in results)


@pytest.mark.slow
class TestFullPipeline:
    """Full pipeline integration tests."""
    
    @pytest.fixture
    def setup_pipeline(self):
        """Set up the full pipeline with real data."""
        from pipeline.loader import load_books
        from pipeline.chunker import BookChunker, ChunkConfig
        from pipeline.embedder import ChunkEmbedder
        from pipeline.verifier import VerificationPipeline
        
        dataset_dir = Path(__file__).parent.parent / "Dataset"
        
        # Load books
        books = load_books(str(dataset_dir))
        
        # Chunk with smaller size for testing
        config = ChunkConfig(chunk_size=300, overlap_front=50, overlap_back=50)
        chunker = BookChunker(config)
        chunker.chunk_books(books)
        
        # Embed
        embedder = ChunkEmbedder()
        embedder.embed_chunks(chunker.chunks)
        
        # Create pipeline
        pipeline = VerificationPipeline(chunker.chunks, embedder)
        
        return pipeline
    
    def test_verify_single_backstory(self, setup_pipeline):
        """Test verifying a single backstory."""
        pipeline = setup_pipeline
        
        backstory = "Paganel fell in love with geography after reading Captain Cook's journal at age twelve."
        
        prediction, details = pipeline.verify_backstory(
            backstory=backstory,
            character="Jacques Paganel",
            book_name="In Search of the Castaways",
            verbose=True
        )
        
        assert prediction in [0, 1]
        assert 'claims' in details
        assert 'verdicts' in details
    
    def test_evaluate_samples(self, setup_pipeline):
        """Test evaluating multiple samples."""
        from pipeline.verifier import Evaluator
        import pandas as pd
        
        pipeline = setup_pipeline
        evaluator = Evaluator(pipeline)
        
        # Load a few samples
        dataset_dir = Path(__file__).parent.parent / "Dataset"
        df = pd.read_csv(dataset_dir / "train.csv")
        
        samples = df.head(3).to_dict('records')
        
        summary = evaluator.evaluate_dataset(samples, verbose=True)
        
        assert summary['total_samples'] == 3
        assert 0 <= summary['accuracy'] <= 1


# ============================================================================
# Smoke Test
# ============================================================================

@pytest.mark.slow
def test_smoke_test():
    """
    Smoke test: Run full pipeline on 2 samples.
    Should complete in < 5 minutes.
    """
    from pathlib import Path
    import pandas as pd
    from pipeline.loader import load_books
    from pipeline.chunker import BookChunker, ChunkConfig
    from pipeline.embedder import ChunkEmbedder
    from pipeline.verifier import VerificationPipeline, Evaluator
    
    print("\n" + "="*60)
    print("SMOKE TEST - Full Pipeline")
    print("="*60)
    
    dataset_dir = Path(__file__).parent.parent / "Dataset"
    
    # Step 1: Load books
    print("\n1. Loading books...")
    books = load_books(str(dataset_dir))
    print(f"   Loaded {len(books)} books")
    
    # Step 2: Chunk
    print("\n2. Chunking books...")
    config = ChunkConfig(chunk_size=400, overlap_front=100, overlap_back=100)
    chunker = BookChunker(config)
    chunker.chunk_books(books)
    print(f"   Created {len(chunker.chunks)} chunks")
    
    # Step 3: Embed
    print("\n3. Embedding chunks...")
    embedder = ChunkEmbedder()
    embedder.embed_chunks(chunker.chunks)
    print(f"   Embeddings shape: {embedder.embeddings.shape}")
    
    # Step 4: Create pipeline
    print("\n4. Creating verification pipeline...")
    pipeline = VerificationPipeline(chunker.chunks, embedder)
    
    # Step 5: Load test samples
    print("\n5. Loading test samples...")
    df = pd.read_csv(dataset_dir / "train.csv")
    
    # Get 1 consistent and 1 contradict sample
    consistent_sample = df[df['label'] == 'consistent'].iloc[0].to_dict()
    contradict_sample = df[df['label'] == 'contradict'].iloc[0].to_dict()
    samples = [consistent_sample, contradict_sample]
    
    print(f"   Sample 1: {consistent_sample['char']} (consistent)")
    print(f"   Sample 2: {contradict_sample['char']} (contradict)")
    
    # Step 6: Evaluate
    print("\n6. Running evaluation...")
    evaluator = Evaluator(pipeline)
    summary = evaluator.evaluate_dataset(samples, verbose=True)
    
    # Assertions
    assert summary['total_samples'] == 2
    print(f"\n   Accuracy: {summary['accuracy']:.2%}")
    
    print("\n" + "="*60)
    print("SMOKE TEST COMPLETE")
    print("="*60)
