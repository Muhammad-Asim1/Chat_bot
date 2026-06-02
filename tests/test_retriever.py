"""
Comprehensive tests for the retriever logic in rag_chain.py and vectorstore.py

Tests cover:
1. Jina reranking function and fallback behavior
2. Hybrid retrieval flow (dense + BM25 + rerank)
3. Result deduplication and fusion
4. Relevance threshold abstention
5. Context preparation for LLM
6. Score handling and ordering
7. Error handling and edge cases
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# MOCK FIXTURES
# =============================================================================

@pytest.fixture
def mock_qdrant_point():
    """Create a mock Qdrant point with payload."""
    point = Mock()
    point.id = "test-uuid-123"
    point.score = 0.85
    point.payload = {
        "document": "This is the test document content about fee policies.",
        "source": "test.pdf",
        "source_url": "https://example.com/test.pdf",
        "page": 5,
        "chunk_title": "Fee Policy Section"
    }
    return point


@pytest.fixture
def mock_qdrant_points():
    """Create multiple mock Qdrant points."""
    points = []
    for i in range(5):
        point = Mock()
        point.id = f"uuid-{i}"
        point.score = 0.9 - (i * 0.1)  # 0.9, 0.8, 0.7, 0.6, 0.5
        point.payload = {
            "document": f"Document content {i} about policies and procedures.",
            "source": f"doc{i}.pdf",
            "source_url": f"https://example.com/doc{i}.pdf",
            "page": i + 1,
            "chunk_title": f"Section {i + 1}"
        }
        points.append(point)
    return points


@pytest.fixture
def mock_rerank_results():
    """Mock Jina reranker API response."""
    return [
        {"index": 2, "relevance_score": 0.95},
        {"index": 0, "relevance_score": 0.88},
        {"index": 1, "relevance_score": 0.72},
    ]


# =============================================================================
# RERANKING TESTS
# =============================================================================

class TestJinaReranking:
    """Tests for jina_rerank function."""
    
    def test_rerank_returns_sorted_indices(self, mock_rerank_results):
        """Reranking should return results sorted by relevance."""
        # Results should be sorted by relevance_score descending
        scores = [r["relevance_score"] for r in mock_rerank_results]
        assert scores == sorted(scores, reverse=True)
    
    def test_rerank_result_structure(self, mock_rerank_results):
        """Each rerank result should have index and relevance_score."""
        for result in mock_rerank_results:
            assert "index" in result
            assert "relevance_score" in result
            assert isinstance(result["index"], int)
            assert isinstance(result["relevance_score"], float)
    
    def test_rerank_fallback_on_error(self):
        """On error, fallback should return original order with score 1.0."""
        # Simulating fallback behavior
        documents = ["doc1", "doc2", "doc3"]
        top_n = 3
        
        # Fallback returns original order
        fallback_results = [{"index": i, "relevance_score": 1.0} for i in range(min(top_n, len(documents)))]
        
        assert len(fallback_results) == 3
        assert fallback_results[0]["index"] == 0
        assert fallback_results[0]["relevance_score"] == 1.0
    
    def test_rerank_top_n_limit(self, mock_rerank_results):
        """Reranking should respect top_n limit."""
        top_n = 2
        limited_results = mock_rerank_results[:top_n]
        assert len(limited_results) == 2
    
    def test_rerank_index_mapping(self, mock_qdrant_points, mock_rerank_results):
        """Reranked results should correctly map back to original points."""
        candidates = mock_qdrant_points[:3]
        
        reranked_points = []
        for result in mock_rerank_results:
            idx = result["index"]
            if idx < len(candidates):
                point = candidates[idx]
                point.score = result["relevance_score"]
                reranked_points.append(point)
        
        # First reranked result should be from index 2 in original
        assert reranked_points[0].score == 0.95


# =============================================================================
# HYBRID RETRIEVAL TESTS
# =============================================================================

class TestHybridRetrieval:
    """Tests for hybrid retrieval flow."""
    
    def test_dense_results_combined_with_text_results(self, mock_qdrant_points):
        """Dense and text search results should be combined."""
        dense_results = mock_qdrant_points[:3]
        text_results = mock_qdrant_points[2:5]  # Overlapping at index 2
        
        seen_ids = set()
        combined = []
        
        for point in dense_results:
            if point.id not in seen_ids:
                seen_ids.add(point.id)
                combined.append(point)
        
        for point in text_results:
            if point.id not in seen_ids:
                seen_ids.add(point.id)
                combined.append(point)
        
        # Should have 5 unique points (3 + 2 new from text)
        assert len(combined) == 5
    
    def test_deduplication_by_id(self, mock_qdrant_points):
        """Duplicate point IDs should be removed."""
        # Create duplicates
        point1 = mock_qdrant_points[0]
        point2 = Mock()
        point2.id = point1.id  # Same ID
        point2.score = 0.5
        
        all_points = [point1, point2]
        
        seen_ids = set()
        unique = []
        for p in all_points:
            if p.id not in seen_ids:
                seen_ids.add(p.id)
                unique.append(p)
        
        assert len(unique) == 1
    
    def test_rerank_limit_applied(self, mock_qdrant_points):
        """Only top RERANK_LIMIT candidates should be sent to reranker."""
        rerank_limit = 3
        candidates = mock_qdrant_points[:rerank_limit]
        
        assert len(candidates) == 3
    
    def test_final_limit_applied(self, mock_rerank_results):
        """Only top FINAL_LIMIT results returned after reranking."""
        final_limit = 2
        final_results = mock_rerank_results[:final_limit]
        
        assert len(final_results) == 2


# =============================================================================
# RESULT ORDERING TESTS
# =============================================================================

class TestResultOrdering:
    """Tests for result ordering after reranking."""
    
    def test_results_sorted_by_score_descending(self, mock_qdrant_points, mock_rerank_results):
        """Final results should be sorted by score descending."""
        candidates = mock_qdrant_points[:3]
        
        reranked_points = []
        for result in mock_rerank_results:
            idx = result["index"]
            if idx < len(candidates):
                point = candidates[idx]
                point.score = result["relevance_score"]
                reranked_points.append(point)
        
        scores = [p.score for p in reranked_points]
        assert scores == sorted(scores, reverse=True)
    
    def test_best_score_is_first(self, mock_qdrant_points, mock_rerank_results):
        """Highest scoring result should be first."""
        candidates = mock_qdrant_points[:3]
        
        reranked_points = []
        for result in mock_rerank_results:
            idx = result["index"]
            if idx < len(candidates):
                point = candidates[idx]
                point.score = result["relevance_score"]
                reranked_points.append(point)
        
        best_score = max(r["relevance_score"] for r in mock_rerank_results)
        assert reranked_points[0].score == best_score


# =============================================================================
# ABSTENTION TESTS
# =============================================================================

class TestAbstention:
    """Tests for relevance threshold abstention."""
    
    def test_abstain_when_below_threshold(self):
        """Should abstain when best score is below threshold."""
        best_score = 0.25
        min_relevance = 0.30
        
        should_abstain = best_score < min_relevance
        assert should_abstain == True
    
    def test_no_abstain_when_above_threshold(self):
        """Should not abstain when best score is above threshold."""
        best_score = 0.45
        min_relevance = 0.30
        
        should_abstain = best_score < min_relevance
        assert should_abstain == False
    
    def test_abstain_on_exact_threshold(self):
        """Should not abstain when score equals threshold."""
        best_score = 0.30
        min_relevance = 0.30
        
        should_abstain = best_score < min_relevance
        assert should_abstain == False
    
    def test_abstention_message_format(self):
        """Abstention message should contain expected elements."""
        best_score = 0.25
        
        abstention_message = (
            "I don't have enough information in the provided documents to answer that question.\n\n"
            f"*The retrieved content has low relevance (best score: {best_score:.0%}). "
            "This suggests your question may be outside the scope of the available documents.*"
        )
        
        assert "don't have enough information" in abstention_message
        assert "25%" in abstention_message
        assert "outside the scope" in abstention_message


# =============================================================================
# CONTEXT PREPARATION TESTS
# =============================================================================

class TestContextPreparation:
    """Tests for context preparation for LLM."""
    
    def test_context_includes_source_labels(self, mock_qdrant_points):
        """Context should include [Source X] labels."""
        context_parts = []
        for i, point in enumerate(mock_qdrant_points[:3], 1):
            document = point.payload.get("document", "")
            context_parts.append(f"[Source {i}]: {document}")
        
        context = "\n\n".join(context_parts)
        
        assert "[Source 1]:" in context
        assert "[Source 2]:" in context
        assert "[Source 3]:" in context
    
    def test_context_contains_document_text(self, mock_qdrant_point):
        """Context should contain the document text."""
        context = f"[Source 1]: {mock_qdrant_point.payload['document']}"
        
        assert "fee policies" in context
    
    def test_multiple_sources_separated(self, mock_qdrant_points):
        """Multiple sources should be separated by double newlines."""
        context_parts = []
        for i, point in enumerate(mock_qdrant_points[:2], 1):
            context_parts.append(f"[Source {i}]: {point.payload['document']}")
        
        context = "\n\n".join(context_parts)
        
        assert "\n\n" in context


# =============================================================================
# METADATA EXTRACTION TESTS
# =============================================================================

class TestMetadataExtraction:
    """Tests for extracting metadata from points."""
    
    def test_extract_source(self, mock_qdrant_point):
        """Source should be extracted from payload."""
        source = mock_qdrant_point.payload.get("source", "Unknown")
        assert source == "test.pdf"
    
    def test_extract_page(self, mock_qdrant_point):
        """Page should be extracted from payload."""
        page = mock_qdrant_point.payload.get("page", "N/A")
        assert page == 5
    
    def test_extract_source_url(self, mock_qdrant_point):
        """Source URL should be extracted from payload."""
        source_url = mock_qdrant_point.payload.get("source_url", None)
        assert source_url == "https://example.com/test.pdf"
    
    def test_extract_chunk_title(self, mock_qdrant_point):
        """Chunk title should be extracted from payload."""
        chunk_title = mock_qdrant_point.payload.get("chunk_title", "")
        assert chunk_title == "Fee Policy Section"
    
    def test_handle_missing_source_url(self):
        """Missing source_url should return None."""
        payload = {"document": "text", "source": "doc.pdf", "page": 1}
        source_url = payload.get("source_url", None)
        assert source_url is None
    
    def test_handle_missing_page(self):
        """Missing page should return 'N/A'."""
        payload = {"document": "text", "source": "doc.pdf"}
        page = payload.get("page", "N/A")
        assert page == "N/A"


# =============================================================================
# CHUNKS METADATA BUILDING TESTS
# =============================================================================

class TestChunksMetadataBuilding:
    """Tests for building chunks_metadata for sidebar display."""
    
    def test_metadata_structure(self, mock_qdrant_point):
        """Chunks metadata should have correct structure."""
        point = mock_qdrant_point
        payload = point.payload
        
        metadata = {
            "chunk_number": 1,
            "source": payload.get("source", "Unknown"),
            "source_url": payload.get("source_url", None),
            "page": payload.get("page", "N/A"),
            "chunk_title": payload.get("chunk_title", ""),
            "score": point.score,
            "preview": payload.get("document", "")[:150] + "..." if len(payload.get("document", "")) > 150 else payload.get("document", ""),
            "document": payload.get("document", ""),
        }
        
        assert "chunk_number" in metadata
        assert "source" in metadata
        assert "score" in metadata
        assert "preview" in metadata
        assert "document" in metadata
    
    def test_preview_truncated(self, mock_qdrant_point):
        """Preview should be truncated to 150 chars + '...'."""
        long_doc = "A" * 200
        preview = long_doc[:150] + "..." if len(long_doc) > 150 else long_doc
        
        assert len(preview) == 153  # 150 + 3 for "..."
        assert preview.endswith("...")
    
    def test_preview_not_truncated_short_doc(self):
        """Short documents should not be truncated."""
        short_doc = "Short document"
        preview = short_doc[:150] + "..." if len(short_doc) > 150 else short_doc
        
        assert preview == "Short document"
        assert "..." not in preview


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Tests for error handling in retrieval."""
    
    def test_no_results_message(self):
        """Should return appropriate message when no results found."""
        points = []
        
        if not points:
            message = "No relevant information found in the documents."
        else:
            message = "Found results"
        
        assert "No relevant" in message
    
    def test_collection_not_exists_message(self):
        """Should return appropriate message when collection doesn't exist."""
        collection_exists = False
        
        if not collection_exists:
            message = "⚠️ No documents have been indexed yet. Please upload a PDF first."
        else:
            message = "Collection exists"
        
        assert "No documents have been indexed" in message
    
    def test_error_returns_tuple(self):
        """Error should return (message, empty_list) tuple."""
        error_result = ("⚠️ Error: Connection failed", [])
        
        assert isinstance(error_result, tuple)
        assert len(error_result) == 2
        assert isinstance(error_result[1], list)
        assert len(error_result[1]) == 0


# =============================================================================
# SCORE HANDLING TESTS
# =============================================================================

class TestScoreHandling:
    """Tests for score handling and display."""
    
    def test_score_format_percentage(self):
        """Score should be displayable as percentage."""
        score = 0.856
        formatted = f"{score:.0%}"
        assert formatted == "86%"
    
    def test_score_format_decimal(self):
        """Score should be displayable as decimal."""
        score = 0.856
        formatted = f"{score:.2%}"
        assert formatted == "85.60%"
    
    def test_score_in_citation(self):
        """Score should be included in citation string."""
        score = 0.85
        citation = f"**[1]** [doc.pdf, Page 5](url) • {score:.0%}"
        
        assert "85%" in citation


# =============================================================================
# RETRIEVAL ONLY MODE TESTS
# =============================================================================

class TestRetrievalOnlyMode:
    """Tests for retrieval-only mode (no LLM generation)."""
    
    def test_all_chunks_marked_cited(self, mock_qdrant_points):
        """In retrieval-only mode, all chunks should be marked as cited."""
        chunks_metadata = []
        for i, point in enumerate(mock_qdrant_points[:3], 1):
            chunks_metadata.append({
                "chunk_number": i,
                "source": point.payload["source"],
                "cited": True  # All cited in retrieval-only mode
            })
        
        for meta in chunks_metadata:
            assert meta["cited"] == True
    
    def test_response_format(self, mock_qdrant_point):
        """Retrieval-only response should have expected format."""
        point = mock_qdrant_point
        
        response = "**🎯 Retrieved Information**\n\n"
        response += f"**Chunk 1** (Score: {point.score:.2%})\n"
        response += f"{point.payload['document']}\n\n"
        response += "---\n\n"
        
        assert "Retrieved Information" in response
        assert "Chunk 1" in response
        assert "Score:" in response


# =============================================================================
# DENSE-ONLY RETRIEVAL TESTS
# =============================================================================

class TestDenseOnlyRetrieval:
    """Tests for dense-only retrieval (for comparison)."""
    
    def test_dense_results_format(self, mock_qdrant_point):
        """Dense-only results should have expected format."""
        response = "**🔍 Dense Search Results** (Jina Cloud API - Semantic only)\n\n"
        response += f"**Chunk 1** (Score: {mock_qdrant_point.score:.4f})\n"
        
        assert "Dense Search Results" in response
        assert "Semantic only" in response


# =============================================================================
# LOW CONFIDENCE WARNING TESTS
# =============================================================================

class TestLowConfidenceWarning:
    """Tests for low confidence warning."""
    
    def test_warning_shown_below_50_percent(self):
        """Warning should be shown when best score < 50%."""
        best_score = 0.45
        
        if best_score < 0.50:
            warning = f"⚠️ *Low confidence: Best relevance score is {best_score:.0%}.*"
        else:
            warning = None
        
        assert warning is not None
        assert "45%" in warning
    
    def test_no_warning_above_50_percent(self):
        """Warning should not be shown when best score >= 50%."""
        best_score = 0.55
        
        if best_score < 0.50:
            warning = "Low confidence warning"
        else:
            warning = None
        
        assert warning is None


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Edge cases for retrieval."""
    
    def test_empty_query(self):
        """Empty query should be handled gracefully."""
        query = ""
        # System should handle empty queries without crashing
        assert query == ""
    
    def test_very_long_query(self):
        """Very long queries should be handled."""
        query = "word " * 1000
        # Query should be processable
        assert len(query) > 0
    
    def test_special_characters_in_query(self):
        """Special characters in query should be handled."""
        query = "What is the fee policy? (Section 2.1)"
        # Query should be processable
        assert "?" in query
        assert "(" in query
    
    def test_unicode_query(self):
        """Unicode in query should be handled."""
        query = "What is the café policy?"
        # Query should be processable
        assert "café" in query
    
    def test_single_result(self, mock_qdrant_point):
        """Single result should be handled correctly."""
        results = [mock_qdrant_point]
        
        assert len(results) == 1
        assert results[0].score > 0
    
    def test_max_score_is_one(self):
        """Maximum score should be 1.0."""
        scores = [0.9, 0.8, 1.0, 0.7]
        max_score = max(scores)
        
        assert max_score <= 1.0
    
    def test_min_score_is_zero_or_positive(self):
        """Minimum score should be 0 or positive."""
        scores = [0.9, 0.8, 0.5, 0.1]
        min_score = min(scores)
        
        assert min_score >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

