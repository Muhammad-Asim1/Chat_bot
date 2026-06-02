"""
Shared pytest fixtures and configuration for the test suite.

This file is automatically loaded by pytest and makes fixtures available
to all test modules without explicit imports.
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# SAMPLE TEXT FIXTURES
# =============================================================================

@pytest.fixture
def sample_policy_text():
    """Sample policy document text for testing."""
    return """
NUST FEE POLICY SALIENT FEATURES

1. Introduction

This document outlines the fee payment policies and procedures for all students.

2. Fee Payment Deadlines

Students must pay their tuition fee within 15 days of invoice issuance on QALAM.
International students are charged in advance on an annual basis.
A 2% fine is applied if fees are paid after the deadline.

3. Financial Assistance

Students facing financial challenges should contact the institute for:
• Instalment options
• Fee deferment
• Scholarship opportunities

SANCTIONS FOR NON-PAYMENT

Students who fail to deposit fees by the extended deadline will have a sanction 
placed and will not be allowed to register for the semester.
"""


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing retrieval and citations."""
    return [
        {
            "document": "Students must pay their tuition fee within 15 days of invoice issuance on QALAM.",
            "source": "NUST-Fee-Policy.pdf",
            "source_url": "https://nust.edu.pk/fee-policy.pdf",
            "page": 1,
            "chunk_title": "Fee Payment Deadlines"
        },
        {
            "document": "International students are charged in advance on an annual basis.",
            "source": "NUST-Fee-Policy.pdf",
            "source_url": "https://nust.edu.pk/fee-policy.pdf",
            "page": 1,
            "chunk_title": "International Students"
        },
        {
            "document": "A 2% fine is applied if fees are paid after the deadline.",
            "source": "NUST-Fee-Policy.pdf",
            "source_url": "https://nust.edu.pk/fee-policy.pdf",
            "page": 2,
            "chunk_title": "Late Payment Penalties"
        }
    ]


@pytest.fixture
def sample_llm_response():
    """Sample LLM JSON response for testing citation parsing."""
    return {
        "answer": "Students must pay fees within 15 days [1]. International students pay annually [2]. Late payment incurs a 2% fine [3].",
        "highlights": {
            "1": "pay their tuition fee within 15 days",
            "2": "charged in advance on an annual basis",
            "3": "2% fine is applied"
        }
    }


@pytest.fixture
def sample_llm_json_string():
    """Sample LLM response as JSON string."""
    return '''{"answer": "The fee deadline is 15 days [1].", "highlights": {"1": "within 15 days of invoice issuance"}}'''


# =============================================================================
# MOCK FIXTURES
# =============================================================================

@pytest.fixture
def mock_point_factory():
    """Factory for creating mock Qdrant points."""
    from unittest.mock import Mock
    
    def create_point(id, score, document, source, page, source_url=None, chunk_title=""):
        point = Mock()
        point.id = id
        point.score = score
        point.payload = {
            "document": document,
            "source": source,
            "source_url": source_url,
            "page": page,
            "chunk_title": chunk_title
        }
        return point
    
    return create_point


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def test_config():
    """Test configuration settings."""
    return {
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "prefetch_limit": 10,
        "rerank_limit": 6,
        "final_limit": 3,
        "min_relevance_score": 0.30,
        "strict_citation_mode": True
    }


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def assert_normalized_equal(text1: str, text2: str):
    """Assert two texts are equal after normalization."""
    from rag_chain import normalize_text_for_matching
    assert normalize_text_for_matching(text1) == normalize_text_for_matching(text2)

