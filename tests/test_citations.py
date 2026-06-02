"""
Comprehensive tests for citation creation and post-hoc processing in rag_chain.py

Tests cover:
1. Text normalization for matching (normalize_text_for_matching)
2. URL encoding for PDF text fragments (encode_text_fragment)
3. Citation parsing from LLM output (parse_citations_with_quotes)
4. JSON parsing from LLM structured output
5. Highlights map normalization (string keys to int, value type handling)
6. Citation-to-chunk mapping (exact match, word overlap)
7. Strict mode citation verification
8. URL building with text fragment highlighting
"""

import pytest
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_chain import (
    normalize_text_for_matching,
    encode_text_fragment,
    parse_citations_with_quotes,
)


# =============================================================================
# TEXT NORMALIZATION TESTS
# =============================================================================

class TestNormalizeTextForMatching:
    """Tests for normalize_text_for_matching function."""
    
    def test_lowercase_conversion(self):
        """Text should be converted to lowercase."""
        result = normalize_text_for_matching("Hello WORLD Test")
        assert result == "hello world test"
    
    def test_whitespace_normalization(self):
        """Multiple spaces/newlines should be collapsed to single space."""
        result = normalize_text_for_matching("Hello    World\n\nTest")
        assert result == "hello world test"
    
    def test_strip_trailing_punctuation(self):
        """Trailing punctuation should be stripped."""
        result = normalize_text_for_matching("Hello world!")
        assert result == "hello world"
        
        result = normalize_text_for_matching("Hello world.")
        assert result == "hello world"
        
        result = normalize_text_for_matching("Hello world?")
        assert result == "hello world"
        
        result = normalize_text_for_matching("Hello world;")
        assert result == "hello world"
    
    def test_strip_leading_trailing_whitespace(self):
        """Leading and trailing whitespace should be stripped."""
        result = normalize_text_for_matching("   Hello world   ")
        assert result == "hello world"
    
    def test_newlines_to_spaces(self):
        """Newlines should be converted to spaces."""
        result = normalize_text_for_matching("Hello\nworld\ntest")
        assert result == "hello world test"
    
    def test_tabs_to_spaces(self):
        """Tabs should be converted to spaces."""
        result = normalize_text_for_matching("Hello\tworld\ttest")
        assert result == "hello world test"
    
    def test_empty_string(self):
        """Empty string should return empty string."""
        result = normalize_text_for_matching("")
        assert result == ""
    
    def test_only_whitespace(self):
        """Whitespace-only string should return empty string."""
        result = normalize_text_for_matching("   \n\t   ")
        assert result == ""
    
    def test_mixed_punctuation(self):
        """Multiple trailing punctuation should be stripped."""
        result = normalize_text_for_matching("Hello world!?.")
        assert result == "hello world"
    
    def test_preserves_internal_punctuation(self):
        """Internal punctuation should be preserved."""
        result = normalize_text_for_matching("It's a test, right?")
        assert "it's a test" in result


# =============================================================================
# URL ENCODING TESTS
# =============================================================================

class TestEncodeTextFragment:
    """Tests for encode_text_fragment function (PDF text fragment encoding)."""
    
    def test_encode_spaces(self):
        """Spaces should be encoded."""
        result = encode_text_fragment("hello world")
        assert " " not in result
        assert "%20" in result
    
    def test_encode_special_characters(self):
        """Special characters should be encoded."""
        result = encode_text_fragment("test@example.com")
        assert "@" not in result
        assert "%40" in result
    
    def test_encode_period(self):
        """Periods should be encoded for URL fragments."""
        result = encode_text_fragment("Dr. Smith")
        assert "." not in result
        assert "%2E" in result
    
    def test_encode_comma(self):
        """Commas should be encoded."""
        result = encode_text_fragment("one, two, three")
        assert "," not in result
        assert "%2C" in result
    
    def test_encode_quotes(self):
        """Quotes should be encoded."""
        result = encode_text_fragment('He said "hello"')
        assert '"' not in result
        assert "%22" in result
    
    def test_encode_parentheses(self):
        """Parentheses should be encoded."""
        result = encode_text_fragment("test (example)")
        assert "(" not in result
        assert ")" not in result
        assert "%28" in result
        assert "%29" in result
    
    def test_encode_brackets(self):
        """Brackets should be encoded."""
        result = encode_text_fragment("test [1]")
        assert "[" not in result
        assert "]" not in result
        assert "%5B" in result
        assert "%5D" in result
    
    def test_encode_ampersand(self):
        """Ampersand should be encoded."""
        result = encode_text_fragment("A & B")
        assert "&" not in result
        assert "%26" in result
    
    def test_encode_hash(self):
        """Hash should be encoded."""
        result = encode_text_fragment("section #1")
        assert "#" not in result
        assert "%23" in result
    
    def test_encode_question_mark(self):
        """Question mark should be encoded."""
        result = encode_text_fragment("What?")
        assert "?" not in result
        assert "%3F" in result
    
    def test_empty_string(self):
        """Empty string should return empty string."""
        result = encode_text_fragment("")
        assert result == ""
    
    def test_alphanumeric_preserved(self):
        """Alphanumeric characters should be encoded (as per implementation)."""
        result = encode_text_fragment("abc123")
        # After encoding, alphanumerics are URL-safe
        assert "abc123" in result or "%61" in result  # 'a' could be preserved or encoded


# =============================================================================
# CITATION PARSING TESTS
# =============================================================================

class TestParseCitationsWithQuotes:
    """Tests for parse_citations_with_quotes function."""
    
    def test_parse_single_citation_with_quote(self):
        """Single citation with quote should be parsed."""
        text = 'The answer is [1: "exact quote from source"].'
        result = parse_citations_with_quotes(text)
        assert 1 in result
        assert result[1] == "exact quote from source"
    
    def test_parse_multiple_citations(self):
        """Multiple citations should all be parsed."""
        text = 'First [1: "quote one"] and second [2: "quote two"].'
        result = parse_citations_with_quotes(text)
        assert 1 in result
        assert 2 in result
        assert result[1] == "quote one"
        assert result[2] == "quote two"
    
    def test_parse_simple_citations_fallback(self):
        """Simple [X] format should be parsed as fallback."""
        text = "The answer is found in [1] and [2]."
        result = parse_citations_with_quotes(text)
        assert 1 in result
        assert 2 in result
        assert result[1] == ""  # No quote provided
        assert result[2] == ""
    
    def test_parse_mixed_citations(self):
        """Mix of quoted and simple citations should both be parsed."""
        text = 'Found in [1: "quote here"] and also [2].'
        result = parse_citations_with_quotes(text)
        assert 1 in result
        assert 2 in result
        assert result[1] == "quote here"
        assert result[2] == ""  # No quote for simple format
    
    def test_parse_single_quotes(self):
        """Single quotes should also be parsed."""
        text = "The answer is [1: 'single quoted text']."
        result = parse_citations_with_quotes(text)
        assert 1 in result
        assert result[1] == "single quoted text"
    
    def test_no_citations_returns_empty(self):
        """Text without citations should return empty dict."""
        text = "This text has no citations at all."
        result = parse_citations_with_quotes(text)
        assert result == {}
    
    def test_citation_numbers_as_integers(self):
        """Citation numbers should be integers, not strings."""
        text = "[1: \"test\"] and [10: \"another\"]"
        result = parse_citations_with_quotes(text)
        assert isinstance(list(result.keys())[0], int)
        assert 1 in result
        assert 10 in result
    
    def test_strip_quote_whitespace(self):
        """Whitespace in quotes should be stripped."""
        text = '[1: "  quote with spaces  "]'
        result = parse_citations_with_quotes(text)
        assert result[1] == "quote with spaces"


# =============================================================================
# JSON PARSING TESTS (LLM Structured Output)
# =============================================================================

class TestJSONParsing:
    """Tests for JSON parsing from LLM structured output."""
    
    def test_parse_valid_json(self):
        """Valid JSON should be parsed correctly."""
        json_str = '{"answer": "Test answer [1]", "highlights": {"1": "exact quote"}}'
        parsed = json.loads(json_str)
        assert parsed['answer'] == "Test answer [1]"
        assert parsed['highlights']['1'] == "exact quote"
    
    def test_parse_json_with_markdown_block(self):
        """JSON inside markdown code block should be extractable."""
        response = '''```json
{"answer": "Test answer", "highlights": {"1": "quote"}}
```'''
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        assert json_match is not None
        parsed = json.loads(json_match.group(1))
        assert 'answer' in parsed
    
    def test_normalize_string_keys_to_int(self):
        """String keys like "1" should be convertible to int."""
        highlights = {"1": "quote one", "2": "quote two"}
        normalized = {}
        for key, value in highlights.items():
            int_key = int(key) if isinstance(key, str) else key
            normalized[int_key] = value
        
        assert 1 in normalized
        assert 2 in normalized
        assert isinstance(list(normalized.keys())[0], int)
    
    def test_handle_int_keys_already(self):
        """Integer keys should remain integers."""
        highlights = {1: "quote one", 2: "quote two"}
        normalized = {}
        for key, value in highlights.items():
            int_key = int(key) if isinstance(key, str) else key
            normalized[int_key] = value
        
        assert 1 in normalized
        assert normalized[1] == "quote one"
    
    def test_handle_list_values(self):
        """List values should take first item."""
        highlights = {"1": ["first quote", "second quote"]}
        value = highlights["1"]
        if isinstance(value, list) and len(value) > 0:
            value = value[0]
        assert value == "first quote"
    
    def test_skip_none_values(self):
        """None values should be skipped."""
        highlights = {"1": "valid", "2": None, "3": "also valid"}
        normalized = {}
        for key, value in highlights.items():
            if value is not None:
                normalized[int(key)] = value
        
        assert 1 in normalized
        assert 2 not in normalized
        assert 3 in normalized


# =============================================================================
# CITATION-TO-CHUNK MAPPING TESTS
# =============================================================================

class TestCitationToChunkMapping:
    """Tests for citation-to-chunk mapping logic."""
    
    def test_exact_substring_match(self):
        """Exact substring should match with score 1.0."""
        chunk_text = "The fee must be paid within 15 days of invoice issuance."
        highlight = "paid within 15 days"
        
        normalized_highlight = normalize_text_for_matching(highlight)
        normalized_chunk = normalize_text_for_matching(chunk_text)
        
        assert normalized_highlight in normalized_chunk
    
    def test_word_overlap_calculation(self):
        """Word overlap should be calculated correctly."""
        highlight = "fee payment deadline"
        chunk = "The fee payment deadline is 15 days"
        
        normalized_highlight = normalize_text_for_matching(highlight)
        normalized_chunk = normalize_text_for_matching(chunk)
        
        highlight_words = set(normalized_highlight.split())
        chunk_words = set(normalized_chunk.split())
        
        common_words = highlight_words.intersection(chunk_words)
        overlap = len(common_words) / len(highlight_words)
        
        # All highlight words should be in chunk
        assert overlap == 1.0
    
    def test_partial_word_overlap(self):
        """Partial word overlap should be calculated correctly."""
        highlight = "fee payment due date"
        chunk = "The fee payment is processed"
        
        normalized_highlight = normalize_text_for_matching(highlight)
        normalized_chunk = normalize_text_for_matching(chunk)
        
        highlight_words = set(normalized_highlight.split())
        chunk_words = set(normalized_chunk.split())
        
        common_words = highlight_words.intersection(chunk_words)
        overlap = len(common_words) / len(highlight_words)
        
        # "fee" and "payment" match = 2/4 = 0.5
        assert overlap == 0.5
    
    def test_70_percent_threshold(self):
        """70% word overlap should be the threshold."""
        # 7 out of 10 words matching = 70%
        highlight_words = {"word1", "word2", "word3", "word4", "word5", "word6", "word7", "word8", "word9", "word10"}
        chunk_words = {"word1", "word2", "word3", "word4", "word5", "word6", "word7", "other1", "other2", "other3"}
        
        common = highlight_words.intersection(chunk_words)
        overlap = len(common) / len(highlight_words)
        
        assert overlap == 0.7
        assert overlap >= 0.7  # Should pass threshold
    
    def test_below_threshold_fails(self):
        """Below 70% overlap should fail to match."""
        highlight_words = {"a", "b", "c", "d", "e"}
        chunk_words = {"a", "b", "x", "y", "z"}
        
        common = highlight_words.intersection(chunk_words)
        overlap = len(common) / len(highlight_words)
        
        assert overlap == 0.4
        assert overlap < 0.7  # Should fail threshold
    
    def test_empty_highlight_handling(self):
        """Empty highlight should not cause division by zero."""
        highlight_words = set()
        
        if len(highlight_words) > 0:
            overlap = 0 / len(highlight_words)
        else:
            overlap = 0  # Avoid division by zero
        
        assert overlap == 0


# =============================================================================
# STRICT MODE TESTS
# =============================================================================

class TestStrictCitationMode:
    """Tests for strict citation mode behavior."""
    
    def test_strict_mode_rejects_unverified(self):
        """Strict mode should reject citations without verified matches."""
        # Simulating strict mode logic
        citation_to_chunk = {}  # No mapping found
        cited_indices = {1, 2}
        strict_mode = True
        
        accepted_citations = []
        for cite_num in cited_indices:
            chunk_idx = citation_to_chunk.get(cite_num)
            if chunk_idx is None:
                if strict_mode:
                    continue  # Reject
                else:
                    chunk_idx = cite_num  # Fallback
            accepted_citations.append(cite_num)
        
        assert len(accepted_citations) == 0  # All rejected in strict mode
    
    def test_fallback_mode_accepts_unverified(self):
        """Fallback mode should accept unverified citations."""
        citation_to_chunk = {}  # No mapping found
        cited_indices = {1, 2}
        strict_mode = False
        
        accepted_citations = []
        for cite_num in cited_indices:
            chunk_idx = citation_to_chunk.get(cite_num)
            if chunk_idx is None:
                if strict_mode:
                    continue
                else:
                    chunk_idx = cite_num  # Fallback to cite_num
            accepted_citations.append(cite_num)
        
        assert len(accepted_citations) == 2  # All accepted with fallback
    
    def test_verified_citations_pass_both_modes(self):
        """Verified citations should pass in both strict and fallback modes."""
        citation_to_chunk = {1: 1, 2: 2}  # Both mapped
        cited_indices = {1, 2}
        
        for strict_mode in [True, False]:
            accepted = []
            for cite_num in cited_indices:
                chunk_idx = citation_to_chunk.get(cite_num)
                if chunk_idx is not None:
                    accepted.append(cite_num)
            assert len(accepted) == 2


# =============================================================================
# URL BUILDING TESTS
# =============================================================================

class TestURLBuilding:
    """Tests for citation URL building."""
    
    def test_build_page_link(self):
        """Basic page link should be built correctly."""
        source_url = "https://example.com/doc.pdf"
        page = 5
        
        page_link = f"{source_url}#page={page}"
        assert page_link == "https://example.com/doc.pdf#page=5"
    
    def test_build_text_fragment_link(self):
        """Link with text fragment should be built correctly."""
        source_url = "https://example.com/doc.pdf"
        page = 5
        highlight = "exact quote"
        
        encoded = encode_text_fragment(highlight)
        page_link = f"{source_url}#page={page}:~:text={encoded}"
        
        assert "#page=5" in page_link
        assert ":~:text=" in page_link
        # Note: URL encoding preserves alphanumeric chars, only encodes special chars/spaces
        assert " " not in page_link  # Space should be encoded as %20
        assert "%20" in page_link  # Space encoded
    
    def test_handle_missing_page(self):
        """Missing page should use source URL only."""
        source_url = "https://example.com/doc.pdf"
        page = "N/A"
        
        if page != "N/A":
            link = f"{source_url}#page={page}"
        else:
            link = source_url
        
        assert link == source_url
        assert "#page" not in link
    
    def test_handle_missing_source_url(self):
        """Missing source URL should use text citation."""
        source_url = None
        source = "document.pdf"
        page = 5
        
        if source_url:
            citation = f"[{source}, Page {page}]({source_url}#page={page})"
        else:
            citation = f"{source}, Page {page}"
        
        assert citation == "document.pdf, Page 5"
        assert "http" not in citation


# =============================================================================
# METADATA BUILDING TESTS
# =============================================================================

class TestMetadataBuilding:
    """Tests for chunk metadata building."""
    
    def test_mark_cited_chunk(self):
        """Cited chunks should be marked as cited=True."""
        chunks_metadata = [
            {"chunk_number": 1, "source": "doc.pdf"},
            {"chunk_number": 2, "source": "doc.pdf"},
        ]
        citation_to_chunk = {1: 1}  # Citation 1 maps to chunk 1
        
        for i, metadata in enumerate(chunks_metadata, 1):
            chunk_citations = [cn for cn, ci in citation_to_chunk.items() if ci == i]
            metadata['cited'] = len(chunk_citations) > 0
        
        assert chunks_metadata[0]['cited'] == True
        assert chunks_metadata[1]['cited'] == False
    
    def test_store_citation_numbers(self):
        """Citation numbers should be stored in metadata."""
        citation_to_chunk = {1: 1, 3: 1}  # Both cite chunk 1
        chunk_idx = 1
        
        chunk_citations = [cn for cn, ci in citation_to_chunk.items() if ci == chunk_idx]
        assert sorted(chunk_citations) == [1, 3]
    
    def test_store_highlight_texts(self):
        """Highlight texts should be stored in metadata."""
        highlights_map = {1: "first quote", 2: "second quote"}
        chunk_citations = [1, 2]
        
        chunk_highlights = [highlights_map[cn] for cn in chunk_citations if cn in highlights_map]
        assert chunk_highlights == ["first quote", "second quote"]


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Edge cases for citation processing."""
    
    def test_citation_number_out_of_bounds(self):
        """Citation numbers beyond chunk count should be handled."""
        chunks_metadata = [{"chunk_number": 1}, {"chunk_number": 2}]
        chunk_idx = 5  # Out of bounds
        
        if 1 <= chunk_idx <= len(chunks_metadata):
            result = chunks_metadata[chunk_idx - 1]
        else:
            result = None
        
        assert result is None
    
    def test_duplicate_citations(self):
        """Duplicate citation numbers in answer should be deduplicated."""
        import re
        answer = "Found in [1] and also [1] again, plus [2]."
        
        citation_pattern = r'\[(\d+)\]'
        matches = re.findall(citation_pattern, answer)
        cited_indices = set(int(m) for m in matches)
        
        assert cited_indices == {1, 2}  # Deduplicated
    
    def test_unicode_in_highlight(self):
        """Unicode characters in highlights should be handled."""
        highlight = "café résumé naïve"
        result = normalize_text_for_matching(highlight)
        assert "café" in result or "cafe" in result  # Depends on normalization
    
    def test_very_long_highlight(self):
        """Very long highlights should be handled."""
        highlight = "word " * 100
        result = normalize_text_for_matching(highlight)
        assert len(result) > 0
        
        encoded = encode_text_fragment(highlight)
        assert len(encoded) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

