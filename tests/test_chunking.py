"""
Tests for text chunking logic.

Critical because chunking affects search quality.
"""

import pytest
from service.core.chunking import chunk_text, estimate_tokens


class TestChunking:
    """Test the chunking logic."""

    def test_empty_text(self):
        """Empty text should return empty list."""
        result = chunk_text("")
        assert result == []

    def test_small_text_single_chunk(self, sample_short_text):
        """Text under 500 tokens should stay as one chunk."""
        chunks = chunk_text(sample_short_text)

        assert len(chunks) == 1
        assert chunks[0][0] == sample_short_text
        assert chunks[0][1] == 0  # chunk_index

    def test_medium_text_multiple_chunks(self, sample_medium_text):
        """Medium text should split into multiple chunks."""
        chunks = chunk_text(sample_medium_text)

        # Should have multiple chunks
        assert len(chunks) > 1

        # Each chunk should have text and index
        for i, (text, index) in enumerate(chunks):
            assert len(text) > 0
            assert index == i

    def test_large_text_sliding_window(self, sample_large_text):
        """Large text should use sliding window."""
        chunks = chunk_text(sample_large_text)

        # Should have many chunks
        assert len(chunks) > 3

        # Chunks should have content
        for text, index in chunks:
            assert len(text) > 0
            assert estimate_tokens(text) < 600  # Should be under limit

    def test_whitespace_only_text(self):
        """Text with only whitespace should return empty."""
        result = chunk_text("   \n\n   \t  ")
        assert result == []

    def test_single_word(self):
        """Single word should return one chunk."""
        chunks = chunk_text("Hello")
        assert len(chunks) == 1
        assert chunks[0][0] == "Hello"


class TestTokenEstimation:
    """Test token estimation."""

    def test_estimate_tokens_basic(self):
        """Should estimate tokens reasonably."""
        text = "Hello world this is a test"
        tokens = estimate_tokens(text)

        # 6 words = ~8 tokens (1 token ≈ 0.75 words)
        assert tokens > 0
        assert tokens < 20  # Sanity check

    def test_estimate_tokens_empty(self):
        """Empty text should have zero tokens."""
        assert estimate_tokens("") == 0
