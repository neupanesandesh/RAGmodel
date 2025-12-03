"""
Test configuration and shared fixtures.
"""

import pytest


@pytest.fixture
def sample_short_text():
    """Short text for testing (< 500 tokens)."""
    return "This is a short piece of text for testing."


@pytest.fixture
def sample_medium_text():
    """Medium text for testing (500-1500 tokens)."""
    paragraph = """
    Machine learning is a subset of artificial intelligence that enables systems to learn
    and improve from experience without being explicitly programmed. It focuses on the
    development of computer programs that can access data and use it to learn for themselves.
    The process of learning begins with observations or data, such as examples, direct
    experience, or instruction, in order to look for patterns in data and make better
    decisions in the future based on the examples that we provide.
    """
    # Repeat to get ~700 tokens
    return (paragraph + "\n\n") * 15


@pytest.fixture
def sample_large_text():
    """Large text for testing (> 1500 tokens)."""
    sentence = "This is a sentence in a very long document. "
    # Repeat to get ~2000 tokens
    return sentence * 400
