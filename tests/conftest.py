"""Shared test fixtures.

Integration tests are opt-in via RUN_INTEGRATION=1 so CI can decide
whether to spin up a Weaviate container. Unit tests always run.
"""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless RUN_INTEGRATION=1."""
    if os.getenv("RUN_INTEGRATION") == "1":
        return
    skip = pytest.mark.skip(reason="set RUN_INTEGRATION=1 to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def sample_docs() -> list[dict]:
    return [
        {
            "url": "https://example.com/a",
            "text": "The refund policy allows returns within 30 days.",
            "meta": {"rating": 5, "author": "Ada"},
        },
        {
            "url": "https://example.com/b",
            "text": "Product SKU-7741 shipped with damaged packaging.",
            "meta": {"rating": 2, "author": "Ben"},
        },
        {
            "url": "https://example.com/c",
            "text": "Fantastic customer service team resolved the issue quickly.",
            "meta": {"rating": 5, "author": "Cal"},
        },
    ]
