"""Unit tests for the embedder factory and the SentenceTransformer backend.

The ST backend downloads ~130MB on first run; tests are marked ``slow``
and can be skipped with ``pytest -m 'not slow'``.
"""

from __future__ import annotations

import math

import pytest

from service.core.embedder import build_embedder


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown embedder backend"):
        build_embedder("nope")


def test_factory_rejects_gemini_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        build_embedder("gemini", api_key=None)


@pytest.mark.slow
def test_sentence_transformer_backend_embeds_and_normalizes():
    emb = build_embedder("st", model_name="BAAI/bge-small-en-v1.5")
    assert emb.dimensions == 384

    vec = emb.embed_query("refund policy")
    assert len(vec) == 384
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-3, f"expected unit vector, got norm={norm}"

    docs = emb.embed_documents(["first doc", "second doc"])
    assert len(docs) == 2
    assert all(len(v) == 384 for v in docs)


@pytest.mark.slow
def test_sentence_transformer_query_vs_doc_are_different():
    emb = build_embedder("st", model_name="BAAI/bge-small-en-v1.5")
    text = "refund policy is 30 days"
    q = emb.embed_query(text)
    d = emb.embed_documents([text])[0]
    # BGE uses an instruction prefix for queries only; the vectors should
    # not be literally identical.
    assert q != d
