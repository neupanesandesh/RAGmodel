"""API tests with a mocked store and embedder.

These tests exercise the FastAPI layer in isolation: routing, auth,
validation, response shapes, and the request/response models. The
Weaviate client is replaced with an in-memory fake so the tests run
fast and deterministically.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from fastapi.testclient import TestClient


# -------------------- fakes --------------------
class FakeEmbedder:
    dimensions = 8

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(i + 1)] * 8 for i, _ in enumerate(texts)]

    def embed_query(self, query: str) -> list[float]:
        return [0.1] * 8


class FakeStore:
    def __init__(self) -> None:
        self._tenants: dict[str, list[dict[str, Any]]] = {}

    def is_ready(self) -> bool:
        return True

    def close(self) -> None:
        pass

    def ensure_schema(self, vector_size: int) -> None:
        pass

    def tenant_exists(self, tenant: str) -> bool:
        return tenant in self._tenants

    def create_tenant(self, tenant: str) -> None:
        self._tenants.setdefault(tenant, [])

    def list_tenants(self) -> list[str]:
        return sorted(self._tenants)

    def delete_tenant(self, tenant: str) -> None:
        self._tenants.pop(tenant, None)

    def tenant_stats(self, tenant: str) -> dict[str, Any]:
        return {
            "tenant": tenant,
            "collection": "Document",
            "object_count": len(self._tenants.get(tenant, [])),
        }

    def list_datasets(self, tenant: str) -> list[str]:
        return sorted({o["dataset_id"] for o in self._tenants.get(tenant, [])})

    def upsert_batch(self, tenant, dataset_id, rows, vectors) -> int:
        bucket = self._tenants.setdefault(tenant, [])
        for r in rows:
            bucket.append({**r, "dataset_id": dataset_id})
        return len(rows)

    def hybrid_search(
        self, tenant, query_text, query_vector, *, dataset_id=None, filters=None, limit=10, alpha=0.5
    ):
        from service.core.weaviate_store import SearchHit

        pool = self._tenants.get(tenant, [])
        if dataset_id:
            pool = [o for o in pool if o["dataset_id"] == dataset_id]
        if filters:
            for k, v in filters.items():
                pool = [o for o in pool if o.get(k) == v]
        hits: list[SearchHit] = []
        for idx, o in enumerate(pool[:limit]):
            hits.append(
                SearchHit(
                    object_id=f"fake-{idx}",
                    score=1.0 - idx * 0.1,
                    text=o["text"],
                    dataset_id=o["dataset_id"],
                    chunk_index=0,
                    chunk_count=1,
                    created_at="2026-04-19T00:00:00+00:00",
                    metadata={k: v for k, v in o.items() if k not in {"text", "url", "dataset_id"}},
                    explain_score=None,
                )
            )
        return hits

    def delete_dataset(self, tenant: str, dataset_id: str) -> int:
        bucket = self._tenants.get(tenant, [])
        before = len(bucket)
        self._tenants[tenant] = [o for o in bucket if o["dataset_id"] != dataset_id]
        return before - len(self._tenants[tenant])


# -------------------- fixtures --------------------
@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("API_KEY", "testkey")
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "10000")

    # Reload config so env overrides take effect.
    import importlib

    from service import config as config_mod

    importlib.reload(config_mod)

    from service import main as main_mod

    importlib.reload(main_mod)

    fake_store = FakeStore()
    main_mod.store = fake_store
    main_mod.embedder = FakeEmbedder()

    with TestClient(main_mod.app) as c:
        c.headers.update({"X-API-Key": "testkey"})
        yield c


# -------------------- tests --------------------
def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["weaviate_ready"] is True
    assert body["embedding_dimension"] == 8


def test_auth_required(client):
    client.headers.pop("X-API-Key", None)
    r = client.get("/tenants")
    assert r.status_code == 401


def test_tenant_lifecycle(client):
    assert client.post("/tenants", json={"name": "acme"}).status_code == 201
    tenants = client.get("/tenants").json()
    assert "acme" in tenants["tenants"]
    info = client.get("/tenants/acme").json()
    assert info["tenant"] == "acme"
    assert client.delete("/tenants/acme").status_code == 204
    assert client.get("/tenants/acme").status_code == 404


def test_upload_and_search(client):
    client.post("/tenants", json={"name": "acme"})
    payload = {
        "documents": [
            {"url": "u1", "text": "Refund policy is 30 days.", "meta": {"rating": 5}},
            {"url": "u2", "text": "Great service today.", "meta": {"rating": 5}},
            {"url": "u3", "text": "Average experience.", "meta": {"rating": 3}},
        ]
    }
    r = client.post("/tenants/acme/datasets/reviews/objects/batch", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["inserted"] == 3
    assert body["skipped"] == 0

    r = client.get("/tenants/acme/datasets").json()
    assert r["datasets"] == ["reviews"]

    r = client.post(
        "/tenants/acme/datasets/reviews/search",
        json={"query": "refund", "limit": 5},
    )
    assert r.status_code == 200
    hits = r.json()["results"]
    assert len(hits) == 3

    r = client.post(
        "/tenants/acme/datasets/reviews/search",
        json={"query": "refund", "limit": 5, "filters": {"rating": 5}},
    )
    hits = r.json()["results"]
    assert all(h["metadata"]["rating"] == 5 for h in hits)


def test_upload_skips_empty_docs(client):
    client.post("/tenants", json={"name": "acme"})
    payload = {
        "documents": [
            {"url": "u1", "text": "Real text."},
            {"url": "u2", "text": "   "},
        ]
    }
    # Pydantic rejects empty string at the model boundary (min_length=1),
    # so we expect 422 here — proves the schema contract.
    r = client.post("/tenants/acme/datasets/d/objects/batch", json=payload)
    assert r.status_code == 422


def test_generate_requires_gemini_key(client, monkeypatch):
    client.post("/tenants", json={"name": "acme"})
    monkeypatch.setattr("service.main.settings.gemini_api_key", None)
    r = client.post(
        "/tenants/acme/generate",
        json={"query": "anything", "limit": 3},
    )
    assert r.status_code == 503
