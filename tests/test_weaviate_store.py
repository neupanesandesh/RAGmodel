"""Integration tests against a live Weaviate.

Enable with::

    docker compose up -d weaviate
    RUN_INTEGRATION=1 pytest tests/test_weaviate_store.py -v

These are the tests that catch v4-client API drift and schema issues
that mocks cannot model (auto-tenant-creation, inverted-index tuning,
aggregate-group-by, etc.).
"""

from __future__ import annotations

import os
import time
import uuid

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def store():
    from service.core.weaviate_store import WeaviateStore

    s = WeaviateStore(
        host=os.getenv("WEAVIATE_HOST", "localhost"),
        http_port=int(os.getenv("WEAVIATE_HTTP_PORT", "8080")),
        grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
        collection=os.getenv("WEAVIATE_COLLECTION", "Document"),
    )
    s.ensure_schema(vector_size=8)
    yield s
    s.close()


@pytest.fixture
def tenant(store):
    name = f"itest-{uuid.uuid4().hex[:8]}"
    store.create_tenant(name)
    yield name
    try:
        store.delete_tenant(name)
    except Exception:
        pass


def _vec(seed: int) -> list[float]:
    # Deterministic, already-unit-length 8-dim vector.
    import math

    base = [math.sin(seed + i) for i in range(8)]
    norm = math.sqrt(sum(x * x for x in base))
    return [x / norm for x in base]


def test_roundtrip_upsert_search(store, tenant):
    rows = [
        {"text": "refund policy is thirty days", "url": "u1", "rating": 5},
        {"text": "product SKU-7741 arrived damaged", "url": "u2", "rating": 2},
        {"text": "fantastic service and friendly staff", "url": "u3", "rating": 5},
    ]
    vectors = [_vec(i) for i in range(len(rows))]

    inserted = store.upsert_batch(tenant, "reviews", rows, vectors)
    assert inserted == 3

    time.sleep(1.0)  # allow async indexing to settle

    hits = store.hybrid_search(
        tenant=tenant,
        query_text="refund",
        query_vector=_vec(0),
        dataset_id="reviews",
        limit=5,
        alpha=0.3,
    )
    assert hits, "expected matches for 'refund'"
    assert any("refund" in h.text.lower() for h in hits)


def test_upsert_is_idempotent(store, tenant):
    row = [{"text": "stable text", "url": "u"}]
    vec = [_vec(42)]
    store.upsert_batch(tenant, "ds", row, vec)
    store.upsert_batch(tenant, "ds", row, vec)  # should overwrite, not duplicate
    time.sleep(0.5)
    stats = store.tenant_stats(tenant)
    assert stats["object_count"] == 1


def test_list_and_delete_dataset(store, tenant):
    store.upsert_batch(tenant, "a", [{"text": "one", "url": "u"}], [_vec(1)])
    store.upsert_batch(tenant, "b", [{"text": "two", "url": "u"}], [_vec(2)])
    time.sleep(0.5)

    datasets = store.list_datasets(tenant)
    assert sorted(datasets) == ["a", "b"]

    deleted = store.delete_dataset(tenant, "a")
    assert deleted == 1
    time.sleep(0.5)
    assert store.list_datasets(tenant) == ["b"]


def test_tenant_drop(store):
    name = f"itest-drop-{uuid.uuid4().hex[:6]}"
    store.create_tenant(name)
    assert store.tenant_exists(name)
    store.delete_tenant(name)
    assert not store.tenant_exists(name)
