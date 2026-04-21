"""
Vectorstore tests using an in-memory Qdrant (no external server required).

Covers tenant isolation — the single most important invariant of the
multi-tenant design. If tenant A can ever see tenant B's points, the whole
security model breaks. These tests prove the is_tenant filter holds.
"""

import random
from typing import List

import pytest

from service.core.vectorstore import QdrantStore


DIM = 16


def _vec(seed: int) -> List[float]:
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(DIM)]


@pytest.fixture
async def store():
    # ":memory:" spins up a process-local Qdrant that speaks the same API.
    s = QdrantStore(
        url=":memory:",
        collection="test_collection",
        dense_dim=DIM,
    )
    await s.ensure_ready(with_sparse=False)
    yield s
    await s.close()


class TestTenantIsolation:
    async def test_tenant_a_cannot_see_tenant_b(self, store):
        await store.add_points(
            tenant_id="acme",
            dataset_id="ds1",
            texts=["acme doc"],
            dense_vectors=[_vec(1)],
            metadata_list=[{"url": "https://acme.test/1"}],
        )
        await store.add_points(
            tenant_id="umbrella",
            dataset_id="ds1",
            texts=["umbrella doc"],
            dense_vectors=[_vec(2)],
            metadata_list=[{"url": "https://umbrella.test/1"}],
        )

        acme_hits = await store.search(tenant_id="acme", dense_query=_vec(1), k=10)
        assert len(acme_hits) == 1
        assert "acme" in acme_hits[0]["text"]

        umbrella_hits = await store.search(tenant_id="umbrella", dense_query=_vec(2), k=10)
        assert len(umbrella_hits) == 1
        assert "umbrella" in umbrella_hits[0]["text"]

    async def test_list_tenants_returns_all(self, store):
        await store.add_points("a", "d", ["x"], [_vec(1)], [{}])
        await store.add_points("b", "d", ["y"], [_vec(2)], [{}])
        tenants = await store.list_tenants()
        assert set(tenants) == {"a", "b"}


class TestDatasetScoping:
    async def test_dataset_filter_limits_results(self, store):
        await store.add_points("acme", "reviews", ["r1"], [_vec(1)], [{}])
        await store.add_points("acme", "products", ["p1"], [_vec(2)], [{}])

        reviews = await store.search(
            tenant_id="acme", dense_query=_vec(1), dataset_id="reviews", k=10
        )
        assert all(r["metadata"]["dataset_id"] == "reviews" for r in reviews)

    async def test_list_datasets_is_tenant_scoped(self, store):
        await store.add_points("acme", "reviews", ["r"], [_vec(1)], [{}])
        await store.add_points("acme", "products", ["p"], [_vec(2)], [{}])
        await store.add_points("umbrella", "reviews", ["r"], [_vec(3)], [{}])

        acme_ds = await store.list_datasets("acme")
        assert set(acme_ds) == {"reviews", "products"}

        umbrella_ds = await store.list_datasets("umbrella")
        assert umbrella_ds == ["reviews"]


class TestIdempotency:
    async def test_re_ingest_overwrites(self, store):
        await store.add_points("acme", "ds", ["original"], [_vec(1)], [{"v": 1}])
        await store.add_points("acme", "ds", ["updated"], [_vec(1)], [{"v": 2}])

        hits = await store.search(tenant_id="acme", dense_query=_vec(1), k=10)
        assert len(hits) == 1
        assert hits[0]["text"] == "updated"
        assert hits[0]["metadata"]["v"] == 2


class TestDelete:
    async def test_delete_dataset(self, store):
        await store.add_points("acme", "reviews", ["r"], [_vec(1)], [{}])
        await store.add_points("acme", "products", ["p"], [_vec(2)], [{}])

        await store.delete_dataset("acme", "reviews")
        datasets = await store.list_datasets("acme")
        assert datasets == ["products"]

    async def test_delete_tenant_nonexistent_raises(self, store):
        with pytest.raises(ValueError):
            await store.delete_tenant("does-not-exist")
