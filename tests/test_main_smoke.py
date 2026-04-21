"""
Smoke tests for the FastAPI app that don't need a running Qdrant.

Exercises route wiring, OpenAPI schema, and auth gating.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

import service.main as main_module


@pytest.fixture
def client(monkeypatch):
    # Replace the module-level globals with mocks so the handlers don't need
    # a real Qdrant / real embedder. Route wiring is what we're testing.
    fake_store = MagicMock()
    fake_store.list_tenants = AsyncMock(return_value=["acme"])
    fake_store.tenant_info = AsyncMock(
        return_value={"name": "acme", "vector_count": 0, "vector_size": 384, "distance": "Cosine"}
    )
    fake_store.register_tenant = AsyncMock(return_value=None)
    fake_store.close = AsyncMock(return_value=None)

    fake_embedder = MagicMock()
    fake_embedder.model_name = "fake"
    fake_embedder.dimensions = 384

    monkeypatch.setattr(main_module, "vector_store", fake_store)
    monkeypatch.setattr(main_module, "embedder", fake_embedder)
    monkeypatch.setattr(main_module.settings, "api_key", "")

    with TestClient(main_module.app) as c:
        yield c


class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy"
        assert body["embedder_ready"] is True
        assert body["embedder_dimensions"] == 384


class TestDocs:
    def test_openapi_available(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        spec = r.json()
        # Sanity-check a couple of routes we added.
        assert "/collections/{collection_name}/recommend" in spec["paths"]
        assert "/collections/{collection_name}/discover" in spec["paths"]
        assert "/admin/snapshots" in spec["paths"]


class TestAuth:
    def test_admin_endpoint_disabled_without_admin_key(self, client, monkeypatch):
        monkeypatch.setattr(main_module.settings, "admin_api_key", "")
        r = client.get("/admin/snapshots", headers={"X-Admin-Key": "whatever"})
        assert r.status_code == 503

    def test_admin_endpoint_rejects_bad_key(self, client, monkeypatch):
        monkeypatch.setattr(main_module.settings, "admin_api_key", "real-admin")
        main_module.vector_store.list_snapshots = AsyncMock(return_value=[])
        r = client.get("/admin/snapshots", headers={"X-Admin-Key": "wrong"})
        assert r.status_code == 401

    def test_admin_endpoint_accepts_correct_key(self, client, monkeypatch):
        monkeypatch.setattr(main_module.settings, "admin_api_key", "real-admin")
        main_module.vector_store.list_snapshots = AsyncMock(return_value=[])
        r = client.get("/admin/snapshots", headers={"X-Admin-Key": "real-admin"})
        assert r.status_code == 200
        assert r.json() == {"snapshots": []}
