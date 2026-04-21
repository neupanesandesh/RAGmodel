# RAGmodel

[![CI](https://github.com/sandesh-neupane/RAGmodel/actions/workflows/ci.yml/badge.svg)](https://github.com/sandesh-neupane/RAGmodel/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](pyproject.toml)

A multi-tenant RAG microservice built on **Qdrant 1.12**, with native hybrid search (dense + BM25, server-side RRF fusion), open-source embeddings (BGE), Prometheus metrics, and admin snapshot endpoints.

The public HTTP surface is deliberately stable (`/collections/{name}/...`) — the external *collection* name maps to a Qdrant payload-indexed *tenant id* inside a single shared collection. One HNSW graph, one cache, strict per-tenant filtering.

A parallel Weaviate port of the same service lives on the [`w-dev`](../../tree/w-dev) branch for comparison.

---

## Quickstart

```bash
git clone <your-fork>
cd RAGmodel
cp .env.example .env          # edit API_KEY and CORS_ORIGINS before production
docker compose up -d          # starts qdrant + ragmodel
curl http://localhost:5000/health
open http://localhost:5000/docs
```

Add Prometheus + Grafana:

```bash
docker compose -f docker-compose.yml -f docker-compose.observability.yml up -d
# Grafana: http://localhost:3000   (admin/admin, dashboard "RAGmodel Overview")
```

Tests (requires a Qdrant instance, or runs in-memory):

```bash
pip install -e ".[dev]"
pytest
```

---

## API surface

All endpoints require `X-API-Key` (set via `API_KEY` env). Admin endpoints require `X-Admin-Key` (set via `ADMIN_API_KEY`; omitted by default = admin routes return 503).

| Method | Path | Purpose |
|---|---|---|
| `GET`    | `/health` | Liveness + embedder status |
| `GET`    | `/metrics` | Prometheus scrape endpoint |
| `POST`   | `/collections` | Register a tenant (no-op; tenants are implicit from first ingest) |
| `GET`    | `/collections` | List tenants |
| `GET`    | `/collections/{name}` | Tenant info (point count) |
| `DELETE` | `/collections/{name}` | Delete all points for a tenant |
| `GET`    | `/collections/{name}/datasets` | List dataset ids within a tenant |
| `POST`   | `/collections/{name}/documents/batch/{dataset_id}` | Ingest a batch (dense + optional sparse) |
| `DELETE` | `/collections/{name}/documents/{dataset_id}` | Delete one dataset |
| `POST`   | `/collections/{name}/search?hybrid=true` | Search the whole tenant |
| `POST`   | `/collections/{name}/{dataset_id}/search?hybrid=true` | Search one dataset |
| `POST`   | `/collections/{name}/recommend` | Recommend by positive/negative examples (Qdrant-unique) |
| `POST`   | `/collections/{name}/discover` | Discover with target + context pairs (Qdrant-unique) |
| `POST`   | `/admin/snapshots` | Snapshot the shared physical collection |
| `GET`    | `/admin/snapshots` | List snapshots |

See `/docs` for schemas.

### Search example

```bash
curl -X POST http://localhost:5000/collections/acme/search?hybrid=true \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"query": "refund policy", "filters": {"doc_type": "policy"}}'
```

`hybrid=true` prefixes the query with a sparse BM25 prefetch and a dense prefetch, fused server-side with RRF — no Python-side merging.

---

## Why Qdrant

See [`docs/adr/001-why-qdrant.md`](docs/adr/001-why-qdrant.md) for the full decision record. Short version: native multi-tenancy via `is_tenant=True`, server-side RRF hybrid, and recommend/discover primitives kept this service at ~500 lines instead of growing a custom fusion + ACL layer.

---

## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the component diagram, data flow, and tenancy model.

---

## Roadmap

See [`ROADMAP.md`](ROADMAP.md).

---

## License

[MIT](LICENSE) © 2026 Sandesh Neupane
