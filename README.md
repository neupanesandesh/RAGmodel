# RAG Service on Weaviate

A multi-tenant, hybrid-search RAG microservice backed by **Weaviate**. One
Python package, one `docker compose up`, and you have a production-shaped
service with native multi-tenancy, BM25+vector hybrid retrieval, structured
observability, and a typed schema.

Originally prototyped on Qdrant; ported to Weaviate to solve real multi-tenant
and hybrid-search limits hit in the prototype. The reasoning is documented in
[`docs/MIGRATION.md`](docs/MIGRATION.md).

---

## Highlights

- **Native multi-tenancy** — tenant-per-company, shard-isolated, lazy-activated.
  Onboard and drop tenants in O(1) with Weaviate's `tenants.create` /
  `tenants.remove` APIs.
- **Hybrid search by default** — BM25 + dense in a single call with a tunable
  `alpha`. Per-hit score explanation via `MetadataQuery(explain_score=True)`.
- **Bring-your-own-vectors** — `Vectorizer.none()` schema, open-source
  `sentence-transformers` (BGE-small, 384-dim, Apache 2.0) as the default
  embedder. Gemini is optional, never required.
- **Typed schema** — `text`, `url`, `dataset_id`, chunk metadata, and
  user-extensible properties with real types. Bad data fails at ingest, not
  at query time.
- **Deterministic upsert** — UUID5 over `(tenant, dataset_id, chunk_index,
  sha256-of-text)` makes re-uploads idempotent.
- **Production-shaped ops** — API-key auth, per-IP rate limiting, structured
  JSON logging, Prometheus `/metrics`, ready-to-import Grafana dashboard.
- **Open-source end-to-end** — every dependency is OSS; no account required
  to run the stack locally.

---

## Architecture

```
              ┌──────────────────────────────────────┐
              │          FastAPI (service/)          │
              │   X-API-Key • CORS • rate-limit      │
              │   /metrics (Prometheus)              │
              └──────────────┬───────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
  ┌───────────┐       ┌─────────────┐      ┌────────────────┐
  │ Embedder  │       │ Weaviate v4 │      │ Optional: Gemini│
  │  ST/BGE   │       │  (gRPC)     │      │   /generate     │
  │  Gemini*  │       │             │      │                 │
  └───────────┘       └──────┬──────┘      └────────────────┘
                             │
                 ┌───────────┴──────────┐
                 │  Collection: Document│
                 │  multi-tenant (shard)│
                 │  hybrid (BM25+vec)   │
                 │  HNSW + SQ quantizer │
                 └──────────────────────┘
```

- **One** Weaviate collection (`Document`). Companies are **tenants**.
  `dataset_id` is an indexed property, so "search within dataset" is a filter
  on the inverted index, not a scroll.
- Embeddings are computed in the service layer and written with the row
  (`Vectorizer.none()`), so the vector pipeline is portable and testable
  without Weaviate modules.

See [`docs/ADR-002-byov.md`](docs/ADR-002-byov.md) for the BYOV rationale.

---

## Quick start

```bash
git clone <this-repo>
cd RAGmodel
cp .env.example .env
# (optional) edit .env — the defaults work for local development

./dev.sh up --build
# or: docker compose up --build
```

Then:

```bash
curl http://localhost:8000/health
open  http://localhost:8000/docs   # interactive Swagger UI
```

First boot downloads the BGE embedding model (~130 MB) into a named
volume — subsequent starts are instant.

### One-shot smoke test

```bash
pip install requests
BASE_URL=http://localhost:8000 API_KEY=$(grep ^API_KEY .env | cut -d= -f2) \
  python scripts/smoke.py
```

End-to-end: creates a throwaway tenant → uploads 5 docs → runs semantic-leaning
and keyword-leaning hybrid searches → asserts a keyword-biased query surfaces a
known SKU → cleans up.

---

## API

All `/tenants/**` endpoints require `X-API-Key: <API_KEY>`. `/health`, `/metrics`
and `/docs` are public.

| Method | Path | Purpose |
|---|---|---|
| `GET`    | `/health` | Liveness + Weaviate readiness + embedder dims |
| `GET`    | `/metrics` | Prometheus exposition |
| `POST`   | `/tenants` | Create tenant (company) |
| `GET`    | `/tenants` | List tenants |
| `GET`    | `/tenants/{tenant}` | Tenant info + object count |
| `DELETE` | `/tenants/{tenant}` | Drop tenant (O(1) shard removal) |
| `GET`    | `/tenants/{tenant}/datasets` | List dataset IDs in a tenant |
| `POST`   | `/tenants/{tenant}/datasets/{ds}/objects/batch` | Upload batch (BYOV) |
| `DELETE` | `/tenants/{tenant}/datasets/{ds}` | Delete a dataset |
| `POST`   | `/tenants/{tenant}/search` | Hybrid search across the tenant |
| `POST`   | `/tenants/{tenant}/datasets/{ds}/search` | Hybrid search within a dataset |
| `POST`   | `/tenants/{tenant}/generate` | Retrieve + Gemini-generate (needs `GEMINI_API_KEY`) |

### Search body

```json
{
  "query": "refund policy",
  "limit": 5,
  "alpha": 0.5,
  "filters": { "rating": 5 }
}
```

- `alpha = 0.0` → pure BM25 (keyword).
- `alpha = 1.0` → pure vector (semantic).
- `alpha = 0.5` → balanced hybrid (default).

### Python client

```python
from client import RAGClient

c = RAGClient("http://localhost:8000", api_key="...")
c.create_tenant("acme")
c.upload_batch("acme", "reviews", [
    {"url": "u1", "text": "Refund policy is 30 days.", "meta": {"rating": 5}},
    {"url": "u2", "text": "Product SKU-7741 arrived damaged.", "meta": {"rating": 2}},
])

hits = c.search("acme", "SKU-7741", dataset_id="reviews", alpha=0.2, limit=3)
for h in hits:
    print(f"[{h.score:.3f}] {h.text}")
```

---

## Observability

### Prometheus + Grafana (optional overlay)

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.observability.yml \
  up -d
```

- Prometheus scrapes `rag-service:8000/metrics` every 15s.
- Grafana is pre-provisioned with a 7-panel dashboard at
  `http://localhost:3000` (admin / admin on first login).
- Panels: request rate, 5xx rate, in-flight, p50/p95/p99 latency,
  per-endpoint rate, p95 by endpoint, status-code mix, tenant count.

### Structured logging

Single JSON stream on stdout in production; colorized single-line format in
development. A rotating JSONL file sink is optional (`LOG_DIR=./logs`).

```json
{"timestamp":"2026-04-19T09:21:03.412Z","level":"INFO","logger":"requests",
 "message":"POST /tenants/acme/search 200","method":"POST","path":"/tenants/acme/search",
 "status":200,"duration_ms":41.22}
```

---

## Testing

```bash
pip install -e ".[dev]"

# Fast lane: unit tests only, no Weaviate or model download.
pytest -m "not slow and not integration"

# Integration: needs a running Weaviate (compose or service container).
docker compose up -d weaviate
RUN_INTEGRATION=1 pytest tests/test_weaviate_store.py

# End-to-end smoke against a running stack.
python scripts/smoke.py
```

CI runs all three lanes on every push. See
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## Configuration

All settings come from the environment (see [`.env.example`](.env.example)):

| Variable | Default | Notes |
|---|---|---|
| `WEAVIATE_HOST` | `weaviate` | Service name inside docker-compose |
| `WEAVIATE_HTTP_PORT` / `WEAVIATE_GRPC_PORT` | `8080` / `50051` | |
| `WEAVIATE_COLLECTION` | `Document` | Single Weaviate collection, many tenants |
| `EMBEDDER_BACKEND` | `st` | `st` = local BGE; `gemini` = Gemini API |
| `ST_MODEL` | `BAAI/bge-small-en-v1.5` | 384-dim, Apache 2.0 |
| `EMBEDDING_DIMENSION` | `384` | Must match the model |
| `API_KEY` | *(unset)* | Required when `ENVIRONMENT=production` |
| `CORS_ALLOWED_ORIGINS` | *(unset)* | Comma-separated; empty = no cross-origin |
| `RATE_LIMIT_PER_MINUTE` | `120` | Per client IP |
| `GEMINI_API_KEY` | *(unset)* | Only needed for `/generate` or the gemini embedder |

`service/config.py` validates required settings at startup and fails loud.

---

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — Component map, request
  lifecycle, schema design, logging, metrics, testing lanes.
- [`docs/ADR-001-why-weaviate.md`](docs/ADR-001-why-weaviate.md) — Formal
  record of the Weaviate decision (context, alternatives, consequences).
- [`docs/MIGRATION.md`](docs/MIGRATION.md) — The detailed Qdrant → Weaviate
  narrative (multi-tenancy, hybrid quality, schema-first).
- [`docs/ADR-002-byov.md`](docs/ADR-002-byov.md) — Why BYOV instead of
  `text2vec-*` modules.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — What's next and what's explicitly
  out of scope.

---

## License

MIT — see [`LICENSE`](LICENSE).
