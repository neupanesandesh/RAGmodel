# Architecture

This doc walks through how the service is wired end-to-end, with pointers
into the code. Read this after the [README](../README.md) to understand
*why* each file exists, not just *what* it does.

---

## Component map

```
┌─────────────────────────────────────────────────────────────┐
│  service/main.py               FastAPI app + lifespan       │
│    ├─ APIKeyHeader             X-API-Key auth               │
│    ├─ SlowAPI                  per-IP rate limit            │
│    ├─ CORSMiddleware           strict allowlist             │
│    ├─ _log_requests            request middleware → logs    │
│    └─ Instrumentator           /metrics (Prometheus)        │
│                                                             │
│  service/core/embedder.py      BYOV pipeline                │
│    ├─ Embedder (Protocol)      dimensions, embed_query,     │
│    │                           embed_documents              │
│    ├─ SentenceTransformerEmbedder  BGE (default)            │
│    └─ GeminiEmbedder           optional                     │
│                                                             │
│  service/core/weaviate_store.py  Weaviate v4 client wrapper │
│    ├─ ensure_schema            Vectorizer.none() + MT +     │
│    │                           HNSW-SQ + BM25 tuning        │
│    ├─ create/delete_tenant     shard lifecycle              │
│    ├─ upsert_batch             deterministic UUID5, gRPC    │
│    ├─ hybrid_search            BM25 + vector (alpha)        │
│    ├─ list_datasets            aggregate.group_by           │
│    └─ delete_dataset           data.delete_many             │
│                                                             │
│  service/logging_config.py     JSON stdout + rotating file  │
│  service/config.py             env → Settings, validated    │
│  service/models.py             pydantic request/response    │
└─────────────────────────────────────────────────────────────┘
```

---

## Request lifecycle

Follow an upload from HTTP to persisted vectors, using `POST
/tenants/{tenant}/datasets/{ds}/objects/batch`:

1. **Auth** — `require_api_key` dependency (`service/main.py`) compares
   `X-API-Key` against `settings.api_key`. In production startup refuses to
   boot without one (`validate_settings` in `service/config.py`).
2. **Rate limit** — `SlowAPIMiddleware` applies a per-IP budget
   (`RATE_LIMIT_PER_MINUTE`).
3. **Validation** — `BatchUploadRequest` (`service/models.py`) enforces
   `min_length=1` on text, caps batch size, etc. Empty strings get a
   422 at the pydantic boundary.
4. **Embedding** — `embedder.embed_documents(texts)` returns unit-norm
   vectors. The BGE backend uses passage encoding here (no query prefix).
5. **Persistence** — `WeaviateStore.upsert_batch` computes a deterministic
   UUID5 per row, uses the Weaviate v4 gRPC batch API, and returns the
   count. Re-running the same payload upserts, never duplicates.
6. **Observability** — the request middleware writes a structured log line
   with `method/path/status/duration_ms`; the perf logger writes a second
   record with `embed_ms / store_ms / total_ms`; Prometheus records an
   HTTP histogram.

Search (`POST /tenants/{tenant}/datasets/{ds}/search`) follows the same
shape: auth → rate limit → validate `HybridSearchRequest` (alpha 0..1,
limit 1..100) → `embedder.embed_query` (with BGE query prefix) →
`WeaviateStore.hybrid_search` → serialize `SearchHit` to `SearchHitOut`.

---

## Schema design

Defined in `WeaviateStore.ensure_schema`:

- **One Collection, `Document`.** Companies are tenants, not collections.
  This avoids the per-collection HNSW overhead that the alternative pattern
  incurs at thousands of tenants.
- **`Configure.Vectorizer.none()`** — bring-your-own-vectors. Rationale in
  [ADR-002](ADR-002-byov.md).
- **HNSW + SQ quantization** — `ef_construction=256`, `max_connections=32`,
  `ef=128`, scalar quantization enabled. Good recall on the 384-dim BGE
  space while cutting memory ~4×.
- **Multi-tenancy** — `Configure.multi_tenancy(enabled=True,
  auto_tenant_creation=True, auto_tenant_activation=True)`. Tenants are
  shards, cold tenants are lazy-loaded on first query.
- **Inverted-index tuning** — `bm25_b=0.75`, `bm25_k1=1.2` (Okapi
  defaults). `text` uses `WORD` tokenization; `url` and `dataset_id` use
  `FIELD` so exact filters hit the index.
- **Properties** — `text`, `url`, `dataset_id`, `chunk_index`,
  `chunk_count`, `created_at`, plus a small fixed set of typed metadata
  (`rating`, `author`, `category`) that the API surfaces as first-class
  filters. Users pass these through `meta` and Weaviate validates types at
  ingest.

## Multi-tenancy mapping

| Concept | Weaviate | Notes |
|---|---|---|
| Company | Tenant | Shard-isolated, lazy-activated |
| Logical dataset | `dataset_id` property | Indexed filterable; not a new tenant |
| Document / chunk | Object | `chunk_index` kept for future re-chunking |

Design decision: **tenant = company, not tenant = dataset.** This keeps
"search across all of a company's datasets" a single tenant query with an
optional filter, while isolating one company's data from another's at the
shard level. See [MIGRATION.md §1](MIGRATION.md#1-multi-tenancy-at-scale).

`create_tenant` uses `tenants.create`; `delete_tenant` uses
`tenants.remove` — both O(1), never a filter scan.

## Deterministic upsert (idempotency)

`WeaviateStore._deterministic_uuid` computes:

```
uuid5(NAMESPACE_DNS, f"{tenant}::{dataset_id}::{chunk_index}::{sha256(text)[:16]}")
```

Consequences:

- Re-uploading the same document writes to the same UUID — Weaviate
  overwrites instead of duplicating.
- Text changes produce a new UUID, so edits create a new object (old one
  persists until explicitly deleted — surface this via re-ingest tooling
  if needed).
- The tenant and dataset_id are part of the key, so the same raw text in
  two different datasets stays distinct.

Tested by `tests/test_weaviate_store.py::test_upsert_is_idempotent`.

## Hybrid search

`WeaviateStore.hybrid_search`:

```python
result = tenant_collection.query.hybrid(
    query=query_text,                  # BM25 input
    vector=query_vector,               # dense input
    alpha=alpha,                       # 0.0 = BM25, 1.0 = vector
    fusion_type=HybridFusion.RELATIVE_SCORE,
    filters=filter_expr,               # optional property filter
    limit=limit,
    return_metadata=MetadataQuery(score=True, explain_score=True),
)
```

- Fusion is `RELATIVE_SCORE` (normalized min-max fusion) rather than RRF;
  in testing it gave smoother ranking when dense and BM25 scores are on
  very different scales.
- `explain_score` is returned to clients so the UI can show *why* a result
  ranked where it did — useful for debugging recall.
- The filter argument is built from a dict of `{property: value}` pairs
  using `Filter.by_property(k).equal(v)` chained with `&`. `dataset_id` is
  added when the route includes it.

## Listing datasets without a scroll

Naively, "what datasets exist for this tenant?" is a `distinct` query over
all objects — O(n) in the object count, and for a hot tenant that's a lot.

`WeaviateStore.list_datasets` uses
`tenant_collection.aggregate.over_all(group_by=GroupByAggregate(prop="dataset_id"))`
instead. This is O(distinct values), resolved from the inverted index.
For a tenant with 10M objects across 20 datasets, it takes milliseconds.

## Dataset deletion

`WeaviateStore.delete_dataset` calls
`tenant_collection.data.delete_many(where=Filter.by_property("dataset_id").equal(...))`.
The server-side filter hits the inverted index, then removes matching
objects in a single round-trip. Compare to the Qdrant pattern, which
re-writes segments.

---

## Embeddings

`service/core/embedder.py` exposes an `Embedder` Protocol (`dimensions`,
`embed_query`, `embed_documents`) with two implementations:

**`SentenceTransformerEmbedder`** — default. Uses
`BAAI/bge-small-en-v1.5` (384-dim, Apache 2.0). BGE is an **asymmetric**
model: `embed_query` prepends `"Represent this sentence for searching
relevant passages: "` to the input, `embed_documents` does not. Getting
this right is worth a few recall points; getting it wrong silently halves
quality.

**`GeminiEmbedder`** — opt-in via `EMBEDDER_BACKEND=gemini`. Batched
embedding with exponential-backoff retry, respects `EMBEDDING_DIMENSION`.
Only loaded if the user sets a key — the import is lazy.

Both return unit-normalized vectors (BGE's default, explicit in the
Gemini path) so cosine similarity in Weaviate is well-behaved.

---

## Logging

`service/logging_config.py` is the single source of truth. Philosophy:
**one structured stream, many contexts.**

- Dev: colored single-line text on stdout.
- Prod: JSON on stdout (`_json_sink`), plus optional rotating JSONL file
  under `LOG_DIR`.
- `get_component_logger(name)`, `get_request_logger()`,
  `get_performance_logger()` all return a `loguru` logger bound with a
  context tag — they do *not* split into separate files. A log aggregator
  filters on tags.

Records always include:

```
timestamp, level, message, logger, function, line
```

…plus whatever was passed in `extra={...}`. The request middleware
promotes `method/path/status/duration_ms`; the perf logger promotes
`tenant/dataset_id/embed_ms/search_ms/etc`.

---

## Metrics

Wired via `prometheus-fastapi-instrumentator` in `service/main.py`.
Exposes `/metrics` at app level (opt-out via `ENABLE_METRICS=false`).

The Grafana dashboard under `ops/grafana/dashboards/rag-service.json`
queries these standard metrics:

- `http_requests_total` — rate + error-rate panels.
- `http_request_duration_seconds_bucket` — p50/p95/p99 via
  `histogram_quantile`.
- `http_requests_inprogress` — in-flight panel.
- Tenant count is queried via the `/tenants` endpoint and shown as a
  secondary panel.

Import the full overlay with:

```bash
docker compose -f docker-compose.yml -f docker-compose.observability.yml up -d
```

---

## Configuration

`service/config.py` loads environment into a typed `Settings` model and
exposes `validate_settings()`. The FastAPI lifespan calls it *before*
constructing the embedder or Weaviate client, so a bad config fails with
a readable error message before any downstream state is touched.

Everything in `.env.example` is validated. Anything not validated is a
bug — add a check.

---

## Testing

Three lanes, chosen to minimize feedback-loop cost:

| Lane | Runs against | Cost | What it catches |
|---|---|---|---|
| Unit (`test_api.py`, `test_config.py`, `test_embedder.py` fast tests) | `FakeStore` + `FakeEmbedder` | seconds | API shape, auth, validation, config errors |
| Integration (`test_weaviate_store.py`) | real Weaviate via compose / service container | ~1 minute | v4-client API drift, schema correctness, MT behavior |
| Smoke (`scripts/smoke.py`) | real service + real Weaviate | ~2 minutes | end-to-end including hybrid ranking |

CI runs all three on every push; `unit` is required, `integration` and
`smoke` gate release. See `.github/workflows/ci.yml`.

---

## Deployment

- **Dev** — `./dev.sh up` mounts the source tree and enables `--reload`.
- **Prod** — `./prod.sh up -d --build` bakes code into the image, sets
  `ENVIRONMENT=production` (which forces `API_KEY` via
  `validate_settings`), and runs with `restart: unless-stopped`.
- **Observability** — layer the overlay on top of either:
  `docker compose -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.observability.yml up -d`.

The Weaviate container mounts a named volume for persistence. HF model
cache is its own named volume so first-boot downloads survive rebuilds.
