# ADR-001: Use Weaviate as the vector store

- **Status:** Accepted
- **Date:** 2026-04-15
- **Supersedes:** The original Qdrant-based prototype
- **Companion doc:** [`MIGRATION.md`](MIGRATION.md) (the detailed narrative)

---

## Context

This service provides RAG retrieval for a multi-company SaaS. The original
prototype used Qdrant with a "collection-per-company,
`dataset_id`-payload-filter" pattern. Two limits surfaced while the
prototype was running against realistic workloads:

1. **Per-tenant isolation and lifecycle.** Onboarding a new company,
   listing datasets for a company, and dropping a company all got more
   expensive as the tenant count grew. The `list_datasets` operation in
   the old `service/core/vectorstore.py` had to scroll the collection —
   the payload index on `dataset_id` is built for equality filters, not
   distinct-value queries. Dropping a company on the one-collection-per-
   company pattern worked, but hit HNSW-per-collection overhead at
   thousands of tenants; the payload-filter-per-tenant pattern lost
   isolation entirely.
2. **Exact-term queries.** Pure-dense retrieval missed queries that
   contained rare lexical tokens — product SKUs, brand names, numeric
   identifiers. Adding hybrid via Qdrant's sparse-vector support required
   building and maintaining a second (sparse) embedding pipeline plus
   client-side fusion.

A third, smaller concern: Qdrant payloads are untyped, so `rating: "5"`
and `rating: 5` silently coexisted and broke numeric filters.

## Decision

Port the vector store to **Weaviate** using:

- **One collection** (`Document`) with **native multi-tenancy**, mapping
  tenant-per-company. Auto tenant creation and activation.
- **Hybrid search** via `collection.query.hybrid(query, vector, alpha,
  fusion_type=RELATIVE_SCORE)` — single API, no sparse pipeline to own.
- **Typed schema** with indexed filterable properties (`dataset_id`,
  `rating`, `author`, `category`) and `Vectorizer.none()` so the service
  owns the embedding pipeline.
- **Weaviate Python v4 client** (gRPC-first) for batch ingest and query.

Vector-store scope only. FastAPI layer, embedder abstraction, logging,
metrics, and Dockerfile are preserved.

## Alternatives considered

| Option | Why not |
|---|---|
| Stay on Qdrant, add sparse-vector hybrid | Doable, but we'd own a second embedding pipeline and fusion tuning. Doesn't fix multi-tenancy — sparse vectors still live in the same collection. |
| Stay on Qdrant, switch to collection-per-tenant | Per-collection HNSW overhead becomes the bottleneck at thousands of tenants. No lazy activation. |
| Move to Pinecone / managed service | Violates the "open-source, self-hostable" constraint for this project. |
| Move to pgvector | Hybrid is possible (pg_trgm or tsvector) but tuning ranked fusion ourselves is the same problem we wanted to avoid. No first-class multi-tenancy. |
| Weaviate with a `text2vec-*` module | Ties the embedding pipeline to the DB process, complicates testing and portability. See [ADR-002](ADR-002-byov.md). |

## Consequences

**Positive**

- O(1) tenant lifecycle operations (`tenants.create`, `tenants.remove`,
  `tenants.update` for activation state).
- `list_datasets` resolved from the inverted index via
  `aggregate.over_all(group_by="dataset_id")` — milliseconds on a 10M-
  object tenant, versus a scroll.
- Single-call hybrid search. `alpha` is an API-level knob clients can tune
  per query-type.
- Typed schema rejects malformed records at write time. Filters stop
  silently skipping string-where-int rows.
- Weaviate's built-in `/v1/.well-known/ready` gives a first-class
  readiness probe for compose / Kubernetes.

**Negative**

- One more vendor to learn at the ops layer (backup, compaction, tuning).
- The v4 client is gRPC-first; debugging is harder than cURL'ing a REST
  endpoint. Partially mitigated by Weaviate also exposing REST.
- SQ quantization is lossy; we accepted a small recall hit for ~4× memory
  savings on the 384-dim BGE space. Easy to disable if that trade-off
  regresses.
- BYOV plus Weaviate is two pieces to own. We keep the asymmetric-BGE
  query prefix in code, not in a Weaviate module; losing it silently
  halves recall. Tested in `tests/test_embedder.py::test_sentence_transformer_query_vs_doc_are_different`.

**Neutral**

- API surface renamed from `/collections/{name}/...` to `/tenants/{name}/...`.
  Clients will need a version bump (bundled: this is the 2.0.0 release).

## Non-drivers

Explicitly *not* drivers of this decision, to keep the record honest:

- **Raw latency.** Both databases are fast. No single-query latency delta
  that would justify a rewrite by itself.
- **Ecosystem / hype.** Both have healthy communities. This was a
  technical decision for this workload, not a platform bet.
- **Cost.** Both self-hosted. No meaningful cost delta.

## Validation

- Unit tests exercise the new `WeaviateStore` via `FakeStore` at the API
  boundary and prove the contract (`tests/test_api.py`).
- Integration tests exercise a live Weaviate for the subset of behavior
  that mocks cannot model — auto-tenant-creation, aggregate group-by,
  idempotent upsert (`tests/test_weaviate_store.py`).
- The smoke script runs the full stack end-to-end and asserts that a
  keyword-biased hybrid query (`alpha=0.2`) surfaces a known SKU that
  pure-dense would miss (`scripts/smoke.py`).

## References

- [`MIGRATION.md`](MIGRATION.md) — full narrative with code pointers.
- [`ADR-002-byov.md`](ADR-002-byov.md) — why BYOV over `text2vec-*`.
- [Weaviate multi-tenancy docs](https://weaviate.io/developers/weaviate/manage-data/multi-tenancy).
- [Weaviate hybrid search docs](https://weaviate.io/developers/weaviate/search/hybrid).
