# Migration: Qdrant → Weaviate

> This service was originally prototyped on Qdrant. After running the Qdrant
> prototype against realistic tenant counts and query distributions, we hit
> two concrete limits that drove the move to Weaviate. This document records
> the reasoning.

## TL;DR

| Driver | Qdrant reality | Weaviate outcome |
|---|---|---|
| Multi-tenancy at SaaS scale | `dataset_id` / `company_id` payload filter. Works at dozens of tenants; full-scan cost and no isolation at hundreds. No cheap tenant drop. | Native multi-tenancy at the shard level. Lazy-loaded cold tenants. `tenants.remove()` is O(1) per tenant. |
| Hybrid search quality | Pure-dense retrieval missed exact-term queries (SKUs, brand names, numeric filters). Adding hybrid requires a separate sparse-vector pipeline + client-side fusion. | Built-in BM25 + vector hybrid with a single `alpha` parameter. No extra infrastructure. |
| Data-quality at ingest | Schema-less payload. Bad data surfaces weeks later as noisy retrieval. | Typed schema rejects malformed inputs at write time. |

All three were real during the prototype. The multi-tenancy pressure was the
forcing function; hybrid search quality was the second-strongest pull; the
schema win was a bonus.

---

## 1. Multi-tenancy at scale

### The Qdrant pattern we ran

Each company was a Qdrant collection. Within a collection, a `dataset_id`
payload field tagged each point. Search was a payload filter:

```python
FieldCondition(key="dataset_id", match=MatchValue(value="dallas-dentist"))
```

This is the standard pattern and it is correct for small scale. We indexed
`dataset_id` as a `KEYWORD` payload index so filtered search was not a full
scan within a collection.

### Where it broke

1. **No per-tenant isolation.** A hot tenant's vectors and a cold tenant's
   vectors live in the same HNSW graph. There is no way to evict a cold
   tenant from memory; the whole collection is resident.
2. **Dataset deletion is expensive.** `client.delete(points_selector=Filter(...))`
   on a dataset with millions of points rewrites segments.
3. **Listing tenants required a full scroll** (`list_datasets` in the original
   `vectorstore.py`). The payload index on `dataset_id` does not help with
   *distinct-value* queries; it helps with equality filters.
4. **One-collection-per-tenant** (the other common Qdrant pattern) hits the
   opposite wall: HNSW per-collection overhead, per-collection metadata, and
   warm-shard memory pressure as tenant count grows.

### What Weaviate gave us

Native multi-tenancy is a first-class feature, not a workaround:

- Each tenant is its own **shard** with its own HNSW index. Tenants are
  isolated end-to-end.
- **Lazy tenant activation** (`auto_tenant_activation: true`): cold tenants
  are loaded on first query, not on process start. Memory tracks working set.
- **O(1) tenant drop** via `collection.tenants.remove([...])`. No filter
  scan, no segment rewrite.
- **`auto_tenant_creation: true`** means ingest code does not need a
  pre-create step; new companies onboard implicitly.
- Native multi-tenancy is designed for "thousands-of-tenants" SaaS — the
  [documented scale target](https://weaviate.io/developers/weaviate/concepts/data#multi-tenancy).

### Mapping in this repo

| Qdrant concept | Weaviate concept |
|---|---|
| Collection per company (`auditcity`) | Tenant per company |
| `dataset_id` payload field | Indexed property on the `Document` collection |
| Payload index on `dataset_id` | Filterable inverted index on `dataset_id` |
| Filter-by-dataset search | `where` filter within tenant |

One Weaviate collection (`Document`), many tenants, `dataset_id` as a
filterable property. Search-within-dataset is a single tenant query with a
property filter — the filter uses the inverted index, not a scroll.

## 2. Hybrid search quality

### The symptom

While running the Qdrant prototype against logged queries, we found
recall-failures on queries like:

- `"refund policy"` — pure-dense retrieval returned semantically-nearby
  chunks about returns and shipping, but missed the one paragraph that
  literally said "refund policy".
- `"SKU-7741"` — exact product identifier; pure-dense was close to random.
- `"5-star review of Lavon Dental"` — numeric + proper noun; weak.

The common factor: queries that contain rare, exact terms where lexical
match dominates semantic match.

### The Qdrant path forward

Qdrant supports hybrid via sparse vectors. To use it we would have needed:

1. A sparse encoder (BM25 via FastEmbed, or SPLADE).
2. A second vector index per collection for sparse.
3. Client-side fusion logic (RRF, relative-score, or manual weighting).
4. A second embedding step at query time.

Doable, but it meant owning a second embedding pipeline and tuning fusion
weights ourselves.

### What Weaviate gave us

`collection.query.hybrid(...)` takes:

- `query` (for BM25 over the inverted index),
- `vector` (for dense search),
- `alpha` (0.0 = pure BM25, 1.0 = pure vector),
- `fusion_type` (ranked fusion or relative-score).

All of it native, one call, one response, with per-component score
explainability. We went from "build a sparse pipeline" to "set alpha=0.5 and
tune per-query-type."

## 3. Schema-first catches bad data at ingest

Qdrant payloads are untyped dicts. A reviewer could upload `rating: "five"`
(string) next to `rating: 5` (int) and both would store. Filters on `rating
>= 4` would silently skip the string rows. No error, just worse recall.

Weaviate's typed schema rejects the string at write time with a clear error.
The schema is a forcing function for clean data; it eliminated a class of
silent-corruption bug.

This was not a migration *driver*, but it turned out to be the nicest quality
of life win after the port.

---

## What did NOT drive the migration

Being explicit about what we did not move for:

- **Raw speed.** Both databases are fast. We did not see a query-latency
  delta in our benchmarks that would justify a rewrite by itself.
- **Embedding modules.** Weaviate has `text2vec-*` modules; we evaluated
  them but kept a bring-your-own-vector pipeline (`sentence-transformers`
  local, optionally Gemini) for two reasons: (a) control over asymmetric
  query/passage encoding, and (b) portability across deployments that don't
  run the module sidecar. See `docs/ADR-002-byov.md`.
- **Vendor ecosystem.** Both have healthy ecosystems. This was a technical
  decision, not a strategic one.

---

## What we left on the table (future work)

- **`rerank-transformers` module** — would likely improve top-k ordering on
  ambiguous queries. Not wired yet; tracked in `ROADMAP.md`.
- **Cross-references** — currently only one collection (`Document`). If we
  model companies or datasets as first-class objects with cross-refs, we
  unlock graph-style queries.
- **Tenant offloading** — Weaviate 1.26+ can offload cold tenants to S3.
  Useful once tenant count climbs past thousands.

---

## How to read the code

The migration is localized. Only the vector-store layer was rewritten:

- `service/core/weaviate_store.py` — Weaviate v4 client, multi-tenant,
  hybrid search, gRPC batch import, deterministic IDs.
- `service/core/embedder.py` — unchanged interface, new default backend
  (sentence-transformers). Gemini retained as an optional backend.
- `service/main.py` — endpoints renamed to Weaviate idiom (`/tenants/...`
  instead of `/collections/...`). Business logic unchanged.

The FastAPI app, logging, Dockerfile, and tests were preserved.
