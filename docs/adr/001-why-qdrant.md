# ADR-001: Vector database — Qdrant

- **Status**: Accepted
- **Date**: 2026-04-19
- **Deciders**: Sandesh Neupane

## Context

RAGmodel needs a vector store that handles three things well:

1. **Multi-tenant isolation** — many logical customers over one service, without per-tenant index explosion.
2. **Hybrid retrieval** — dense + sparse (BM25-style) fused in one query path. Pure-dense misses exact terms; pure-lexical misses semantics.
3. **Beyond-kNN primitives** — at least recommend-by-example, ideally discover with context pairs, so the service isn't locked into "embed query, top-k" forever.

Plus the usual: stable async Python client, a real metrics endpoint, snapshot/restore, permissive license, and reasonable operational cost at single-node scale.

## Options considered

**Qdrant 1.12**
- Rust; native multi-tenancy via `is_tenant=True` keyword index (one HNSW graph, tenant-partitioned on disk).
- Named vectors for dense + sparse in the same collection; server-side RRF fusion via `FusionQuery` / `Prefetch`.
- Recommend and Discover APIs built in.
- `AsyncQdrantClient`, `:memory:` mode for tests, Prometheus `/metrics`, snapshot API, Apache-2.0.

**Weaviate 1.27**
- Go; first-class hybrid (`alpha` fusion) and BM25 built in.
- Multi-tenancy as isolated tenants under a class — conceptually clean but heavier per-tenant overhead than a shared-graph model.
- Mature GraphQL + REST; good modules ecosystem.
- Fewer "beyond-kNN" primitives than Qdrant (no direct equivalent of Discover with context pairs).

**pgvector (Postgres)**
- Operationally simple if you already run Postgres.
- No native hybrid fusion — you hand-roll RRF across `tsvector` and `<->`.
- Multi-tenancy is just `WHERE tenant_id = ?` — cheap, but no physical partitioning, so large tenants degrade small-tenant latency.
- ivfflat/hnsw works fine at moderate scale; weaker at very high recall + filter combinations.

**Pinecone**
- Managed; fast path to "it works."
- Namespaces give multi-tenancy.
- Hybrid exists but fusion semantics are less transparent.
- Closed-source, per-pod pricing, no local/dev story beyond a mock — fails the "must run offline in CI" bar.

## Decision

**Qdrant.** Specifically the 1.12 feature set: `is_tenant=True`, `query_points` + `Prefetch` + `FusionQuery(RRF)`, `RecommendQuery`, `DiscoverQuery`, `facet`, `AsyncQdrantClient`.

## Consequences

**Positive**

- ~500 lines of service code instead of a hand-rolled fusion layer. The RRF, the tenant partitioning, and the recommend/discover math are all in the database.
- Per-tenant isolation is a *property of the index*, not just a filter. Large tenants don't silently make small ones slower.
- CI runs offline: Qdrant 1.12.4 as a GitHub Actions service, plus `:memory:` mode for unit tests.
- Observability is free: Qdrant exposes its own Prometheus metrics; the Grafana dashboard scrapes both app and DB in the same overlay.

**Negative / open questions**

- The ecosystem is smaller than Weaviate's (fewer LangChain/LlamaIndex integrations pre-wired for the newer features like Discover). Consequence: bespoke client code for the advanced endpoints, which is the code in `service/core/vectorstore.py`.
- Sparse vector support depends on an external encoder (we use `fastembed`'s BM25). That's a second model to warm up at startup.
- Admin ops (snapshots) act on the *shared physical collection* — they're global across tenants. Mitigation: separate `X-Admin-Key`, default-off (503 when unset). A future "per-tenant snapshot" story is in the roadmap.
- Scaling past single-node means Qdrant Cloud or self-managed clustering; the abstraction in `QdrantStore` is thin enough to swap backends if a tenant outgrows this deployment, but the hybrid + recommend + discover paths would need re-implementation on anything else.

## Parallel Weaviate port

A port of the same service to Weaviate lives on the `w-dev` branch. It was built to make the decision honest — "Qdrant because I haven't tried the alternative" isn't a real ADR. The port confirms: Weaviate handles the hybrid case cleanly; the multi-tenancy and recommend/discover cases are where Qdrant's primitives pulled ahead for this specific service.
