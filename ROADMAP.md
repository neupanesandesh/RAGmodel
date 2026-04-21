# Roadmap

Near-term items I'd take on next. Each is a deliberate follow-on to a decision already made in the codebase — not a feature wishlist.

## Retrieval quality

- **Cross-encoder reranker.** Current hybrid returns RRF-fused top-k from Qdrant; adding a small cross-encoder (e.g. `BAAI/bge-reranker-base`) as an optional final pass would trade ~50 ms for a measurable nDCG bump on noisy corpora. Gated by a `rerank=true` query param so latency-sensitive callers opt out.
- **MMR for diversity.** For recommend/discover responses on dense clusters, top-k by similarity tends to return near-duplicates. Maximal marginal relevance post-pass would fix that without touching the index.
- **Multi-dense routing.** The named-vector schema already supports more than one dense vector. Adding a second dense model (e.g. multilingual E5) and routing by language detection would extend the service beyond English without migrating the collection.

## Performance & cost

- **Scalar / binary quantization.** Qdrant supports both. Binary quantization on the dense vector would cut RAM ~32× at a small recall cost — worth running as an A/B once a real corpus is loaded.
- **Async sparse encoder.** `SparseEncoder.encode_documents` is currently synchronous; a thread-pool wrap would let ingest overlap sparse + dense encoding.
- **gRPC client.** The REST client is fine for dev; production should switch `AsyncQdrantClient(prefer_grpc=True)` and benchmark.

## Multi-tenancy & admin

- **Per-tenant snapshots.** Qdrant snapshots are per-collection. Since all tenants share one physical collection, current snapshots are global. A shard-level or filter-based export path would let us hand a single tenant their data without leaking others.
- **Tenant quotas.** Track `points / tenant` and rate-limit ingest once a tenant crosses a threshold. Needed before onboarding a noisy-neighbor tenant.
- **Per-tenant API keys.** Right now `X-API-Key` is service-wide. A mapping of key → allowed tenant would move auth from "anyone with the key can hit any tenant" to proper ACL.

## Operations

- **Structured request IDs.** Propagate a `request_id` header through logs + Prometheus exemplars so a Grafana panel can jump straight to the matching log line.
- **Ingest pipeline as a task queue.** Large batches currently block the request. Moving ingest to a background worker (Celery or a bare `asyncio.Queue` consumer) would turn `POST /documents/batch` into "accepted, here's a job id."
- **Chaos test in CI.** Add a test that kills Qdrant mid-ingest and asserts the service recovers cleanly — guards against the "works on my machine" class of bugs that reconnection logic usually hides.

## Documentation

- **Benchmark note.** A reproducible `make bench` that ingests a small fixture corpus and prints p50/p95 search latency for dense-only vs hybrid, so anyone reading the repo can see the cost of `hybrid=true` on their own hardware.
- **Weaviate comparison.** The `w-dev` branch proves the port works; a side-by-side table of latency + code size would make the ADR's claims checkable instead of asserted.
