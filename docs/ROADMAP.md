# Roadmap

Tracked, in rough priority order. Items marked (stretch) are not planned
for the initial release.

## Near term

- [ ] **Rerank step** via `rerank-transformers` module for top-k reordering.
      Particularly useful for hybrid queries where BM25 and vector rankings
      disagree strongly.
- [ ] **Tenant-aware rate limiting** — currently slowapi limits per client
      IP; extend to per-tenant quotas.
- [ ] **Streaming batch upload** endpoint for datasets that do not fit in
      a single POST body.
- [ ] **Aggregations endpoint** — surface Weaviate's `Aggregate` API
      (counts, numeric stats) without forcing clients to write GraphQL.

## Mid term

- [ ] **Tenant offloading** to S3-compatible cold storage
      (Weaviate 1.26+). Useful once tenant count climbs past thousands.
- [ ] **Weaviate Cluster mode** — the current compose runs single-node.
      Add a k8s chart and document replication factor choice.
- [ ] **Per-property BM25 tuning** — move `bm25_b` / `bm25_k1` into a
      config surface so tenants can tune recall per field.

## Stretch

- [ ] **Cross-references** between a `Company` class and `Document`
      objects, enabling graph-style queries ("find reviews written about
      company X that also mention competitor Y").
- [ ] **Multi-vector per object** — store separate embeddings for
      `title` and `body` and let the query choose which to hit.
- [ ] **PII redaction pass** at ingest time, configurable per tenant.

## Explicitly out of scope

- Replacing the embedder with a proprietary-only path. The default must
  remain open source and runnable offline.
- Training custom embedders. We use off-the-shelf models and let the
  hybrid fusion absorb model imperfections.
