# ADR-002: Bring-your-own-vectors (BYOV), not Weaviate modules

## Context

Weaviate offers two ways to attach vectors to objects:

1. **Vectorizer modules** — e.g. `text2vec-openai`, `text2vec-cohere`,
   `text2vec-transformers`. Weaviate runs or calls the embedder; the
   client sends raw text.
2. **Bring-your-own-vectors (BYOV)** — the application computes vectors
   and sends them alongside each object. The collection is configured
   with `vectorizer_config=Vectorizer.none()`.

We chose BYOV.

## Decision

Embeddings are computed in the application process via a pluggable
`Embedder` interface (`service/core/embedder.py`), with two concrete
backends:

- **`SentenceTransformerEmbedder`** (default) — `BAAI/bge-small-en-v1.5`.
  Open source, Apache 2.0, 384-dim. Asymmetric query encoding via BGE's
  recommended prefix.
- **`GeminiEmbedder`** (optional) — `gemini-embedding-001`.

Vectors are attached to each object at batch-insert time.

## Reasons

1. **Portability.** BYOV runs the same way on self-hosted Weaviate,
   Weaviate Cloud, and an embedded test container. Module support varies
   per deployment target; e.g. `text2vec-google` requires an API-key
   module wired into the Weaviate binary, which is not the default in
   the OSS image.
2. **Asymmetric query encoding stays in our hands.** BGE's query
   instruction prefix and Gemini's `RETRIEVAL_QUERY` vs
   `RETRIEVAL_DOCUMENT` task types are retrieval-quality knobs. Owning
   the embedder means owning those knobs.
3. **Model swaps are a one-file change.** Changing the embedding model
   in a module-based setup means reindexing against a different Weaviate
   container configuration. With BYOV, we change the embedder class and
   reindex with the new dimensions.
4. **Offline reproducibility.** The default backend works without any
   external API call, which keeps CI and the showcase deterministic.

## Tradeoffs we accept

- **The Docker image is larger** (PyTorch + sentence-transformers).
  Mitigations: multi-arch `hf_cache` volume so model weights survive
  rebuilds, and a documented path to swap the backend to `gemini` (tiny
  image) if that matters more than offline.
- **No server-side re-vectorize-on-update.** If the underlying text
  property changes, the application must recompute the vector. With a
  module, Weaviate would do this. In practice our inputs are immutable
  (review/product records), so this is not exercised.

## When we would revisit

- If we add streaming inputs where the application cannot afford to
  block on an in-process encode, we would move to the
  `text2vec-transformers` sidecar module and let Weaviate do the work.
- If we want `rerank-transformers` at query time, that is a module
  concern regardless of how the initial vectorization is done; it is
  orthogonal to this decision.
