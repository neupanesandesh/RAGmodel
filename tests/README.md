# Tests

Focused tests for the parts most likely to regress.

## Running

```bash
pip install -e ".[dev]"
pytest                          # full suite
pytest tests/test_config.py -v  # single file
pytest --cov=service --cov=client
```

`test_vectorstore.py` runs against an in-process Qdrant via `QdrantStore(url=":memory:")` — no external service required. The CI workflow also runs the suite against a real Qdrant 1.12.4 service container (`.github/workflows/ci.yml`).

## Files

| File | Covers |
|---|---|
| `test_config.py` | `Settings` defaults + `validate_settings()` — rejects default API key and wildcard CORS in production. |
| `test_chunking.py` | Text chunking edge cases (empty, small, paragraph split, sliding window). |
| `test_vectorstore.py` | Tenant isolation, dataset scoping, idempotent ingest via UUID5 ids, delete. |
| `test_main_smoke.py` | FastAPI wiring — health, OpenAPI surface, API-key and admin-key gating. |

## What we don't test

- Third-party libraries (fastembed, sentence-transformers, qdrant-client internals).
- Qdrant behaviour that's trivially covered by the client (the wrappers add no logic).
- Anything that would require loading a real embedding model — `test_main_smoke.py` mocks the embedder and vector store instead.
