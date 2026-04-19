"""End-to-end smoke test.

Runs against a live service (default: http://localhost:8000). Creates a
throwaway tenant, uploads a small fixture, runs hybrid searches that
exercise dense-leaning and keyword-leaning queries, optionally hits the
/generate endpoint, and cleans up after itself.

Usage:
    python scripts/smoke.py
    BASE_URL=http://localhost:8000 API_KEY=... python scripts/smoke.py
    WITH_GENERATE=1 python scripts/smoke.py   # requires GEMINI_API_KEY on the server
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

# Allow running from the repo root without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from client import RAGClient  # noqa: E402

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY") or None
WITH_GENERATE = os.getenv("WITH_GENERATE", "0") == "1"

TENANT = f"smoke-{uuid.uuid4().hex[:8]}"
DATASET = "sample-reviews"

FIXTURE = [
    {
        "url": "https://example.com/r1",
        "text": "The refund policy is 30 days from purchase. Receipt required.",
        "meta": {"rating": 5, "author": "Ada", "category": "policy"},
    },
    {
        "url": "https://example.com/r2",
        "text": "Incredible service. The team went above and beyond to help us.",
        "meta": {"rating": 5, "author": "Ben", "category": "review"},
    },
    {
        "url": "https://example.com/r3",
        "text": "Average experience. Food was fine but wait time was long.",
        "meta": {"rating": 3, "author": "Cal", "category": "review"},
    },
    {
        "url": "https://example.com/r4",
        "text": "Product SKU-7741 arrived damaged. Contacted support for replacement.",
        "meta": {"rating": 2, "author": "Dee", "category": "support"},
    },
    {
        "url": "https://example.com/r5",
        "text": "Best experience ever. Staff were friendly and knowledgeable.",
        "meta": {"rating": 5, "author": "Eli", "category": "review"},
    },
]


def _step(title: str) -> None:
    print(f"\n==> {title}")


def main() -> int:
    client = RAGClient(BASE_URL, api_key=API_KEY, timeout=60.0)

    _step("health")
    health = client.health()
    print(health)
    assert health.get("status") == "ok", f"service not healthy: {health}"
    assert health.get("weaviate_ready"), "Weaviate not ready"

    _step(f"create tenant {TENANT}")
    print(client.create_tenant(TENANT))

    try:
        _step(f"upload {len(FIXTURE)} docs")
        up = client.upload_batch(TENANT, DATASET, FIXTURE)
        print(up)
        assert up["inserted"] == len(FIXTURE), up

        # Weaviate indexes asynchronously; small wait avoids race on the very
        # next query. The service itself does not need this in production.
        time.sleep(1.5)

        _step("list datasets")
        datasets = client.list_datasets(TENANT)
        print(datasets)
        assert DATASET in datasets

        _step("search: semantic-leaning 'great service' (alpha=0.7)")
        hits = client.search(TENANT, "great service", dataset_id=DATASET, alpha=0.7, limit=3)
        for h in hits:
            print(f"  [{h.score:.3f}] {h.text[:80]}")
        assert hits, "expected at least one hit"

        _step("search: keyword-leaning 'SKU-7741' (alpha=0.2)")
        hits = client.search(TENANT, "SKU-7741", dataset_id=DATASET, alpha=0.2, limit=3)
        for h in hits:
            print(f"  [{h.score:.3f}] {h.text[:80]}")
        assert hits and "SKU-7741" in hits[0].text, "keyword-biased hybrid should surface the SKU row"

        _step("search: filtered (rating=5)")
        hits = client.search(
            TENANT, "experience", dataset_id=DATASET, limit=5, filters={"rating": 5}
        )
        for h in hits:
            print(f"  [{h.score:.3f}] rating={h.metadata.get('rating')} {h.text[:60]}")
        assert all(h.metadata.get("rating") == 5 for h in hits), "filter should exclude rating!=5"

        if WITH_GENERATE:
            _step("generate (retrieval + Gemini)")
            resp = client.generate(
                TENANT, "What is the refund policy?", dataset_id=DATASET, limit=3
            )
            print(f"  answer: {resp.get('answer', '')[:300]}")
            assert resp.get("answer"), "generate returned empty answer"

        _step("delete dataset")
        print(client.delete_dataset(TENANT, DATASET))

    finally:
        _step(f"delete tenant {TENANT}")
        try:
            client.delete_tenant(TENANT)
        except Exception as e:
            print(f"  cleanup warning: {e}")

    print("\nOK smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
