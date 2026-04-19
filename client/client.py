"""Python client for the Weaviate-backed RAG service.

Minimal, typed, no surprises. Mirrors the HTTP API one-to-one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import requests


@dataclass
class SearchHit:
    object_id: str
    score: float
    text: str
    dataset_id: str
    chunk_index: int
    chunk_count: int
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)
    explain_score: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SearchHit":
        md = dict(d.get("metadata") or {})
        return cls(
            object_id=d["object_id"],
            score=float(d["score"]),
            text=d["text"],
            dataset_id=d["dataset_id"],
            chunk_index=int(d["chunk_index"]),
            chunk_count=int(d["chunk_count"]),
            created_at=d.get("created_at", ""),
            metadata=md,
            explain_score=d.get("explain_score"),
        )


class RAGClient:
    """HTTP client for the Weaviate-backed RAG service.

    Example:
        >>> c = RAGClient("http://localhost:8000", api_key="...")
        >>> c.create_tenant("auditcity")
        >>> c.upload_batch(
        ...     "auditcity",
        ...     "dallas-dentist",
        ...     [{"url": "...", "text": "...", "meta": {"rating": 5}}],
        ... )
        >>> hits = c.search("auditcity", query="refund policy", dataset_id="dallas-dentist")
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()
        if api_key:
            self._session.headers["X-API-Key"] = api_key

    # --- low-level ---------------------------------------------------
    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> dict:
        resp = self._session.request(
            method=method,
            url=f"{self._base}{path}",
            json=json,
            params=params,
            timeout=self._timeout,
        )
        if not resp.ok:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise RuntimeError(f"{method} {path} -> {resp.status_code}: {detail}")
        if resp.status_code == 204 or not resp.content:
            return {}
        return resp.json()

    # --- meta --------------------------------------------------------
    def health(self) -> dict:
        return self._request("GET", "/health")

    # --- tenants -----------------------------------------------------
    def create_tenant(self, tenant: str) -> dict:
        return self._request("POST", "/tenants", json={"name": tenant})

    def list_tenants(self) -> list[str]:
        return self._request("GET", "/tenants")["tenants"]

    def get_tenant(self, tenant: str) -> dict:
        return self._request("GET", f"/tenants/{tenant}")

    def delete_tenant(self, tenant: str) -> None:
        self._request("DELETE", f"/tenants/{tenant}")

    def list_datasets(self, tenant: str) -> list[str]:
        return self._request("GET", f"/tenants/{tenant}/datasets")["datasets"]

    # --- ingest ------------------------------------------------------
    def upload_batch(
        self,
        tenant: str,
        dataset_id: str,
        documents: list[dict[str, Any]],
    ) -> dict:
        return self._request(
            "POST",
            f"/tenants/{tenant}/datasets/{dataset_id}/objects/batch",
            json={"documents": documents},
        )

    def delete_dataset(self, tenant: str, dataset_id: str) -> dict:
        return self._request(
            "DELETE", f"/tenants/{tenant}/datasets/{dataset_id}"
        )

    # --- search ------------------------------------------------------
    def search(
        self,
        tenant: str,
        query: str,
        *,
        dataset_id: Optional[str] = None,
        alpha: float = 0.5,
        limit: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchHit]:
        body: dict[str, Any] = {"query": query, "alpha": alpha, "limit": limit}
        if filters:
            body["filters"] = filters
        path = (
            f"/tenants/{tenant}/datasets/{dataset_id}/search"
            if dataset_id
            else f"/tenants/{tenant}/search"
        )
        resp = self._request("POST", path, json=body)
        return [SearchHit.from_dict(r) for r in resp.get("results", [])]

    # --- generate (RAG) ---------------------------------------------
    def generate(
        self,
        tenant: str,
        query: str,
        *,
        dataset_id: Optional[str] = None,
        alpha: float = 0.5,
        limit: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> dict:
        body: dict[str, Any] = {
            "query": query,
            "alpha": alpha,
            "limit": limit,
        }
        if dataset_id:
            body["dataset_id"] = dataset_id
        if filters:
            body["filters"] = filters
        return self._request("POST", f"/tenants/{tenant}/generate", json=body)
