"""
Async Qdrant store with native multi-tenancy and hybrid search.

Design notes
------------
- ONE physical Qdrant collection holds all tenants. Tenant isolation is done
  via a `tenant_id` payload index flagged `is_tenant=True` (Qdrant 1.7+),
  which is the vendor-recommended pattern for multi-tenant RAG. Avoids the
  "N HNSW graphs" anti-pattern we'd get with a collection per customer.

- Two named vectors per point:
    * "dense"  — float vectors (cosine) from the embedder
    * "sparse" — BM25 / SPLADE (optional, only populated when hybrid enabled)

- Search uses `query_points` (Qdrant 1.10+). Hybrid search uses server-side
  RRF fusion via `Prefetch` + `FusionQuery` — no Python-side merging.

- `list_datasets` uses the `facet` API (Qdrant 1.12+) instead of scrolling
  the whole collection, so it stays O(unique values) rather than O(points).

- Deterministic UUID5 point IDs make ingest idempotent: re-uploading the
  same (tenant, dataset, chunk_index) overwrites rather than duplicates.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from service.logging_config import get_component_logger

logger = get_component_logger("vectorstore")


_TENANT_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _point_id(tenant_id: str, dataset_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_TENANT_NAMESPACE, f"{tenant_id}::{dataset_id}::{chunk_index}"))


class QdrantStore:
    """Async wrapper around a single multi-tenant Qdrant collection."""

    def __init__(
        self,
        url: str,
        collection: str,
        dense_dim: int,
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        self.collection = collection
        self.dense_dim = dense_dim
        self.client = AsyncQdrantClient(
            url=url,
            api_key=api_key,
            timeout=timeout,
            prefer_grpc=False,
        )
        logger.info(
            "QdrantStore initialised",
            url=url,
            collection=collection,
            dense_dim=dense_dim,
        )

    # --------------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------------
    async def ensure_ready(self, with_sparse: bool) -> None:
        """Create the physical collection and payload indexes if missing."""
        existing = {c.name for c in (await self.client.get_collections()).collections}

        if self.collection not in existing:
            vectors_config = {
                "dense": qm.VectorParams(size=self.dense_dim, distance=qm.Distance.COSINE),
            }
            sparse_config = (
                {"sparse": qm.SparseVectorParams(index=qm.SparseIndexParams(on_disk=False))}
                if with_sparse
                else None
            )
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_config,
            )
            logger.info(
                "physical collection created",
                name=self.collection,
                hybrid=with_sparse,
            )

        await self._ensure_payload_indexes()

    async def _ensure_payload_indexes(self) -> None:
        """Idempotent; Qdrant returns 200 even when an index already exists."""
        # Tenant index — powers multi-tenant isolation. is_tenant=True tells
        # Qdrant to optimise the HNSW graph for per-tenant queries.
        await self.client.create_payload_index(
            collection_name=self.collection,
            field_name="tenant_id",
            field_schema=qm.KeywordIndexParams(
                type=qm.KeywordIndexType.KEYWORD,
                is_tenant=True,
            ),
        )
        await self.client.create_payload_index(
            collection_name=self.collection,
            field_name="dataset_id",
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )
        logger.info("payload indexes ready")

    async def close(self) -> None:
        await self.client.close()

    # --------------------------------------------------------------
    # Tenants (mapped to external "collections")
    # --------------------------------------------------------------
    async def register_tenant(self, tenant_id: str) -> None:
        """No-op on the physical collection; tenants are implicit from ingest.
        Exists so the public `POST /collections/{name}` route keeps working.
        """
        logger.info("tenant registered", tenant_id=tenant_id)

    async def delete_tenant(self, tenant_id: str) -> int:
        """Delete every point belonging to a tenant."""
        if not await self.tenant_exists(tenant_id):
            raise ValueError(f"Tenant '{tenant_id}' does not exist")

        flt = qm.Filter(must=[qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id))])
        await self.client.delete(collection_name=self.collection, points_selector=flt)
        logger.warning("tenant deleted", tenant_id=tenant_id)
        return 1

    async def tenant_exists(self, tenant_id: str) -> bool:
        count = await self.client.count(
            collection_name=self.collection,
            count_filter=qm.Filter(
                must=[qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id))]
            ),
            exact=False,
        )
        return count.count > 0

    async def list_tenants(self, limit: int = 10_000) -> List[str]:
        result = await self.client.facet(
            collection_name=self.collection,
            key="tenant_id",
            limit=limit,
        )
        return sorted([hit.value for hit in result.hits])

    async def tenant_info(self, tenant_id: str) -> Dict[str, Any]:
        if not await self.tenant_exists(tenant_id):
            raise ValueError(f"Tenant '{tenant_id}' does not exist")

        count = await self.client.count(
            collection_name=self.collection,
            count_filter=qm.Filter(
                must=[qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id))]
            ),
            exact=True,
        )
        info = await self.client.get_collection(collection_name=self.collection)
        dense_params = info.config.params.vectors["dense"]
        return {
            "name": tenant_id,
            "vector_count": count.count,
            "vector_size": dense_params.size,
            "distance": dense_params.distance.value if hasattr(dense_params.distance, "value") else str(dense_params.distance),
        }

    # --------------------------------------------------------------
    # Datasets (logical sub-grouping inside a tenant)
    # --------------------------------------------------------------
    async def list_datasets(self, tenant_id: str, limit: int = 10_000) -> List[str]:
        result = await self.client.facet(
            collection_name=self.collection,
            key="dataset_id",
            facet_filter=qm.Filter(
                must=[qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id))]
            ),
            limit=limit,
        )
        return sorted([hit.value for hit in result.hits])

    async def delete_dataset(self, tenant_id: str, dataset_id: str) -> int:
        flt = qm.Filter(
            must=[
                qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id)),
                qm.FieldCondition(key="dataset_id", match=qm.MatchValue(value=dataset_id)),
            ]
        )
        existing = await self.client.count(collection_name=self.collection, count_filter=flt, exact=False)
        if existing.count == 0:
            raise ValueError(f"Dataset '{dataset_id}' does not exist in tenant '{tenant_id}'")

        await self.client.delete(collection_name=self.collection, points_selector=flt)
        logger.warning("dataset deleted", tenant_id=tenant_id, dataset_id=dataset_id)
        return existing.count

    # --------------------------------------------------------------
    # Ingest
    # --------------------------------------------------------------
    async def add_points(
        self,
        tenant_id: str,
        dataset_id: str,
        texts: Sequence[str],
        dense_vectors: Sequence[Sequence[float]],
        metadata_list: Sequence[Dict[str, Any]],
        sparse_vectors: Optional[Sequence[qm.SparseVector]] = None,
    ) -> int:
        if not (len(texts) == len(dense_vectors) == len(metadata_list)):
            raise ValueError("texts, dense_vectors, metadata_list must align in length")
        if sparse_vectors is not None and len(sparse_vectors) != len(texts):
            raise ValueError("sparse_vectors length must match texts length")

        timestamp = datetime.now(timezone.utc).isoformat()
        chunk_count = len(texts)

        points: List[qm.PointStruct] = []
        for i, (text, dense, meta) in enumerate(zip(texts, dense_vectors, metadata_list)):
            vector: Dict[str, Any] = {"dense": list(dense)}
            if sparse_vectors is not None:
                vector["sparse"] = sparse_vectors[i]

            payload = {
                "tenant_id": tenant_id,
                "dataset_id": dataset_id,
                "chunk_index": i,
                "chunk_count": chunk_count,
                "text": text,
                "created_at": timestamp,
                **(meta or {}),
            }

            points.append(
                qm.PointStruct(
                    id=_point_id(tenant_id, dataset_id, i),
                    vector=vector,
                    payload=payload,
                )
            )

        await self.client.upsert(collection_name=self.collection, points=points, wait=True)
        logger.info(
            "points upserted",
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            count=len(points),
            hybrid=sparse_vectors is not None,
        )
        return len(points)

    # --------------------------------------------------------------
    # Search
    # --------------------------------------------------------------
    def _build_filter(
        self,
        tenant_id: str,
        dataset_id: Optional[str],
        extra: Optional[Dict[str, Any]],
    ) -> qm.Filter:
        must: List[qm.FieldCondition] = [
            qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id))
        ]
        if dataset_id:
            must.append(qm.FieldCondition(key="dataset_id", match=qm.MatchValue(value=dataset_id)))
        if extra:
            for k, v in extra.items():
                if v is None or v == "":
                    continue
                must.append(qm.FieldCondition(key=k, match=qm.MatchValue(value=v)))
        return qm.Filter(must=must)

    async def search(
        self,
        tenant_id: str,
        dense_query: Sequence[float],
        dataset_id: Optional[str] = None,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        sparse_query: Optional[qm.SparseVector] = None,
    ) -> List[Dict[str, Any]]:
        # Normalise placeholder values that arrive from the API layer.
        if dataset_id is not None:
            dataset_id = dataset_id.strip()
            if dataset_id in {"", "string", "null"}:
                dataset_id = None

        query_filter = self._build_filter(tenant_id, dataset_id, filters)
        prefetch_limit = max(k * 4, 20)

        if sparse_query is not None:
            response = await self.client.query_points(
                collection_name=self.collection,
                prefetch=[
                    qm.Prefetch(
                        query=list(dense_query),
                        using="dense",
                        limit=prefetch_limit,
                        filter=query_filter,
                    ),
                    qm.Prefetch(
                        query=sparse_query,
                        using="sparse",
                        limit=prefetch_limit,
                        filter=query_filter,
                    ),
                ],
                query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                limit=k,
                with_payload=True,
            )
        else:
            response = await self.client.query_points(
                collection_name=self.collection,
                query=list(dense_query),
                using="dense",
                query_filter=query_filter,
                limit=k,
                with_payload=True,
            )

        return [self._format_hit(p) for p in response.points]

    # --------------------------------------------------------------
    # Recommend / Discover (Qdrant-unique query types)
    # --------------------------------------------------------------
    async def recommend(
        self,
        tenant_id: str,
        positive_vectors: Sequence[Sequence[float]],
        negative_vectors: Sequence[Sequence[float]] = (),
        dataset_id: Optional[str] = None,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not positive_vectors and not negative_vectors:
            raise ValueError("recommend requires at least one positive or negative example")

        query_filter = self._build_filter(tenant_id, dataset_id, filters)
        response = await self.client.query_points(
            collection_name=self.collection,
            query=qm.RecommendQuery(
                recommend=qm.RecommendInput(
                    positive=[list(v) for v in positive_vectors],
                    negative=[list(v) for v in negative_vectors],
                    strategy=qm.RecommendStrategy.AVERAGE_VECTOR,
                )
            ),
            using="dense",
            query_filter=query_filter,
            limit=k,
            with_payload=True,
        )
        return [self._format_hit(p) for p in response.points]

    async def discover(
        self,
        tenant_id: str,
        target_vector: Optional[Sequence[float]],
        context_pairs: Sequence[tuple[Sequence[float], Sequence[float]]],
        dataset_id: Optional[str] = None,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if target_vector is None and not context_pairs:
            raise ValueError("discover requires target or at least one context pair")

        query_filter = self._build_filter(tenant_id, dataset_id, filters)
        context = [
            qm.ContextPair(positive=list(pos), negative=list(neg))
            for pos, neg in context_pairs
        ]
        response = await self.client.query_points(
            collection_name=self.collection,
            query=qm.DiscoverQuery(
                discover=qm.DiscoverInput(
                    target=list(target_vector) if target_vector is not None else None,
                    context=context,
                )
            ),
            using="dense",
            query_filter=query_filter,
            limit=k,
            with_payload=True,
        )
        return [self._format_hit(p) for p in response.points]

    # --------------------------------------------------------------
    # Snapshots (admin; operates on the shared physical collection)
    # --------------------------------------------------------------
    async def create_snapshot(self) -> Dict[str, Any]:
        desc = await self.client.create_snapshot(collection_name=self.collection)
        return {
            "name": desc.name,
            "creation_time": getattr(desc, "creation_time", None),
            "size": getattr(desc, "size", None),
        }

    async def list_snapshots(self) -> List[Dict[str, Any]]:
        descs = await self.client.list_snapshots(collection_name=self.collection)
        return [
            {
                "name": d.name,
                "creation_time": getattr(d, "creation_time", None),
                "size": getattr(d, "size", None),
            }
            for d in descs
        ]

    @staticmethod
    def _format_hit(point: Any) -> Dict[str, Any]:
        payload = point.payload or {}
        core = {
            "dataset_id": payload.get("dataset_id"),
            "chunk_index": payload.get("chunk_index"),
            "chunk_count": payload.get("chunk_count"),
            "created_at": payload.get("created_at"),
        }
        extras = {
            k: v
            for k, v in payload.items()
            if k not in {"text", "chunk_index", "chunk_count", "dataset_id", "created_at", "tenant_id"}
        }
        return {
            "score": float(point.score) if point.score is not None else 0.0,
            "text": payload.get("text", ""),
            "metadata": {**core, **extras},
        }
