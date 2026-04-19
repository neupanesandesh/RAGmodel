"""Weaviate vector store.

Responsibilities:
  - Connect to Weaviate over gRPC+HTTP and manage the client lifecycle.
  - Ensure the `Document` collection schema exists with the expected
    properties, HNSW configuration, inverted-index tuning, and native
    multi-tenancy enabled.
  - Provide CRUD + hybrid-search operations scoped to a tenant, with
    deterministic idempotent writes.

Design notes (kept here because they affect reading the code):
  - Tenant = company. `dataset_id` is a filterable property on objects.
    A single `Document` collection holds every tenant's data, isolated at
    the shard level by Weaviate's native multi-tenancy.
  - Auto tenant creation + activation is on. Clients never have to
    pre-create a tenant before writing.
  - Object UUIDs are deterministic (`uuid5` over tenant + dataset_id +
    chunk_index + text-hash). Re-uploading the same logical row upserts
    instead of duplicating.
  - BYOV (bring-your-own-vectors). The collection is configured with
    `Vectorizer.none()`; the embedder runs in-process and vectors are
    attached to each object at batch-insert time.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import weaviate
from weaviate.classes.config import (
    Configure,
    DataType,
    Property,
    Tokenization,
    VectorDistances,
)
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter, HybridFusion, MetadataQuery
from weaviate.classes.tenants import Tenant
from weaviate.exceptions import UnexpectedStatusCodeError
from weaviate.util import generate_uuid5

from service.logging_config import get_component_logger

logger = get_component_logger("weaviate_store")

# Deterministic UUID namespace so the same logical row always maps to the
# same object UUID across runs, processes, and deployments.
_UUID_NAMESPACE = uuid.UUID("6ba7b812-9dad-11d1-80b4-00c04fd430c8")

# Core property names. Anything else the caller passes in `metadata` is
# allowed through Weaviate's auto-schema (AUTOSCHEMA_ENABLED=true on the
# server) and will be added as a typed property on first occurrence.
_CORE_PROPS = {
    "text",
    "url",
    "dataset_id",
    "chunk_index",
    "chunk_count",
    "created_at",
}


@dataclass(frozen=True)
class SearchHit:
    """Normalized search result surfaced to the API layer."""

    object_id: str
    score: float
    text: str
    dataset_id: str
    chunk_index: int
    chunk_count: int
    created_at: str
    metadata: dict[str, Any]
    explain_score: str | None = None


class WeaviateStore:
    """Thin, opinionated wrapper over the Weaviate v4 client.

    One instance owns one long-lived connection. Call `close()` at shutdown.
    """

    def __init__(
        self,
        host: str,
        http_port: int,
        grpc_port: int,
        collection: str,
        api_key: str | None = None,
    ) -> None:
        self._collection_name = collection
        auth = (
            weaviate.auth.AuthApiKey(api_key=api_key) if api_key else None
        )
        self._client = weaviate.connect_to_custom(
            http_host=host,
            http_port=http_port,
            http_secure=False,
            grpc_host=host,
            grpc_port=grpc_port,
            grpc_secure=False,
            auth_credentials=auth,
            skip_init_checks=False,
        )
        logger.success(
            "Weaviate connected",
            extra={"host": host, "http_port": http_port, "grpc_port": grpc_port},
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        try:
            self._client.close()
        except Exception as e:  # best-effort, shutdown path
            logger.warning(f"Weaviate close raised: {e}")

    def is_ready(self) -> bool:
        try:
            return self._client.is_ready()
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def ensure_schema(self, vector_size: int) -> None:
        """Create the `Document` collection if missing. Idempotent."""
        if self._client.collections.exists(self._collection_name):
            logger.debug(
                "Collection already present",
                extra={"collection": self._collection_name},
            )
            return

        logger.info(
            "Creating collection",
            extra={"collection": self._collection_name, "vector_size": vector_size},
        )

        self._client.collections.create(
            name=self._collection_name,
            description=(
                "Multi-tenant document store. One tenant per company; "
                "dataset_id is an indexed property for within-tenant filtering."
            ),
            vectorizer_config=Configure.Vectorizer.none(),  # BYOV
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE,
                ef_construction=256,
                max_connections=32,
                ef=128,
                quantizer=Configure.VectorIndex.Quantizer.sq(),
            ),
            inverted_index_config=Configure.inverted_index(
                bm25_b=0.75,
                bm25_k1=1.2,
                index_null_state=True,
                index_property_length=True,
            ),
            multi_tenancy_config=Configure.multi_tenancy(
                enabled=True,
                auto_tenant_creation=True,
                auto_tenant_activation=True,
            ),
            properties=[
                Property(
                    name="text",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.WORD,
                    description="Document body; tokenized for BM25.",
                ),
                Property(
                    name="url",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.FIELD,
                    index_filterable=True,
                    index_searchable=False,
                    description="Source URL (exact-match only).",
                ),
                Property(
                    name="dataset_id",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.FIELD,
                    index_filterable=True,
                    index_searchable=False,
                    description="Logical sub-namespace within a tenant.",
                ),
                Property(
                    name="chunk_index",
                    data_type=DataType.INT,
                    index_filterable=True,
                ),
                Property(
                    name="chunk_count",
                    data_type=DataType.INT,
                    index_filterable=False,
                ),
                Property(
                    name="created_at",
                    data_type=DataType.DATE,
                    index_filterable=True,
                ),
                Property(
                    name="rating",
                    data_type=DataType.NUMBER,
                    index_filterable=True,
                    description="Optional numeric score (e.g. review star rating).",
                ),
                Property(
                    name="author",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.FIELD,
                    index_filterable=True,
                    index_searchable=False,
                ),
                Property(
                    name="category",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.FIELD,
                    index_filterable=True,
                    index_searchable=False,
                ),
            ],
        )
        logger.success(
            "Collection created",
            extra={"collection": self._collection_name},
        )

    # ------------------------------------------------------------------
    # Tenants
    # ------------------------------------------------------------------
    def _coll(self, tenant: str):
        return self._client.collections.get(self._collection_name).with_tenant(tenant)

    def list_tenants(self) -> list[str]:
        tenants = self._client.collections.get(self._collection_name).tenants.get()
        return sorted(tenants.keys())

    def tenant_exists(self, tenant: str) -> bool:
        tenants = self._client.collections.get(self._collection_name).tenants.get()
        return tenant in tenants

    def create_tenant(self, tenant: str) -> None:
        coll = self._client.collections.get(self._collection_name)
        try:
            coll.tenants.create([Tenant(name=tenant)])
        except UnexpectedStatusCodeError as e:
            if "already exists" in str(e).lower():
                return
            raise

    def delete_tenant(self, tenant: str) -> None:
        coll = self._client.collections.get(self._collection_name)
        coll.tenants.remove([tenant])

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def upsert_batch(
        self,
        tenant: str,
        dataset_id: str,
        rows: list[dict[str, Any]],
        vectors: list[list[float]],
    ) -> int:
        """Batch-upsert rows. Idempotent by (tenant, dataset_id, chunk_index, text-hash).

        Each row is a dict with at least 'text' and optionally 'url' plus any
        flat metadata (rating, author, category, custom fields, ...). The
        vector list must be the same length as rows.

        Returns the number of objects submitted.
        """
        if len(rows) != len(vectors):
            raise ValueError("rows and vectors must be equal length")
        if not rows:
            return 0

        coll = self._coll(tenant)
        timestamp = datetime.now(timezone.utc)
        chunk_count = len(rows)
        objects: list[DataObject] = []

        for idx, (row, vec) in enumerate(zip(rows, vectors)):
            text = row.get("text") or ""
            if not text.strip():
                continue

            props: dict[str, Any] = {
                "text": text,
                "url": row.get("url", ""),
                "dataset_id": dataset_id,
                "chunk_index": row.get("chunk_index", idx),
                "chunk_count": row.get("chunk_count", chunk_count),
                "created_at": timestamp,
            }

            # Pass-through metadata. Unknown keys are absorbed by Weaviate
            # auto-schema into new typed properties on first write.
            for k, v in row.items():
                if k in {"text", "url", "chunk_index", "chunk_count"}:
                    continue
                if v is None:
                    continue
                props[k] = v

            obj_uuid = self._deterministic_uuid(tenant, dataset_id, props["chunk_index"], text)
            objects.append(DataObject(properties=props, vector=vec, uuid=obj_uuid))

        if not objects:
            return 0

        with coll.batch.dynamic() as batch:
            for obj in objects:
                batch.add_object(
                    properties=obj.properties,
                    vector=obj.vector,
                    uuid=obj.uuid,
                )

        failed = coll.batch.failed_objects
        if failed:
            logger.error(
                f"Batch upsert had {len(failed)} failures",
                extra={
                    "tenant": tenant,
                    "dataset_id": dataset_id,
                    "sample_error": str(failed[0].message) if failed else None,
                },
            )
            raise RuntimeError(
                f"{len(failed)} object(s) failed to write; first error: {failed[0].message}"
            )

        logger.success(
            "Batch upsert complete",
            extra={
                "tenant": tenant,
                "dataset_id": dataset_id,
                "submitted": len(objects),
            },
        )
        return len(objects)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def hybrid_search(
        self,
        tenant: str,
        query_text: str,
        query_vector: list[float],
        *,
        dataset_id: str | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
        alpha: float = 0.5,
    ) -> list[SearchHit]:
        """Hybrid (BM25 + vector) search scoped to a tenant.

        `alpha` is Weaviate's fusion knob: 0.0 = pure BM25, 1.0 = pure vector.
        0.5 is a reasonable default; tune per-query-type in production.
        """
        coll = self._coll(tenant)
        flt = self._build_filter(dataset_id=dataset_id, filters=filters)

        response = coll.query.hybrid(
            query=query_text,
            vector=query_vector,
            alpha=alpha,
            limit=limit,
            fusion_type=HybridFusion.RELATIVE_SCORE,
            filters=flt,
            return_metadata=MetadataQuery(score=True, explain_score=True),
        )

        hits: list[SearchHit] = []
        for obj in response.objects:
            props = obj.properties or {}
            hits.append(
                SearchHit(
                    object_id=str(obj.uuid),
                    score=float(obj.metadata.score or 0.0),
                    text=str(props.get("text", "")),
                    dataset_id=str(props.get("dataset_id", "")),
                    chunk_index=int(props.get("chunk_index", 0) or 0),
                    chunk_count=int(props.get("chunk_count", 0) or 0),
                    created_at=_isoformat_or_empty(props.get("created_at")),
                    metadata={
                        k: v for k, v in props.items() if k not in _CORE_PROPS
                    },
                    explain_score=obj.metadata.explain_score,
                )
            )
        return hits

    # ------------------------------------------------------------------
    # Dataset operations
    # ------------------------------------------------------------------
    def delete_dataset(self, tenant: str, dataset_id: str) -> int:
        """Delete every object in a tenant with the given dataset_id.

        Returns the number of objects deleted (best-effort count from the
        Weaviate response).
        """
        coll = self._coll(tenant)
        result = coll.data.delete_many(
            where=Filter.by_property("dataset_id").equal(dataset_id),
            verbose=True,
        )
        deleted = int(getattr(result, "successful", 0) or 0)
        logger.warning(
            "Dataset deleted",
            extra={"tenant": tenant, "dataset_id": dataset_id, "deleted": deleted},
        )
        return deleted

    def list_datasets(self, tenant: str) -> list[str]:
        """List distinct dataset_ids in a tenant using an aggregate query.

        This is O(number-of-distinct-datasets), not O(number-of-objects).
        """
        coll = self._coll(tenant)
        try:
            agg = coll.aggregate.over_all(
                group_by="dataset_id",
                total_count=False,
            )
            groups = getattr(agg, "groups", None) or []
            return sorted(
                str(g.grouped_by.value) for g in groups if g.grouped_by is not None
            )
        except Exception as e:
            logger.warning(
                f"Aggregate group_by fallback: {e}",
                extra={"tenant": tenant},
            )
            # Fallback: iterate (only runs if the aggregate API misbehaves).
            seen: set[str] = set()
            for obj in coll.iterator(include_vector=False):
                ds = (obj.properties or {}).get("dataset_id")
                if ds:
                    seen.add(str(ds))
            return sorted(seen)

    def tenant_stats(self, tenant: str) -> dict[str, Any]:
        """Return object count and schema info for a tenant."""
        coll = self._coll(tenant)
        count = 0
        try:
            agg = coll.aggregate.over_all(total_count=True)
            count = int(agg.total_count or 0)
        except Exception as e:
            logger.warning(f"Aggregate count failed: {e}", extra={"tenant": tenant})
        return {
            "tenant": tenant,
            "collection": self._collection_name,
            "object_count": count,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _deterministic_uuid(
        tenant: str, dataset_id: str, chunk_index: int | str, text: str
    ) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        key = f"{tenant}::{dataset_id}::{chunk_index}::{digest}"
        return str(uuid.uuid5(_UUID_NAMESPACE, key))

    @staticmethod
    def _build_filter(
        *, dataset_id: str | None, filters: dict[str, Any] | None
    ) -> Filter | None:
        clauses: list[Filter] = []

        if dataset_id:
            ds = dataset_id.strip()
            if ds and ds not in {"string", "null"}:
                clauses.append(Filter.by_property("dataset_id").equal(ds))

        if filters:
            for key, value in filters.items():
                if value is None or value == "":
                    continue
                clauses.append(Filter.by_property(key).equal(value))

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return Filter.all_of(clauses)


def _isoformat_or_empty(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return str(value)
