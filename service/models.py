"""API request and response models."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Tenants
# ---------------------------------------------------------------------------
class TenantCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)


class TenantInfo(BaseModel):
    tenant: str
    collection: str
    object_count: int


class TenantList(BaseModel):
    tenants: list[str]
    count: int


# ---------------------------------------------------------------------------
# Datasets (sub-namespaces within a tenant)
# ---------------------------------------------------------------------------
class DatasetList(BaseModel):
    tenant: str
    datasets: list[str]
    count: int


class DatasetDeleteResponse(BaseModel):
    tenant: str
    dataset_id: str
    objects_deleted: int


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
class DocumentIn(BaseModel):
    """One pre-structured record (a review, product, Q&A, etc.).

    Each record is stored as a single atomic object. We do not auto-chunk
    structured inputs because splitting a review mid-sentence throws away
    context the embedding model should see.
    """

    url: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    meta: Optional[dict[str, Any]] = None


class BatchUploadRequest(BaseModel):
    documents: list[DocumentIn] = Field(..., min_length=1)


class BatchUploadResponse(BaseModel):
    tenant: str
    dataset_id: str
    inserted: int
    skipped: int
    warnings: list[str] = []
    timing_ms: dict[str, float]


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
class HybridSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    filters: Optional[dict[str, Any]] = Field(
        default=None,
        description="Equality filters on indexed properties (rating, author, category, ...)",
    )
    alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Hybrid fusion weight. 0.0 = pure BM25, 1.0 = pure vector.",
    )
    limit: int = Field(default=10, ge=1, le=100)


class SearchHitMetadata(BaseModel):
    """Optional / dynamic metadata attached to a hit."""

    model_config = ConfigDict(extra="allow")


class SearchHitOut(BaseModel):
    object_id: str
    score: float
    text: str
    dataset_id: str
    chunk_index: int
    chunk_count: int
    created_at: str
    metadata: SearchHitMetadata
    explain_score: Optional[str] = None


class SearchResponse(BaseModel):
    tenant: str
    dataset_id: Optional[str]
    query: str
    alpha: float
    results: list[SearchHitOut]
    count: int
    timing_ms: dict[str, float]


# ---------------------------------------------------------------------------
# Generative RAG (Day 3 endpoint)
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=1)
    filters: Optional[dict[str, Any]] = None
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    limit: int = Field(default=5, ge=1, le=20)
    dataset_id: Optional[str] = None


class GenerateResponse(BaseModel):
    tenant: str
    query: str
    answer: str
    sources: list[SearchHitOut]
    timing_ms: dict[str, float]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    weaviate_ready: bool
    embedder_backend: str
    embedding_dimension: int
    version: str = "2.0.0"
