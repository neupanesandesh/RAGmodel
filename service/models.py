"""
API Models

Pydantic models for request validation and response formatting.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# Collection Models
class CollectionCreate(BaseModel):
    """Request model for creating a collection."""
    name: str = Field(..., description="Collection name", min_length=1, max_length=100)
    # vector_size: int = Field(768, description="Vector dimension size", ge=128, le=3072)


class CollectionResponse(BaseModel):
    """Response model for collection information."""
    name: str
    vector_count: int
    vector_size: int
    distance: str


class CollectionList(BaseModel):
    """Response model for listing collections."""
    collections: List[str]


# Document Models
class DocumentDelete(BaseModel):
    """Request model for deleting a document."""
    dataset_id: str = Field(..., description="Dataset identifier to delete", min_length=1)


class DocumentDeleteResponse(BaseModel):
    """Response model after deleting a document."""
    dataset_id: str
    message: str


class SimpleDocument(BaseModel):
    """Single document in simple format."""
    url: str = Field(..., description="Source URL for this document", min_length=1)
    text: str = Field(..., description="Text content to embed", min_length=1)
    meta: Optional[Dict[str, Any]] = Field(None, description="Optional metadata (rating, author, date, etc.)")


class BatchDocumentAdd(BaseModel):
    """Request model for batch upload of preprocessed documents."""
    documents: List[SimpleDocument] = Field(..., description="List of documents in {url, text, meta} format")


class BatchDocumentAddResponse(BaseModel):
    """Response model after batch document upload."""
    dataset_id: str
    documents_processed: int
    chunks_stored: int
    documents_skipped: int
    warnings: List[str] = []
    message: str


# Search Models
class SearchRequest(BaseModel):
    """Request model for searching."""
    query: str = Field(..., description="Search query text", min_length=1)
    filters: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional metadata filters (doc_type, rating, category, etc.)",
        examples=[{"doc_type": "reviews"}, None]
    )


class SearchResultMetadata(BaseModel):
    """Metadata for a search result."""
    model_config = {"extra": "allow"}  # Allow additional flexible fields

    dataset_id: str  # Dataset identifier
    chunk_index: int
    chunk_count: int
    created_at: str
    # Additional metadata fields (doc_type, rating, category, etc.) allowed dynamically


class SearchResult(BaseModel):
    """Single search result."""
    score: float = Field(..., description="Similarity score (0-1, higher is better)")
    text: str = Field(..., description="Matched text chunk")
    metadata: SearchResultMetadata


class SearchResponse(BaseModel):
    """Response model for search results."""
    query: str
    results: List[SearchResult]
    count: int


# Error Models
class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None


# Health Check
class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    gemini_configured: bool
    qdrant_configured: bool
    version: str = "1.0.0"


# 