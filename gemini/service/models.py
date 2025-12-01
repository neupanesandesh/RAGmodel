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
class DocumentAdd(BaseModel):
    """Request model for adding a document."""
    doc_id: str = Field(..., description="Unique document identifier", min_length=1)
    text: str = Field(..., description="Document text content", min_length=1)
    namespace: Optional[str] = Field(None, description="Optional namespace for grouping")


class DocumentAddResponse(BaseModel):
    """Response model after adding a document."""
    doc_id: str
    chunks_stored: int
    namespace: Optional[str]
    message: str


class DocumentDelete(BaseModel):
    """Request model for deleting a document."""
    doc_id: str = Field(..., description="Document ID to delete", min_length=1)


class DocumentDeleteResponse(BaseModel):
    """Response model after deleting a document."""
    doc_id: str
    message: str


# Search Models
class SearchRequest(BaseModel):
    """Request model for searching."""
    query: str = Field(..., description="Search query text", min_length=1)
    k: int = Field(10, description="Number of results to return", ge=1, le=100)
    namespace: Optional[str] = Field(None, description="Optional namespace filter")


class SearchResultMetadata(BaseModel):
    """Metadata for a search result."""
    doc_id: str
    chunk_index: int
    chunk_count: int
    namespace: Optional[str]
    created_at: str


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