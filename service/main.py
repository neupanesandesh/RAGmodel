"""
FastAPI Embedding Service

Main application that exposes REST API endpoints for embedding operations.
"""

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time
import json
from service.config import settings, validate_settings
from service.logging_config import setup_logging, get_component_logger, get_request_logger, get_performance_logger
from service.models import (
    CollectionCreate, CollectionResponse, CollectionList,
    DocumentDelete, DocumentDeleteResponse,
    SimpleDocument, BatchDocumentAdd, BatchDocumentAddResponse,
    SearchRequest, SearchResponse, SearchResult, SearchResultMetadata,
    HealthResponse, ErrorResponse
)
from service.core.embedder import GeminiEmbedder
from service.core.vectorstore import QdrantStore

# Initialize logger for main module
logger = get_component_logger("main")


# Global instances (initialized on startup)
embedder = None
vector_store = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, vector_store

    # Setup logging first
    setup_logging(
        environment=settings.environment,
        log_level=settings.log_level,
        log_dir=settings.log_dir,
        retention_days=settings.log_retention_days,
        rotation_size=settings.log_rotation_size,
    )

    logger.info("="*50)
    logger.info("Starting Embedding Service")
    logger.info("="*50)

    # Validate configuration
    try:
        validate_settings()
        logger.success("Configuration validated successfully")
    except ValueError as e:
        logger.error(f"Startup Failed - Configuration validation error: {e}")
        raise

    # Initialize services
    try:
        embedder = GeminiEmbedder(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            dimensions=settings.embedding_dimension
        )
        logger.success(
            "Gemini embedder initialized",
            extra={"model": settings.gemini_model, "dimensions": settings.embedding_dimension}
        )

        vector_store = QdrantStore(
            url=settings.qdrant_url,
            # api_key=settings.qdrant_api_key
        )
        logger.success("Qdrant vector store connected", extra={"url": settings.qdrant_url})

    except Exception as e:
        logger.exception(f"Failed to initialize services: {e}")
        raise

    logger.success("Service ready and listening", extra={
        "host": settings.service_host,
        "port": settings.service_port,
        "environment": settings.environment
    })
    logger.info("="*50)

    yield

    logger.info("Shutting down gracefully")


# Create FastAPI app
app = FastAPI(
    title="Embedding Service",
    description="RAG service",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware to log all HTTP requests and responses.
    """
    request_logger = get_request_logger()
    start_time = time.time()

    # Log incoming request
    request_logger.info(
        f"{request.method} {request.url.path}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "query_params": str(request.query_params),
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown"),
        }
    )

    # Process request
    try:
        response = await call_next(request)
        process_time = time.time() - start_time

        # Log response
        request_logger.info(
            f"{request.method} {request.url.path} - {response.status_code}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "process_time_ms": round(process_time * 1000, 2),
            }
        )

        # Add custom header with process time
        response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))
        return response

    except Exception as e:
        process_time = time.time() - start_time
        request_logger.error(
            f"{request.method} {request.url.path} - ERROR",
            extra={
                "method": request.method,
                "path": request.url.path,
                "error": str(e),
                "process_time_ms": round(process_time * 1000, 2),
            }
        )
        raise


# Health Check Endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify service status.
    """
    return HealthResponse(
        status="healthy",
        gemini_configured=bool(settings.gemini_api_key),
        qdrant_configured=bool(settings.qdrant_url)
    )


# Collection Endpoints
@app.post("/collections", response_model=dict, status_code=status.HTTP_201_CREATED)
async def create_collection(request: CollectionCreate):
    try:
        vector_store.create_collection(
            collection_name=request.name,
            vector_size=settings.embedding_dimension
        )
        logger.info(
            f"Collection created: {request.name}",
            extra={"collection_name": request.name, "vector_size": settings.embedding_dimension}
        )
        return {
            "message": f"Collection '{request.name}' created successfully",
            "name": request.name,
            "vector_size": settings.embedding_dimension
        }
    except Exception as e:
        logger.error(
            f"Failed to create collection: {request.name}",
            extra={"collection_name": request.name, "error": str(e)}
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@app.get("/collections", response_model=CollectionList)
async def list_collections():
    """
    List all available collections.
    """
    try:
        collections = vector_store.list_collections()
        return CollectionList(collections=collections)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/collections/{collection_name}", response_model=CollectionResponse)
async def get_collection_info(collection_name: str):
    """
    Get information about a specific collection.
    """
    try:
        info = vector_store.get_collection_info(collection_name)
        return CollectionResponse(**info)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@app.get("/collections/{collection_name}/datasets")
async def list_datasets(collection_name: str):
    """
    List all dataset IDs in a collection.

    Useful for verifying uploads and checking what datasets exist.

    Returns:
        List of dataset_ids
    """
    try:
        datasets = vector_store.list_datasets(collection_name)
        return {
            "collection": collection_name,
            "datasets": datasets,
            "count": len(datasets)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@app.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """
    Delete a collection and all its documents.
    """
    try:
        vector_store.delete_collection(collection_name)
        logger.warning(
            f"Collection deleted: {collection_name}",
            extra={"collection_name": collection_name}
        )
        return {"message": f"Collection '{collection_name}' deleted successfully"}
    except Exception as e:
        # Return 404 if collection doesn't exist, 400 for other errors
        error_msg = str(e)
        logger.error(
            f"Failed to delete collection: {collection_name}",
            extra={"collection_name": collection_name, "error": error_msg}
        )
        if "does not exist" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_msg
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )

@app.post("/collections/{collection_name}/documents/batch/{dataset_id}", response_model=BatchDocumentAddResponse)
async def add_documents_batch(
    collection_name: str,
    dataset_id: str,
    request: BatchDocumentAdd
):
    """
    Add multiple preprocessed documents in batch (simple format).

    Each document should be in format: {url, text, meta}
    - One document = one chunk (no auto-chunking)
    - Each chunk stores its own url and metadata
    - Perfect for preprocessed reviews, products, Q&A, etc.

    Args:
        collection_name: Name of the collection
        dataset_id: Unique identifier for this dataset
        request: Batch upload request with list of documents

    Example:
        POST /collections/auditcity/documents/batch/lavon-family-dental
        {
          "documents": [
            {
              "url": "https://review1.com",
              "text": "Great service!",
              "meta": {"rating": 5, "author": "John"}
            },
            {
              "url": "https://review2.com",
              "text": "Not satisfied",
              "meta": {"rating": 2, "author": "Jane"}
            }
          ]
        }
    """
    start_time = time.time()
    perf_logger = get_performance_logger()

    try:
        if not request.documents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No documents provided"
            )

        logger.info(
            f"[1/3] Processing {len(request.documents)} documents...",
            extra={"dataset_id": dataset_id, "documents_count": len(request.documents)}
        )

        # Process each document
        texts = []
        metadatas = []
        warnings = []
        skipped_count = 0

        for idx, doc in enumerate(request.documents):
            # Validate text
            if not doc.text or not doc.text.strip():
                warning = f"Document {idx}: Empty text, skipping"
                warnings.append(warning)
                skipped_count += 1
                continue

            # Build metadata with url
            metadata = {"url": doc.url}

            # Add optional meta fields
            if doc.meta:
                metadata.update(doc.meta)

            texts.append(doc.text)
            metadatas.append(metadata)

        if not texts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No valid documents found. All {len(request.documents)} documents skipped."
            )

        logger.success(
            f"✓ Processed {len(texts)} valid documents ({skipped_count} skipped)",
            extra={
                "dataset_id": dataset_id,
                "valid_documents": len(texts),
                "skipped_documents": skipped_count
            }
        )

        # Log warnings if any
        if warnings:
            for warning in warnings[:10]:
                logger.warning(warning, extra={"dataset_id": dataset_id})
            if len(warnings) > 10:
                logger.warning(
                    f"... and {len(warnings) - 10} more warnings",
                    extra={"dataset_id": dataset_id}
                )

        # Step 2: Generate embeddings
        logger.info(
            f"[2/3] Embedding {len(texts)} documents (this may take several minutes)...",
            extra={"dataset_id": dataset_id, "texts_count": len(texts)}
        )
        embed_start = time.time()
        embeddings = embedder.embed_batch(texts, task_type="RETRIEVAL_DOCUMENT")
        embed_time = time.time() - embed_start

        logger.success(
            f"✓ Embedding complete: {len(texts)} documents embedded in {embed_time:.1f}s",
            extra={"dataset_id": dataset_id, "embedding_time_seconds": round(embed_time, 2)}
        )

        # Step 3: Store in Qdrant
        logger.info(
            f"[3/3] Storing {len(texts)} documents to Qdrant...",
            extra={"dataset_id": dataset_id, "collection": collection_name}
        )
        store_start = time.time()

        # Store each document as a chunk with its metadata
        chunks_stored = 0
        for idx, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
            # Create a single "chunk" tuple for this document
            chunks = [(text, idx)]

            # Store with its specific metadata
            stored = vector_store.add_document(
                collection_name=collection_name,
                doc_id=f"{dataset_id}_doc_{idx}",
                chunks=chunks,
                embeddings=[embedding],
                dataset_id=dataset_id,
                metadata=metadata
            )
            chunks_stored += stored

        store_time = time.time() - store_start

        logger.success(
            f"✓ Storage complete: {chunks_stored} documents stored in {store_time:.1f}s",
            extra={"dataset_id": dataset_id, "chunks_stored": chunks_stored, "storage_time_seconds": round(store_time, 2)}
        )

        total_time = time.time() - start_time

        # Log performance metrics
        perf_extra = {
            "operation": "add_documents_batch",
            "collection": collection_name,
            "dataset_id": dataset_id,
            "documents_processed": len(texts),
            "documents_skipped": skipped_count,
            "chunks_stored": chunks_stored,
            "timing": {
                "total_ms": round(total_time * 1000, 2),
                "embedding_ms": round(embed_time * 1000, 2),
                "storage_ms": round(store_time * 1000, 2),
            }
        }

        perf_logger.info(
            f"Batch documents added: {dataset_id}",
            extra=perf_extra
        )

        logger.success(
            f"✓✓✓ UPLOAD COMPLETE: {dataset_id} → {chunks_stored} documents in {total_time:.1f}s total",
            extra={
                "dataset_id": dataset_id,
                "collection": collection_name,
                "documents": chunks_stored,
                "skipped": skipped_count,
                "total_time_seconds": round(total_time, 2)
            }
        )

        return BatchDocumentAddResponse(
            dataset_id=dataset_id,
            documents_processed=len(texts),
            chunks_stored=chunks_stored,
            documents_skipped=skipped_count,
            warnings=warnings,
            message=f"Successfully uploaded {chunks_stored} documents ({skipped_count} skipped)"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            f"Failed to add batch documents {dataset_id} to {collection_name}: {str(e)}",
            extra={"dataset_id": dataset_id, "collection": collection_name}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add batch documents: {str(e)}"
        )


@app.delete("/collections/{collection_name}/documents/{dataset_id}")
async def delete_document(collection_name: str, dataset_id: str):
    """
    Delete a document/dataset and all its chunks from a collection.
    """
    try:
        vector_store.delete_document(collection_name, dataset_id)
        logger.warning(
            f"Document deleted from {collection_name}: {dataset_id}",
            extra={"collection": collection_name, "dataset_id": dataset_id}
        )
        return {
            "message": f"Document '{dataset_id}' deleted successfully",
            "dataset_id": dataset_id
        }
    except Exception as e:
        # Return 404 if collection/document doesn't exist, 400 for other errors
        error_msg = str(e)
        logger.error(
            f"Failed to delete document {dataset_id} from {collection_name}",
            extra={"collection": collection_name, "dataset_id": dataset_id, "error": error_msg}
        )
        if "does not exist" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_msg
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )


# Search Endpoints

# 1. Search within entire collection (all datasets)
@app.post("/collections/{collection_name}/search", response_model=SearchResponse)
async def search_collection(
    collection_name: str,
    request: SearchRequest,
    k: int = 5
):
    """
    Search for similar documents across entire collection (all datasets).

    Args:
        collection_name: Collection to search in (e.g., 'auditcity')
        request: Search request body with query and optional filters
        k: Number of results to return (default: 5)

    Example:
        POST /collections/auditcity/search?k=10
        Body: {"query": "great service", "filters": {"doc_type": "reviews"}}
    """
    start_time = time.time()
    perf_logger = get_performance_logger()

    try:
        # Step 1: Embed the search query
        embed_start = time.time()
        query_vector = embedder.embed_for_search(request.query)
        embed_time = time.time() - embed_start

        # Step 2: Search in Qdrant (dataset_id=None searches entire collection)
        search_start = time.time()
        results = vector_store.search(
            collection_name=collection_name,
            query_vector=query_vector,
            dataset_id=None,  # Search all datasets
            k=k,
            filters=request.filters
        )
        search_time = time.time() - search_start

        # Step 3: Format results
        formatted_results = [
            SearchResult(
                score=r["score"],
                text=r["text"],
                metadata=SearchResultMetadata(**r["metadata"])
            )
            for r in results
        ]

        total_time = time.time() - start_time

        # Log performance metrics
        perf_logger.info(
            f"Search completed in entire collection: {request.query[:50]}...",
            extra={
                "operation": "search_collection",
                "collection": collection_name,
                "dataset_id": None,
                "search_scope": "entire_collection",
                "filters": request.filters,
                "query_length": len(request.query),
                "k": k,
                "results_count": len(formatted_results),
                "timing": {
                    "total_ms": round(total_time * 1000, 2),
                    "embedding_ms": round(embed_time * 1000, 2),
                    "search_ms": round(search_time * 1000, 2),
                }
            }
        )

        logger.info(
            f"Search in {collection_name} (all datasets) returned {len(formatted_results)} results",
            extra={"collection": collection_name, "dataset_id": None, "results": len(formatted_results)}
        )

        return SearchResponse(
            query=request.query,
            results=formatted_results,
            count=len(formatted_results)
        )

    except Exception as e:
        logger.exception(
            f"Search failed in {collection_name}: {str(e)}",
            extra={"collection": collection_name, "query": request.query[:100]}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


# 2. Search within specific dataset
@app.post("/collections/{collection_name}/{dataset_id}/search", response_model=SearchResponse)
async def search_dataset(
    collection_name: str,
    dataset_id: str,
    request: SearchRequest,
    k: int = 5
):
    """
    Search for similar documents in a specific dataset.

    Args:
        collection_name: Collection to search in (e.g., 'auditcity')
        dataset_id: Dataset identifier to search within (e.g., 'dallas-dentist')
        request: Search request body with query and optional filters
        k: Number of results to return (default: 5)

    Example:
        POST /collections/auditcity/dallas-dentist/search?k=10
        Body: {"query": "great service", "filters": {"doc_type": "reviews"}}
    """
    start_time = time.time()
    perf_logger = get_performance_logger()

    try:
        # Step 1: Embed the search query
        embed_start = time.time()
        query_vector = embedder.embed_for_search(request.query)
        embed_time = time.time() - embed_start

        # Step 2: Search in Qdrant
        search_start = time.time()
        results = vector_store.search(
            collection_name=collection_name,
            query_vector=query_vector,
            dataset_id=dataset_id,
            k=k,
            filters=request.filters
        )
        search_time = time.time() - search_start

        # Step 3: Format results
        formatted_results = [
            SearchResult(
                score=r["score"],
                text=r["text"],
                metadata=SearchResultMetadata(**r["metadata"])
            )
            for r in results
        ]

        total_time = time.time() - start_time

        # Log performance metrics
        perf_logger.info(
            f"Search completed in dataset '{dataset_id}': {request.query[:50]}...",
            extra={
                "operation": "search_dataset",
                "collection": collection_name,
                "dataset_id": dataset_id,
                "search_scope": f"dataset:{dataset_id}",
                "filters": request.filters,
                "query_length": len(request.query),
                "k": k,
                "results_count": len(formatted_results),
                "timing": {
                    "total_ms": round(total_time * 1000, 2),
                    "embedding_ms": round(embed_time * 1000, 2),
                    "search_ms": round(search_time * 1000, 2),
                }
            }
        )

        logger.info(
            f"Search in {collection_name}/{dataset_id} returned {len(formatted_results)} results",
            extra={"collection": collection_name, "dataset_id": dataset_id, "results": len(formatted_results)}
        )

        return SearchResponse(
            query=request.query,
            results=formatted_results,
            count=len(formatted_results)
        )

    except Exception as e:
        logger.exception(
            f"Search failed in {collection_name}/{dataset_id}: {str(e)}",
            extra={"collection": collection_name, "dataset_id": dataset_id, "query": request.query[:100]}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint with service information.
    """
    return {
        "service": "Embedding Service",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


def run_service():
    """
    Entry point for console script.
    Used when running: embedding-service
    """
    import uvicorn
    uvicorn.run(
        "service.main:app",
        host=settings.service_host,
        port=settings.service_port,
        reload=False
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,  # Use app directly when running as script
        host=settings.service_host,
        port=settings.service_port,
        reload=True  # Enable auto-reload during development
    )