"""
FastAPI Embedding Service

Main application that exposes REST API endpoints for embedding operations.
"""

from fastapi import FastAPI, HTTPException, status, File, UploadFile, Form, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time
from service.config import settings, validate_settings
from service.logging_config import setup_logging, get_component_logger, get_request_logger, get_performance_logger
from service.models import (
    CollectionCreate, CollectionResponse, CollectionList,
    DocumentAdd, DocumentAddResponse,
    DocumentDelete, DocumentDeleteResponse,
    SearchRequest, SearchResponse, SearchResult, SearchResultMetadata,
    HealthResponse, ErrorResponse
)
from service.core.chunking import chunk_text
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


# Raw Text Document Endpoint
@app.post("/collections/{collection_name}/documents/text", response_model=DocumentAddResponse)
async def add_document_raw_text(
    collection_name: str,
    doc_id: str,
    namespace: str = None,
    text: str = Body(..., media_type="text/plain", max_length=10_000_000)  # 10MB max (~2M words, ~500 pages)
):
    start_time = time.time()
    perf_logger = get_performance_logger()

    try:
        # Validate text
        if not text or not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text content cannot be empty"
            )

        # Step 1: Chunk the text
        chunk_start = time.time()
        chunks = chunk_text(text)
        chunk_time = time.time() - chunk_start

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text resulted in no valid chunks"
            )

        # Step 2: Embed all chunks
        embed_start = time.time()
        chunk_texts = [text for text, _ in chunks]
        embeddings = embedder.embed_batch(chunk_texts, task_type="RETRIEVAL_DOCUMENT")
        embed_time = time.time() - embed_start

        # Step 3: Store in Qdrant
        store_start = time.time()
        chunks_stored = vector_store.add_document(
            collection_name=collection_name,
            doc_id=doc_id,
            chunks=chunks,
            embeddings=embeddings,
            namespace=namespace
        )
        store_time = time.time() - store_start

        total_time = time.time() - start_time

        # Log performance metrics
        perf_logger.info(
            f"Document added: {doc_id}",
            extra={
                "operation": "add_document",
                "doc_id": doc_id,
                "collection": collection_name,
                "namespace": namespace,
                "chunks_count": chunks_stored,
                "text_length": len(text),
                "timing": {
                    "total_ms": round(total_time * 1000, 2),
                    "chunking_ms": round(chunk_time * 1000, 2),
                    "embedding_ms": round(embed_time * 1000, 2),
                    "storage_ms": round(store_time * 1000, 2),
                }
            }
        )

        logger.success(
            f"Document added to {collection_name}: {doc_id} ({chunks_stored} chunks)",
            extra={"doc_id": doc_id, "chunks": chunks_stored}
        )

        return DocumentAddResponse(
            doc_id=doc_id,
            chunks_stored=chunks_stored,
            namespace=namespace,
            message=f"Document added successfully with {chunks_stored} chunks"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            f"Failed to add document {doc_id} to {collection_name}: {str(e)}",
            extra={"doc_id": doc_id, "collection": collection_name}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add document: {str(e)}"
        )



@app.delete("/collections/{collection_name}/documents/{doc_id}")
async def delete_document(collection_name: str, doc_id: str):
    """
    Delete a document and all its chunks from a collection.
    """
    try:
        vector_store.delete_document(collection_name, doc_id)
        return {
            "message": f"Document '{doc_id}' deleted successfully",
            "doc_id": doc_id
        }
    except Exception as e:
        # Return 404 if collection/document doesn't exist, 400 for other errors
        error_msg = str(e)
        if "does not exist" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_msg
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )


# Search Endpoint
@app.post("/collections/{collection_name}/search", response_model=SearchResponse)
async def search(collection_name: str, request: SearchRequest):
    """
    Search for similar documents in a collection.

    This is the retrieval operation - developers search for relevant content.
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
            k=request.k,
            namespace=request.namespace
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
            f"Search completed: {request.query[:50]}...",
            extra={
                "operation": "search",
                "collection": collection_name,
                "namespace": request.namespace,
                "query_length": len(request.query),
                "k": request.k,
                "results_count": len(formatted_results),
                "timing": {
                    "total_ms": round(total_time * 1000, 2),
                    "embedding_ms": round(embed_time * 1000, 2),
                    "search_ms": round(search_time * 1000, 2),
                }
            }
        )

        logger.info(
            f"Search in {collection_name} returned {len(formatted_results)} results",
            extra={"collection": collection_name, "results": len(formatted_results)}
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