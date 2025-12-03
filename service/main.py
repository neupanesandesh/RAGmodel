"""
FastAPI Embedding Service

Main application that exposes REST API endpoints for embedding operations.
"""

from fastapi import FastAPI, HTTPException, status, File, UploadFile, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from service.config import settings, validate_settings
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



# Global instances (initialized on startup)
embedder = None
vector_store = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, vector_store
    
    print("\n" + "="*50)
    print("Starting Embedding Service")
    print("="*50)
    
    try:
        validate_settings()
    except ValueError as e:
        print(f"\n❌ Startup Failed: {e}")
        raise
    
    # Initialize services
    try:
        embedder = GeminiEmbedder(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            dimensions=settings.embedding_dimension
        )
        print("✓ Gemini embedder initialized")
        
        vector_store = QdrantStore(
            url=settings.qdrant_url,
            # api_key=settings.qdrant_api_key
        )
        print("✓ Qdrant vector store connected")
        
    except Exception as e:
        print(f"\n❌ Failed to initialize services: {e}")
        raise
    
    print("\n🚀 Service ready!")
    print(f"   Listening on {settings.service_host}:{settings.service_port}")
    print("="*50 + "\n")
    
    yield
    
    print("\n✓ Shutting down gracefully")


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
        return {
            "message": f"Collection '{request.name}' created successfully",
            "name": request.name,
            "vector_size": settings.embedding_dimension
        }
    except Exception as e:
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
        return {"message": f"Collection '{collection_name}' deleted successfully"}
    except Exception as e:
        # Return 404 if collection doesn't exist, 400 for other errors
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


# Raw Text Document Endpoint
@app.post("/collections/{collection_name}/documents/text", response_model=DocumentAddResponse)
async def add_document_raw_text(
    collection_name: str,
    doc_id: str,
    namespace: str = None,
    text: str = Body(..., media_type="text/plain")
):

    try:
        # Validate text
        if not text or not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text content cannot be empty"
            )

        # Step 1: Chunk the text
        chunks = chunk_text(text)
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text resulted in no valid chunks"
            )

        # Step 2: Embed all chunks
        chunk_texts = [text for text, _ in chunks]
        embeddings = embedder.embed_batch(chunk_texts, task_type="RETRIEVAL_DOCUMENT")

        # Step 3: Store in Qdrant
        chunks_stored = vector_store.add_document(
            collection_name=collection_name,
            doc_id=doc_id,
            chunks=chunks,
            embeddings=embeddings,
            namespace=namespace
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
    try:
        # Step 1: Embed the search query
        query_vector = embedder.embed_for_search(request.query)
        
        # Step 2: Search in Qdrant
        results = vector_store.search(
            collection_name=collection_name,
            query_vector=query_vector,
            k=request.k,
            namespace=request.namespace
        )
        
        # Step 3: Format results
        formatted_results = [
            SearchResult(
                score=r["score"],
                text=r["text"],
                metadata=SearchResultMetadata(**r["metadata"])
            )
            for r in results
        ]
        
        return SearchResponse(
            query=request.query,
            results=formatted_results,
            count=len(formatted_results)
        )
        
    except Exception as e:
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