"""
FastAPI entry point for the RAGmodel service.

Public URL surface (`/collections/**`) is preserved for backwards compatibility:
the external *collection name* maps 1:1 to an internal *tenant id* inside a
single multi-tenant Qdrant collection. This lets us showcase Qdrant's native
multi-tenancy + hybrid search without breaking any caller written against v1.
"""

from pathlib import Path
from contextlib import asynccontextmanager
import time

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from service.config import settings, validate_settings
from service.logging_config import (
    get_component_logger,
    get_performance_logger,
    get_request_logger,
    setup_logging,
)
from service.models import (
    BatchDocumentAdd,
    BatchDocumentAddResponse,
    CollectionCreate,
    CollectionList,
    CollectionResponse,
    DiscoverRequest,
    HealthResponse,
    RecommendRequest,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SearchResultMetadata,
    SnapshotInfo,
    SnapshotListResponse,
)
from service.core.embedder import Embedder, TaskType, build_embedder
from service.core.sparse import SparseEncoder, get_sparse_encoder
from service.core.vectorstore import QdrantStore

logger = get_component_logger("main")

embedder: Embedder | None = None
vector_store: QdrantStore | None = None
sparse_encoder: SparseEncoder | None = None


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_admin_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)


def require_api_key(api_key: str | None = Depends(_api_key_header)) -> None:
    if not settings.api_key and not settings.is_production:
        return
    if not api_key or api_key != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")


def require_admin_key(admin_key: str | None = Depends(_admin_key_header)) -> None:
    if not settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin endpoints are disabled (ADMIN_API_KEY not set)",
        )
    if not admin_key or admin_key != settings.admin_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing admin key")


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])


# --------------------------------------------------------------------------
# Lifespan
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, vector_store, sparse_encoder

    setup_logging(environment=settings.environment, log_level=settings.log_level)
    logger.info("starting RAGmodel service", environment=settings.environment)
    validate_settings()

    embedder = build_embedder(
        backend=settings.embedder_backend,
        model_name=settings.embedder_model,
        expected_dimension=settings.embedding_dimension,
        device=settings.embedder_device,
    )
    logger.info(
        "embedder ready",
        backend=settings.embedder_backend,
        model=embedder.model_name,
        dim=embedder.dimensions,
    )

    vector_store = QdrantStore(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        dense_dim=embedder.dimensions,
        api_key=settings.qdrant_api_key,
        timeout=settings.qdrant_timeout,
    )
    await vector_store.ensure_ready(with_sparse=settings.hybrid_search_enabled)
    logger.info("qdrant ready", url=settings.qdrant_url, collection=settings.qdrant_collection)

    if settings.hybrid_search_enabled:
        sparse_encoder = get_sparse_encoder(settings.sparse_model)
        sparse_encoder.warmup()
        logger.info("hybrid search enabled", sparse_model=settings.sparse_model)

    yield

    if vector_store is not None:
        await vector_store.close()
    logger.info("shutting down")


app = FastAPI(
    title="RAGmodel",
    description="Multi-tenant RAG service on Qdrant with hybrid search and native observability.",
    version="2.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)

if settings.metrics_enabled:
    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        excluded_handlers=["/metrics", "/health", "/"],
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_logger = get_request_logger()
    start = time.time()
    try:
        response = await call_next(request)
    except Exception as e:
        elapsed_ms = round((time.time() - start) * 1000, 2)
        request_logger.error(
            "request failed",
            method=request.method,
            path=request.url.path,
            error=str(e),
            elapsed_ms=elapsed_ms,
        )
        raise
    elapsed_ms = round((time.time() - start) * 1000, 2)
    request_logger.info(
        "request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        elapsed_ms=elapsed_ms,
    )
    response.headers["X-Process-Time"] = str(elapsed_ms)
    return response


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        qdrant_configured=bool(settings.qdrant_url),
        embedder_ready=embedder is not None,
        embedder_model=embedder.model_name if embedder else None,
        embedder_dimensions=embedder.dimensions if embedder else None,
    )


# --------------------------------------------------------------------------
# Collections  (external name == internal tenant id)
# --------------------------------------------------------------------------
@app.post(
    "/collections",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_api_key)],
)
async def create_collection(request: CollectionCreate):
    try:
        await vector_store.register_tenant(request.name)
        logger.info("tenant registered via POST /collections", tenant=request.name)
        return {
            "message": f"Collection '{request.name}' registered successfully",
            "name": request.name,
            "vector_size": settings.embedding_dimension,
        }
    except Exception as e:
        logger.error("register tenant failed", tenant=request.name, error=str(e))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/collections", response_model=CollectionList, dependencies=[Depends(require_api_key)])
async def list_collections():
    try:
        return CollectionList(collections=await vector_store.list_tenants())
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get(
    "/collections/{collection_name}",
    response_model=CollectionResponse,
    dependencies=[Depends(require_api_key)],
)
async def get_collection_info(collection_name: str):
    try:
        return CollectionResponse(**await vector_store.tenant_info(collection_name))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get(
    "/collections/{collection_name}/datasets",
    dependencies=[Depends(require_api_key)],
)
async def list_datasets(collection_name: str):
    try:
        datasets = await vector_store.list_datasets(collection_name)
        return {"collection": collection_name, "datasets": datasets, "count": len(datasets)}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.delete(
    "/collections/{collection_name}",
    dependencies=[Depends(require_api_key)],
)
async def delete_collection(collection_name: str):
    try:
        await vector_store.delete_tenant(collection_name)
        logger.warning("tenant deleted", tenant=collection_name)
        return {"message": f"Collection '{collection_name}' deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------
@app.post(
    "/collections/{collection_name}/documents/batch/{dataset_id}",
    response_model=BatchDocumentAddResponse,
    dependencies=[Depends(require_api_key)],
)
@limiter.limit(settings.rate_limit)
async def add_documents_batch(
    request: Request,
    collection_name: str,
    dataset_id: str,
    body: BatchDocumentAdd,
):
    if embedder is None or vector_store is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    perf_logger = get_performance_logger()
    start = time.time()

    if not body.documents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No documents provided")

    texts: list[str] = []
    metadatas: list[dict] = []
    warnings: list[str] = []
    skipped = 0

    for idx, doc in enumerate(body.documents):
        if not doc.text or not doc.text.strip():
            warnings.append(f"Document {idx}: empty text, skipping")
            skipped += 1
            continue
        meta = {"url": doc.url}
        if doc.meta:
            meta.update(doc.meta)
        texts.append(doc.text)
        metadatas.append(meta)

    if not texts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No valid documents (all {len(body.documents)} skipped)",
        )

    embed_start = time.time()
    dense_vectors = embedder.embed_batch(texts, task_type=TaskType.RETRIEVAL_DOCUMENT)
    sparse_vectors = sparse_encoder.encode_documents(texts) if sparse_encoder is not None else None
    embed_ms = round((time.time() - embed_start) * 1000, 2)

    store_start = time.time()
    chunks_stored = await vector_store.add_points(
        tenant_id=collection_name,
        dataset_id=dataset_id,
        texts=texts,
        dense_vectors=dense_vectors,
        metadata_list=metadatas,
        sparse_vectors=sparse_vectors,
    )
    store_ms = round((time.time() - store_start) * 1000, 2)
    total_ms = round((time.time() - start) * 1000, 2)

    perf_logger.info(
        "batch ingest",
        op="add_documents_batch",
        collection=collection_name,
        dataset_id=dataset_id,
        documents=len(texts),
        skipped=skipped,
        chunks_stored=chunks_stored,
        hybrid=sparse_vectors is not None,
        total_ms=total_ms,
        embed_ms=embed_ms,
        store_ms=store_ms,
    )

    return BatchDocumentAddResponse(
        dataset_id=dataset_id,
        documents_processed=len(texts),
        chunks_stored=chunks_stored,
        documents_skipped=skipped,
        warnings=warnings,
        message=f"Uploaded {chunks_stored} documents ({skipped} skipped)",
    )


@app.delete(
    "/collections/{collection_name}/documents/{dataset_id}",
    dependencies=[Depends(require_api_key)],
)
async def delete_document(collection_name: str, dataset_id: str):
    try:
        deleted = await vector_store.delete_dataset(collection_name, dataset_id)
        logger.warning("dataset deleted", tenant=collection_name, dataset=dataset_id, deleted=deleted)
        return {
            "message": f"Document '{dataset_id}' deleted successfully",
            "dataset_id": dataset_id,
            "deleted_points": deleted,
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------
async def _run_search(
    collection_name: str,
    dataset_id: str | None,
    body: SearchRequest,
    k: int,
    hybrid: bool,
) -> SearchResponse:
    if embedder is None or vector_store is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    use_hybrid = hybrid and sparse_encoder is not None

    perf_logger = get_performance_logger()
    start = time.time()

    embed_start = time.time()
    dense_q = embedder.embed_for_search(body.query)
    sparse_q = sparse_encoder.encode_query(body.query) if use_hybrid else None
    embed_ms = round((time.time() - embed_start) * 1000, 2)

    search_start = time.time()
    results = await vector_store.search(
        tenant_id=collection_name,
        dense_query=dense_q,
        dataset_id=dataset_id,
        k=k,
        filters=body.filters,
        sparse_query=sparse_q,
    )
    search_ms = round((time.time() - search_start) * 1000, 2)

    formatted = [
        SearchResult(
            score=r["score"],
            text=r["text"],
            metadata=SearchResultMetadata(**r["metadata"]),
        )
        for r in results
    ]

    perf_logger.info(
        "search",
        collection=collection_name,
        dataset_id=dataset_id,
        k=k,
        hybrid=use_hybrid,
        results=len(formatted),
        total_ms=round((time.time() - start) * 1000, 2),
        embed_ms=embed_ms,
        search_ms=search_ms,
    )

    return SearchResponse(query=body.query, results=formatted, count=len(formatted))


@app.post(
    "/collections/{collection_name}/search",
    response_model=SearchResponse,
    dependencies=[Depends(require_api_key)],
)
@limiter.limit(settings.rate_limit)
async def search_collection(
    request: Request,
    collection_name: str,
    body: SearchRequest,
    k: int = 5,
    hybrid: bool = False,
):
    try:
        return await _run_search(collection_name, None, body, k, hybrid)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("search failed", collection=collection_name, error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post(
    "/collections/{collection_name}/{dataset_id}/search",
    response_model=SearchResponse,
    dependencies=[Depends(require_api_key)],
)
@limiter.limit(settings.rate_limit)
async def search_dataset(
    request: Request,
    collection_name: str,
    dataset_id: str,
    body: SearchRequest,
    k: int = 5,
    hybrid: bool = False,
):
    try:
        return await _run_search(collection_name, dataset_id, body, k, hybrid)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("search failed", collection=collection_name, dataset_id=dataset_id, error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# --------------------------------------------------------------------------
# Recommend / Discover (Qdrant-unique)
# --------------------------------------------------------------------------
@app.post(
    "/collections/{collection_name}/recommend",
    response_model=SearchResponse,
    dependencies=[Depends(require_api_key)],
)
@limiter.limit(settings.rate_limit)
async def recommend(
    request: Request,
    collection_name: str,
    body: RecommendRequest,
    k: int = 5,
    dataset_id: str | None = None,
):
    if embedder is None or vector_store is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    if not body.positive_texts and not body.negative_texts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one positive or negative text",
        )

    positive_vecs = (
        embedder.embed_batch(body.positive_texts, task_type=TaskType.RETRIEVAL_QUERY)
        if body.positive_texts
        else []
    )
    negative_vecs = (
        embedder.embed_batch(body.negative_texts, task_type=TaskType.RETRIEVAL_QUERY)
        if body.negative_texts
        else []
    )

    try:
        results = await vector_store.recommend(
            tenant_id=collection_name,
            positive_vectors=positive_vecs,
            negative_vectors=negative_vecs,
            dataset_id=dataset_id,
            k=k,
            filters=body.filters,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    formatted = [
        SearchResult(
            score=r["score"],
            text=r["text"],
            metadata=SearchResultMetadata(**r["metadata"]),
        )
        for r in results
    ]
    query_summary = f"+{len(body.positive_texts)}/-{len(body.negative_texts)}"
    return SearchResponse(query=query_summary, results=formatted, count=len(formatted))


@app.post(
    "/collections/{collection_name}/discover",
    response_model=SearchResponse,
    dependencies=[Depends(require_api_key)],
)
@limiter.limit(settings.rate_limit)
async def discover(
    request: Request,
    collection_name: str,
    body: DiscoverRequest,
    k: int = 5,
    dataset_id: str | None = None,
):
    if embedder is None or vector_store is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    if body.target is None and not body.context:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide target or at least one context pair",
        )

    target_vec = (
        embedder.embed_for_search(body.target) if body.target else None
    )
    context_pairs: list[tuple[list[float], list[float]]] = []
    for pair in body.context:
        pos_vec = embedder.embed_for_search(pair.positive)
        neg_vec = embedder.embed_for_search(pair.negative)
        context_pairs.append((pos_vec, neg_vec))

    try:
        results = await vector_store.discover(
            tenant_id=collection_name,
            target_vector=target_vec,
            context_pairs=context_pairs,
            dataset_id=dataset_id,
            k=k,
            filters=body.filters,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    formatted = [
        SearchResult(
            score=r["score"],
            text=r["text"],
            metadata=SearchResultMetadata(**r["metadata"]),
        )
        for r in results
    ]
    query_summary = body.target or f"context={len(body.context)} pairs"
    return SearchResponse(query=query_summary, results=formatted, count=len(formatted))


# --------------------------------------------------------------------------
# Admin: snapshots of the shared physical collection
# --------------------------------------------------------------------------
@app.post(
    "/admin/snapshots",
    response_model=SnapshotInfo,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_admin_key)],
)
async def create_snapshot():
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    snapshot = await vector_store.create_snapshot()
    logger.warning("snapshot created", name=snapshot["name"])
    return SnapshotInfo(**snapshot)


@app.get(
    "/admin/snapshots",
    response_model=SnapshotListResponse,
    dependencies=[Depends(require_admin_key)],
)
async def list_snapshots():
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    snapshots = await vector_store.list_snapshots()
    return SnapshotListResponse(snapshots=[SnapshotInfo(**s) for s in snapshots])


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
async def root():
    html_file = STATIC_DIR / "index.html"
    if not html_file.exists():
        return HTMLResponse("<h1>RAGmodel</h1><p>See <a href='/docs'>/docs</a>.</p>")
    return HTMLResponse(content=html_file.read_text(encoding="utf-8"))


@app.get("/api")
async def api_info():
    return {"service": "RAGmodel", "version": "2.0.0", "status": "running", "docs": "/docs"}


def run_service():
    import uvicorn

    uvicorn.run(
        "service.main:app",
        host=settings.service_host,
        port=settings.service_port,
        reload=False,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.service_host,
        port=settings.service_port,
        reload=True,
    )
