"""FastAPI entrypoint for the Weaviate-backed RAG service."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.responses import JSONResponse

from service.config import settings, validate_settings
from service.core.embedder import Embedder, build_embedder
from service.core.weaviate_store import WeaviateStore
from service.logging_config import (
    get_component_logger,
    get_performance_logger,
    get_request_logger,
    setup_logging,
)
from service.models import (
    BatchUploadRequest,
    BatchUploadResponse,
    DatasetDeleteResponse,
    DatasetList,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    HybridSearchRequest,
    SearchHitOut,
    SearchResponse,
    TenantCreate,
    TenantInfo,
    TenantList,
)

logger = get_component_logger("main")

# ---------------------------------------------------------------------------
# Auth + rate limiting
# ---------------------------------------------------------------------------
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Depends(_api_key_header)) -> None:
    expected = settings.api_key
    if not expected:
        # No API_KEY configured → auth disabled. In production startup
        # validation refuses to boot without one, so this branch only
        # runs in dev.
        return
    if key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header",
        )


limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{settings.rate_limit_per_minute}/minute"],
)


# ---------------------------------------------------------------------------
# App singletons
# ---------------------------------------------------------------------------
store: WeaviateStore | None = None
embedder: Embedder | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, embedder

    setup_logging(
        environment=settings.environment,
        log_level=settings.log_level,
        log_dir=settings.log_dir,
        retention_days=settings.log_retention_days,
        rotation_size=settings.log_rotation_size,
    )

    logger.info("Starting RAG service")
    validate_settings()

    embedder = build_embedder(
        backend=settings.embedder_backend,
        model_name=settings.st_model,
        api_key=settings.gemini_api_key,
        model=settings.gemini_embedding_model,
        dimensions=settings.embedding_dimension,
    )

    store = WeaviateStore(
        host=settings.weaviate_host,
        http_port=settings.weaviate_http_port,
        grpc_port=settings.weaviate_grpc_port,
        collection=settings.weaviate_collection,
        api_key=settings.weaviate_api_key,
    )
    store.ensure_schema(vector_size=embedder.dimensions)

    logger.success(
        "Service ready",
        extra={
            "embedder": settings.embedder_backend,
            "dimensions": embedder.dimensions,
            "collection": settings.weaviate_collection,
            "port": settings.service_port,
        },
    )

    try:
        yield
    finally:
        logger.info("Shutting down")
        if store is not None:
            store.close()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RAG Embedding Service (Weaviate)",
    description=(
        "Multi-tenant hybrid-search RAG microservice backed by Weaviate. "
        "Tenant-per-company, typed schema, BM25+vector hybrid, "
        "bring-your-own-vectors."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# CORS: strict by default; override via CORS_ALLOWED_ORIGINS env.
_raw_origins = settings.cors_allowed_origins.strip()
_allowed_origins = (
    [o.strip() for o in _raw_origins.split(",") if o.strip()]
    if _raw_origins
    else []
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=bool(_allowed_origins),
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)

# Rate limiting
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": f"Rate limit exceeded: {exc.detail}"},
    )


# Prometheus metrics — exposes /metrics with request/error/duration histograms.
if settings.enable_metrics:
    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


# Request logging middleware
@app.middleware("http")
async def _log_requests(request: Request, call_next):
    req_logger = get_request_logger()
    start = time.time()
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = round((time.time() - start) * 1000, 2)
        req_logger.exception(
            f"{request.method} {request.url.path} — UNCAUGHT",
            extra={
                "method": request.method,
                "path": request.url.path,
                "duration_ms": elapsed_ms,
            },
        )
        raise
    elapsed_ms = round((time.time() - start) * 1000, 2)
    req_logger.info(
        f"{request.method} {request.url.path} {response.status_code}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": elapsed_ms,
        },
    )
    response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
    return response


# ---------------------------------------------------------------------------
# Public endpoints (no auth)
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        weaviate_ready=bool(store and store.is_ready()),
        embedder_backend=settings.embedder_backend,
        embedding_dimension=embedder.dimensions if embedder else 0,
    )


STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
async def root():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return HTMLResponse("<h1>RAG service running</h1><p>See /docs</p>")
    return HTMLResponse(index.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Tenants
# ---------------------------------------------------------------------------
@app.post(
    "/tenants",
    response_model=TenantInfo,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_api_key)],
    tags=["tenants"],
)
async def create_tenant(body: TenantCreate) -> TenantInfo:
    _require_store().create_tenant(body.name)
    return TenantInfo(
        tenant=body.name,
        collection=settings.weaviate_collection,
        object_count=0,
    )


@app.get(
    "/tenants",
    response_model=TenantList,
    dependencies=[Depends(require_api_key)],
    tags=["tenants"],
)
async def list_tenants() -> TenantList:
    tenants = _require_store().list_tenants()
    return TenantList(tenants=tenants, count=len(tenants))


@app.get(
    "/tenants/{tenant}",
    response_model=TenantInfo,
    dependencies=[Depends(require_api_key)],
    tags=["tenants"],
)
async def get_tenant(tenant: str) -> TenantInfo:
    s = _require_store()
    if not s.tenant_exists(tenant):
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant}' not found")
    stats = s.tenant_stats(tenant)
    return TenantInfo(
        tenant=stats["tenant"],
        collection=stats["collection"],
        object_count=stats["object_count"],
    )


@app.delete(
    "/tenants/{tenant}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_api_key)],
    tags=["tenants"],
)
async def delete_tenant(tenant: str) -> None:
    s = _require_store()
    if not s.tenant_exists(tenant):
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant}' not found")
    s.delete_tenant(tenant)


@app.get(
    "/tenants/{tenant}/datasets",
    response_model=DatasetList,
    dependencies=[Depends(require_api_key)],
    tags=["datasets"],
)
async def list_datasets(tenant: str) -> DatasetList:
    s = _require_store()
    if not s.tenant_exists(tenant):
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant}' not found")
    datasets = s.list_datasets(tenant)
    return DatasetList(tenant=tenant, datasets=datasets, count=len(datasets))


# ---------------------------------------------------------------------------
# Batch upload
# ---------------------------------------------------------------------------
@app.post(
    "/tenants/{tenant}/datasets/{dataset_id}/objects/batch",
    response_model=BatchUploadResponse,
    dependencies=[Depends(require_api_key)],
    tags=["ingest"],
)
async def upload_batch(
    tenant: str, dataset_id: str, body: BatchUploadRequest
) -> BatchUploadResponse:
    s = _require_store()
    e = _require_embedder()
    perf = get_performance_logger()
    t0 = time.time()

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    skipped = 0

    for idx, doc in enumerate(body.documents):
        if not doc.text or not doc.text.strip():
            skipped += 1
            warnings.append(f"doc[{idx}]: empty text skipped")
            continue
        row: dict[str, Any] = {"text": doc.text, "url": doc.url}
        if doc.meta:
            for k, v in doc.meta.items():
                if k in {"text", "url"}:
                    continue
                row[k] = v
        rows.append(row)

    if not rows:
        raise HTTPException(
            status_code=400,
            detail=f"All {len(body.documents)} document(s) were invalid",
        )

    t_embed = time.time()
    texts = [r["text"] for r in rows]
    vectors = e.embed_documents(texts)
    embed_ms = round((time.time() - t_embed) * 1000, 2)

    t_store = time.time()
    inserted = s.upsert_batch(tenant=tenant, dataset_id=dataset_id, rows=rows, vectors=vectors)
    store_ms = round((time.time() - t_store) * 1000, 2)

    total_ms = round((time.time() - t0) * 1000, 2)
    perf.info(
        "batch_upload",
        extra={
            "tenant": tenant,
            "dataset_id": dataset_id,
            "inserted": inserted,
            "skipped": skipped,
            "embed_ms": embed_ms,
            "store_ms": store_ms,
            "total_ms": total_ms,
        },
    )

    return BatchUploadResponse(
        tenant=tenant,
        dataset_id=dataset_id,
        inserted=inserted,
        skipped=skipped,
        warnings=warnings[:20],
        timing_ms={"embed": embed_ms, "store": store_ms, "total": total_ms},
    )


@app.delete(
    "/tenants/{tenant}/datasets/{dataset_id}",
    response_model=DatasetDeleteResponse,
    dependencies=[Depends(require_api_key)],
    tags=["datasets"],
)
async def delete_dataset(tenant: str, dataset_id: str) -> DatasetDeleteResponse:
    s = _require_store()
    if not s.tenant_exists(tenant):
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant}' not found")
    deleted = s.delete_dataset(tenant, dataset_id)
    return DatasetDeleteResponse(
        tenant=tenant, dataset_id=dataset_id, objects_deleted=deleted
    )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
@app.post(
    "/tenants/{tenant}/search",
    response_model=SearchResponse,
    dependencies=[Depends(require_api_key)],
    tags=["search"],
)
async def search_tenant(tenant: str, body: HybridSearchRequest) -> SearchResponse:
    return _run_hybrid(tenant=tenant, dataset_id=None, body=body)


@app.post(
    "/tenants/{tenant}/datasets/{dataset_id}/search",
    response_model=SearchResponse,
    dependencies=[Depends(require_api_key)],
    tags=["search"],
)
async def search_dataset(
    tenant: str, dataset_id: str, body: HybridSearchRequest
) -> SearchResponse:
    return _run_hybrid(tenant=tenant, dataset_id=dataset_id, body=body)


# ---------------------------------------------------------------------------
# Generative RAG (retrieval + Gemini generation)
# ---------------------------------------------------------------------------
@app.post(
    "/tenants/{tenant}/generate",
    response_model=GenerateResponse,
    dependencies=[Depends(require_api_key)],
    tags=["rag"],
)
async def generate(tenant: str, body: GenerateRequest) -> GenerateResponse:
    """Retrieve top-k via hybrid search, then ask Gemini to answer grounded
    in those passages. Requires ``GEMINI_API_KEY``; returns 503 otherwise.
    """
    if not settings.gemini_api_key:
        raise HTTPException(
            status_code=503,
            detail="Generation disabled — set GEMINI_API_KEY to enable /generate",
        )

    s = _require_store()
    e = _require_embedder()
    t0 = time.time()

    t_embed = time.time()
    qvec = e.embed_query(body.query)
    embed_ms = round((time.time() - t_embed) * 1000, 2)

    t_search = time.time()
    hits = s.hybrid_search(
        tenant=tenant,
        query_text=body.query,
        query_vector=qvec,
        dataset_id=body.dataset_id,
        filters=body.filters,
        limit=body.limit,
        alpha=body.alpha,
    )
    search_ms = round((time.time() - t_search) * 1000, 2)

    t_gen = time.time()
    answer = _gemini_answer(query=body.query, passages=[h.text for h in hits])
    gen_ms = round((time.time() - t_gen) * 1000, 2)

    total_ms = round((time.time() - t0) * 1000, 2)
    return GenerateResponse(
        tenant=tenant,
        query=body.query,
        answer=answer,
        sources=[_hit_to_out(h) for h in hits],
        timing_ms={
            "embed": embed_ms,
            "search": search_ms,
            "generate": gen_ms,
            "total": total_ms,
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require_store() -> WeaviateStore:
    if store is None:
        raise HTTPException(status_code=503, detail="Store not initialized")
    return store


def _require_embedder() -> Embedder:
    if embedder is None:
        raise HTTPException(status_code=503, detail="Embedder not initialized")
    return embedder


def _run_hybrid(
    *, tenant: str, dataset_id: Optional[str], body: HybridSearchRequest
) -> SearchResponse:
    s = _require_store()
    e = _require_embedder()
    perf = get_performance_logger()
    t0 = time.time()

    t_embed = time.time()
    qvec = e.embed_query(body.query)
    embed_ms = round((time.time() - t_embed) * 1000, 2)

    t_search = time.time()
    hits = s.hybrid_search(
        tenant=tenant,
        query_text=body.query,
        query_vector=qvec,
        dataset_id=dataset_id,
        filters=body.filters,
        limit=body.limit,
        alpha=body.alpha,
    )
    search_ms = round((time.time() - t_search) * 1000, 2)

    total_ms = round((time.time() - t0) * 1000, 2)
    perf.info(
        "hybrid_search",
        extra={
            "tenant": tenant,
            "dataset_id": dataset_id,
            "alpha": body.alpha,
            "limit": body.limit,
            "results": len(hits),
            "embed_ms": embed_ms,
            "search_ms": search_ms,
            "total_ms": total_ms,
        },
    )

    return SearchResponse(
        tenant=tenant,
        dataset_id=dataset_id,
        query=body.query,
        alpha=body.alpha,
        results=[_hit_to_out(h) for h in hits],
        count=len(hits),
        timing_ms={"embed": embed_ms, "search": search_ms, "total": total_ms},
    )


def _hit_to_out(h) -> SearchHitOut:
    from service.models import SearchHitMetadata

    return SearchHitOut(
        object_id=h.object_id,
        score=h.score,
        text=h.text,
        dataset_id=h.dataset_id,
        chunk_index=h.chunk_index,
        chunk_count=h.chunk_count,
        created_at=h.created_at,
        metadata=SearchHitMetadata(**(h.metadata or {})),
        explain_score=h.explain_score,
    )


def _gemini_answer(query: str, passages: list[str]) -> str:
    """Call Gemini to produce a grounded answer. Imports lazily."""
    from google import genai

    client = genai.Client(api_key=settings.gemini_api_key)
    context = "\n\n---\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    prompt = (
        "You are a retrieval-grounded assistant. Answer the user question using "
        "ONLY the numbered passages below. Cite passage numbers in square brackets. "
        "If the passages do not contain the answer, say so.\n\n"
        f"PASSAGES:\n{context}\n\nQUESTION: {query}\n\nANSWER:"
    )
    resp = client.models.generate_content(
        model=settings.gemini_generative_model,
        contents=prompt,
    )
    return (resp.text or "").strip()


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
def run_service() -> None:
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
        "service.main:app",
        host=settings.service_host,
        port=settings.service_port,
        reload=True,
    )
