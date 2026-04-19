"""Pluggable embedder backends.

Two backends are supported out of the box:

  * ``SentenceTransformerEmbedder`` — local, open source, the default.
    Uses BAAI/bge-small-en-v1.5 by default. Asymmetric encoding
    (query vs. passage) via BGE's recommended query instruction prefix.
    No external services, no API keys.

  * ``GeminiEmbedder`` — Google Gemini ``gemini-embedding-001``. Requires
    ``GEMINI_API_KEY``. Uses ``RETRIEVAL_DOCUMENT`` / ``RETRIEVAL_QUERY``
    task types for asymmetric encoding. Retries with exponential backoff
    and jitter on transient errors.

Both backends implement the :class:`Embedder` protocol. The service
chooses between them via the ``EMBEDDER_BACKEND`` setting; no code path
outside this module knows which backend is active.
"""

from __future__ import annotations

import os
import random
import time
from typing import Protocol

import numpy as np

from service.logging_config import get_component_logger

logger = get_component_logger("embedder")


# BGE-v1.5 recommends this instruction prefix for queries only. Passages
# are encoded without a prefix. See the model card:
# https://huggingface.co/BAAI/bge-small-en-v1.5
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    """Shared embedder interface."""

    @property
    def dimensions(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]: ...


# ---------------------------------------------------------------------------
# Sentence-Transformers (default)
# ---------------------------------------------------------------------------
class SentenceTransformerEmbedder:
    """Local embedder backed by ``sentence-transformers``.

    The model is loaded at construction time so the first request does not
    pay the download/load cost.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str | None = None,
    ) -> None:
        # Imported lazily so the dependency is only required when this
        # backend is actually used.
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self._dimensions = int(self._model.get_sentence_embedding_dimension())
        self._is_bge = "bge" in model_name.lower()

        logger.success(
            "SentenceTransformer embedder loaded",
            extra={
                "model": model_name,
                "dimensions": self._dimensions,
                "device": str(self._model.device),
                "asymmetric_queries": self._is_bge,
            },
        )

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, query: str) -> list[float]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        text = f"{_BGE_QUERY_INSTRUCTION}{query}" if self._is_bge else query
        vector = self._model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vector.tolist()


# ---------------------------------------------------------------------------
# Gemini (optional)
# ---------------------------------------------------------------------------
class GeminiEmbedder:
    """Embedder backed by Gemini's ``gemini-embedding-001`` model."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        dimensions: int = 768,
    ) -> None:
        from google import genai  # lazy import

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._dimensions = dimensions
        logger.success(
            "Gemini embedder initialized",
            extra={"model": model, "dimensions": dimensions},
        )

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, query: str) -> list[float]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        return self._embed([query], task_type="RETRIEVAL_QUERY")[0]

    # ------------------------------------------------------------------
    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        if not texts:
            return []

        batch_size = 90  # Gemini hard limit is 100; 90 leaves headroom.
        all_vectors: list[list[float]] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors = self._call_with_retry(
                batch=batch, task_type=task_type, config_cls=types.EmbedContentConfig
            )
            all_vectors.extend(vectors)

        return all_vectors

    def _call_with_retry(
        self,
        batch: list[str],
        task_type: str,
        config_cls,
        max_retries: int = 3,
    ) -> list[list[float]]:
        for attempt in range(max_retries + 1):
            try:
                result = self._client.models.embed_content(
                    model=self._model,
                    contents=batch,
                    config=config_cls(
                        task_type=task_type,
                        output_dimensionality=self._dimensions,
                    ),
                )
                return [self._normalize(e.values) for e in result.embeddings]
            except Exception as e:
                retryable = any(
                    tok in str(e).lower()
                    for tok in ("429", "rate", "500", "503", "timeout", "connection")
                )
                if attempt >= max_retries or not retryable:
                    raise
                delay = (2**attempt) + random.uniform(0, 1)
                logger.warning(
                    "Gemini embed retry",
                    extra={
                        "attempt": attempt + 1,
                        "delay_s": round(delay, 2),
                        "error": str(e)[:200],
                    },
                )
                time.sleep(delay)
        raise RuntimeError("unreachable")

    @staticmethod
    def _normalize(values: list[float]) -> list[float]:
        arr = np.asarray(values, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm == 0.0:
            return arr.tolist()
        return (arr / norm).tolist()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_embedder(backend: str, **kwargs) -> Embedder:
    """Return an embedder instance for the named backend."""
    backend = backend.lower()
    if backend == "st":
        return SentenceTransformerEmbedder(
            model_name=kwargs.get("model_name", "BAAI/bge-small-en-v1.5"),
            device=kwargs.get("device"),
        )
    if backend == "gemini":
        api_key = kwargs.get("api_key") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GeminiEmbedder requires GEMINI_API_KEY")
        return GeminiEmbedder(
            api_key=api_key,
            model=kwargs.get("model", "gemini-embedding-001"),
            dimensions=kwargs.get("dimensions", 768),
        )
    raise ValueError(f"Unknown embedder backend: {backend!r}")
