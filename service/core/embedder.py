"""
Embedding backends.

Only open-source, locally-running models are supported. No external API keys
are required to operate the service — a fresh clone + `docker compose up` is
enough to produce embeddings.

The active backend is controlled by `EMBEDDER_BACKEND` in config:
- "sentence-transformers" (default): HuggingFace sentence-transformers
- "fastembed": Qdrant's FastEmbed (ONNX runtime, smaller image footprint)

Both implement the `Embedder` Protocol below so the rest of the service
doesn't care which one is in use.
"""

from typing import List, Optional, Protocol, runtime_checkable
from enum import Enum

import numpy as np

from service.logging_config import get_component_logger

logger = get_component_logger("embedder")


class TaskType(str, Enum):
    """Retained for backwards compatibility with earlier API callers."""
    RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
    RETRIEVAL_QUERY = "RETRIEVAL_QUERY"


# BGE models expect a query-side instruction prefix for asymmetric retrieval.
# Document side is passed as-is. See https://huggingface.co/BAAI/bge-small-en-v1.5
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@runtime_checkable
class Embedder(Protocol):
    """Minimal contract implemented by every embedder backend."""

    dimensions: int
    model_name: str

    def embed_batch(
        self,
        texts: List[str],
        task_type: TaskType = TaskType.RETRIEVAL_DOCUMENT,
    ) -> List[List[float]]: ...

    def embed_single(
        self,
        text: str,
        task_type: TaskType = TaskType.RETRIEVAL_DOCUMENT,
    ) -> List[float]: ...

    def embed_for_indexing(self, text: str) -> List[float]: ...

    def embed_for_search(self, query: str) -> List[float]: ...


def _normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a vector. Required for cosine similarity on Qdrant."""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


class SentenceTransformerEmbedder:
    """BGE / MiniLM / etc. via the sentence-transformers library."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: Optional[str] = None,
        expected_dimension: Optional[int] = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device or None)
        self.dimensions = int(self._model.get_sentence_embedding_dimension())

        if expected_dimension and expected_dimension != self.dimensions:
            raise ValueError(
                f"Model {model_name} produces {self.dimensions}-dim vectors, "
                f"but EMBEDDING_DIMENSION={expected_dimension}. Fix the config."
            )

        self._is_bge = "bge" in model_name.lower()

        logger.info(
            "SentenceTransformerEmbedder ready",
            model=model_name,
            dimensions=self.dimensions,
            device=device or "auto",
        )

    def _prep(self, text: str, task_type: TaskType) -> str:
        if self._is_bge and task_type == TaskType.RETRIEVAL_QUERY:
            return _BGE_QUERY_INSTRUCTION + text
        return text

    def embed_batch(
        self,
        texts: List[str],
        task_type: TaskType = TaskType.RETRIEVAL_DOCUMENT,
    ) -> List[List[float]]:
        if not texts:
            raise ValueError("Texts list cannot be empty")

        valid = [t for t in texts if t and t.strip()]
        if not valid:
            raise ValueError("All texts are empty")

        prepped = [self._prep(t, task_type) for t in valid]
        arr = self._model.encode(
            prepped,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.astype(np.float32).tolist() for v in arr]

    def embed_single(
        self,
        text: str,
        task_type: TaskType = TaskType.RETRIEVAL_DOCUMENT,
    ) -> List[float]:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        return self.embed_batch([text], task_type=task_type)[0]

    def embed_for_indexing(self, text: str) -> List[float]:
        return self.embed_single(text, task_type=TaskType.RETRIEVAL_DOCUMENT)

    def embed_for_search(self, query: str) -> List[float]:
        return self.embed_single(query, task_type=TaskType.RETRIEVAL_QUERY)


class FastEmbedEmbedder:
    """ONNX-based embedder from Qdrant's FastEmbed. Smaller image, no torch."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        expected_dimension: Optional[int] = None,
    ):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

        probe = next(iter(self._model.embed(["probe"])))
        self.dimensions = len(probe)

        if expected_dimension and expected_dimension != self.dimensions:
            raise ValueError(
                f"Model {model_name} produces {self.dimensions}-dim vectors, "
                f"but EMBEDDING_DIMENSION={expected_dimension}. Fix the config."
            )

        logger.info(
            "FastEmbedEmbedder ready",
            model=model_name,
            dimensions=self.dimensions,
        )

    def embed_batch(
        self,
        texts: List[str],
        task_type: TaskType = TaskType.RETRIEVAL_DOCUMENT,
    ) -> List[List[float]]:
        if not texts:
            raise ValueError("Texts list cannot be empty")
        valid = [t for t in texts if t and t.strip()]
        if not valid:
            raise ValueError("All texts are empty")

        out: List[List[float]] = []
        for vec in self._model.embed(valid):
            arr = np.asarray(vec, dtype=np.float32)
            arr = _normalize(arr)
            out.append(arr.tolist())
        return out

    def embed_single(
        self,
        text: str,
        task_type: TaskType = TaskType.RETRIEVAL_DOCUMENT,
    ) -> List[float]:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        return self.embed_batch([text], task_type=task_type)[0]

    def embed_for_indexing(self, text: str) -> List[float]:
        return self.embed_single(text, task_type=TaskType.RETRIEVAL_DOCUMENT)

    def embed_for_search(self, query: str) -> List[float]:
        return self.embed_single(query, task_type=TaskType.RETRIEVAL_QUERY)


def build_embedder(
    backend: str,
    model_name: str,
    expected_dimension: int,
    device: Optional[str] = None,
) -> Embedder:
    """Factory: pick a backend by name."""
    backend = backend.lower()
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(
            model_name=model_name,
            device=device,
            expected_dimension=expected_dimension,
        )
    if backend == "fastembed":
        return FastEmbedEmbedder(
            model_name=model_name,
            expected_dimension=expected_dimension,
        )
    raise ValueError(f"Unknown embedder backend: {backend!r}")
