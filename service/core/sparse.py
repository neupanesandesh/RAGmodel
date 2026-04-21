"""
Sparse encoder for hybrid search.

Wraps FastEmbed's sparse text encoders (BM25 / SPLADE) and emits Qdrant
`SparseVector` payloads usable in both indexing and query paths.

The model loads lazily on first use — if hybrid search is disabled, the
process never pays the download / init cost.
"""

from typing import Iterable, List, Optional

from qdrant_client.http.models import SparseVector

from service.logging_config import get_component_logger

logger = get_component_logger("sparse")


class SparseEncoder:
    """Lazy FastEmbed wrapper producing Qdrant SparseVectors."""

    def __init__(self, model_name: str = "Qdrant/bm25"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        from fastembed import SparseTextEmbedding

        logger.info("loading sparse model", model=self.model_name)
        self._model = SparseTextEmbedding(model_name=self.model_name)
        logger.info("sparse model ready", model=self.model_name)

    def warmup(self) -> None:
        """Force-load the model and run one tiny encode to prime any ONNX session."""
        self._load()
        _ = list(self._model.embed(["warmup"]))

    def _to_sparse(self, embedding) -> SparseVector:
        # FastEmbed sparse outputs expose `.indices` and `.values`.
        return SparseVector(
            indices=list(embedding.indices.tolist()),
            values=list(embedding.values.tolist()),
        )

    def encode_documents(self, texts: Iterable[str]) -> List[SparseVector]:
        self._load()
        return [self._to_sparse(e) for e in self._model.embed(list(texts))]

    def encode_query(self, text: str) -> SparseVector:
        self._load()
        # FastEmbed exposes `query_embed` for the query side when the model
        # distinguishes (e.g. SPLADE); BM25 treats both identically.
        embed_fn = getattr(self._model, "query_embed", None) or self._model.embed
        gen = embed_fn([text])
        return self._to_sparse(next(iter(gen)))


_encoder: Optional[SparseEncoder] = None


def get_sparse_encoder(model_name: str = "Qdrant/bm25") -> SparseEncoder:
    """Return a process-wide sparse encoder, constructing it once."""
    global _encoder
    if _encoder is None or _encoder.model_name != model_name:
        _encoder = SparseEncoder(model_name=model_name)
    return _encoder
