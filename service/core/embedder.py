from typing import List, Tuple
import numpy as np
from google import genai
from google.genai import types
from enum import Enum
from service.logging_config import get_component_logger

# Initialize logger for embedder module
logger = get_component_logger("embedder")

class TaskType(str, Enum):
    RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
    RETRIEVAL_QUERY = "RETRIEVAL_QUERY"


class GeminiEmbedder:
    
    def __init__(
        self,
        api_key: str = None,
        model: str ="gemini-embedding-001",
        dimensions: int = 768
    ):
        self.model = model
        self.dimensions = dimensions

        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client()

        logger.debug(
            "GeminiEmbedder initialized",
            extra={"model": self.model, "dimensions": self.dimensions}
        )
            

    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """
        Normalize an embedding vector to unit length.
        Required for 768 and 1536 dimensions (3072 comes pre-normalized).
        
        Normalization ensures we measure angular similarity (direction)
        rather than magnitude, which is correct for semantic comparison.
        
        Args:
            embedding: Raw embedding vector
            
        Returns:
            Normalized embedding vector
        """
        embedding_array = np.array(embedding)
        norm = np.linalg.norm(embedding_array)
        
        if norm == 0:
            return embedding  # Avoid division by zero
        
        normalized = embedding_array / norm
        return normalized.tolist()

    def embed_single(
        self,
        text: str,
        task_type: TaskType = TaskType.RETRIEVAL_DOCUMENT
    ) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            task_type: Either "RETRIEVAL_DOCUMENT" (for indexing) or
                        "RETRIEVAL_QUERY" (for search queries)

        Returns:
            Normalized embedding vector

        Raises:
            ValueError: If text is empty or too long
            Exception: If API call fails
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            logger.debug(
                f"Generating single embedding (task_type={task_type})",
                extra={"text_length": len(text), "task_type": task_type}
            )

            result = self.client.models.embed_content(
                model=self.model,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.dimensions
                )
            )

            # Extract embedding values
            embedding = result.embeddings[0].values

            # Normalize if not using 3072 dimensions
            embedding = self._normalize_embedding(embedding)

            logger.debug(
                "Single embedding generated successfully",
                extra={"vector_dimension": len(embedding)}
            )

            return embedding

        except Exception as e:
            logger.error(
                f"Failed to generate single embedding: {str(e)}",
                extra={"text_length": len(text), "task_type": task_type}
            )
            raise Exception(f"Failed to generate embedding: {str(e)}")

    def embed_batch(
        self,
        texts: List[str],
        task_type: TaskType = TaskType.RETRIEVAL_DOCUMENT
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with automatic batching.

        Handles large document chunking by splitting into sub-batches of 50 texts
        to stay within Gemini API limits (max 100 texts per request).

        Args:
            texts: List of texts to embed
            task_type: Either "RETRIEVAL_DOCUMENT" or "RETRIEVAL_QUERY"

        Returns:
            List of normalized embedding vectors

        Raises:
            ValueError: If texts list is empty
            Exception: If API call fails
        """
        # Gemini API batch size limit (keeping safe margin)
        MAX_BATCH_SIZE = 50

        if not texts:
            raise ValueError("Texts list cannot be empty")

        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            raise ValueError("All texts are empty")

        try:
            total_texts = len(valid_texts)
            num_batches = (total_texts + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE

            logger.info(
                f"Generating batch embeddings for {total_texts} texts in {num_batches} batch(es)",
                extra={
                    "batch_size": total_texts,
                    "sub_batches": num_batches,
                    "task_type": task_type,
                    "total_chars": sum(len(t) for t in valid_texts)
                }
            )

            all_embeddings = []

            # Process in sub-batches
            for i in range(0, total_texts, MAX_BATCH_SIZE):
                batch = valid_texts[i:i + MAX_BATCH_SIZE]
                batch_num = (i // MAX_BATCH_SIZE) + 1

                logger.debug(
                    f"Processing sub-batch {batch_num}/{num_batches} ({len(batch)} texts)",
                    extra={"batch_num": batch_num, "batch_size": len(batch)}
                )

                # Call Gemini API for this sub-batch
                result = self.client.models.embed_content(
                    model=self.model,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self.dimensions
                    )
                )

                # Extract and normalize embeddings from this batch
                for emb in result.embeddings:
                    embedding = emb.values
                    embedding = self._normalize_embedding(embedding)
                    all_embeddings.append(embedding)

                logger.debug(
                    f"Sub-batch {batch_num}/{num_batches} completed",
                    extra={"embeddings_generated": len(result.embeddings)}
                )

            logger.success(
                f"Batch embeddings generated successfully: {len(all_embeddings)} vectors",
                extra={
                    "embeddings_count": len(all_embeddings),
                    "dimension": self.dimensions,
                    "sub_batches": num_batches
                }
            )

            return all_embeddings

        except Exception as e:
            logger.error(
                f"Failed to generate batch embeddings: {str(e)}",
                extra={"batch_size": len(valid_texts), "task_type": task_type}
            )
            raise Exception(f"Failed to generate batch embeddings: {str(e)}")

    def embed_for_indexing(self, text: str) -> List[float]:
        """
        Convenience method: Embed text for indexing (storage).
        Uses RETRIEVAL_DOCUMENT task type.

        Args:
            text: Document text to embed

        Returns:
            Normalized embedding vector
        """
        return self.embed_single(text, task_type=TaskType.RETRIEVAL_DOCUMENT)

    def embed_for_search(self, query: str) -> List[float]:
        """
        Convenience method: Embed query for searching.
        Uses RETRIEVAL_QUERY task type.

        Args:
            query: Search query text

        Returns:
            Normalized embedding vector
        """
        return self.embed_single(query, task_type=TaskType.RETRIEVAL_QUERY)

# if __name__ == "__main__":
#     import os
    
#     # Check if API key is available
#     if not os.getenv("GEMINI_API_KEY"):
#         print("Set GEMINI_API_KEY environment variable to test")
#         print("\nExample usage:")
#         print("  embedder = GeminiEmbedder()")
#         print("  vector = embedder.embed_for_indexing('Your text here')")
#         print("  print(f'Vector dimension: {len(vector)}')")
#     else:
#         # Test with actual API
#         embedder = GeminiEmbedder()
        
#         # Test single embedding
#         text = "What is the meaning of life?"
#         embedding = embedder.embed_for_search(text)
#         print(f"Single embedding dimension: {len(embedding)}")
#         print(f"Normalized (should be ~1.0): {np.linalg.norm(embedding):.6f}")
        
#         # Test batch embedding
#         texts = [
#             "What is the meaning of life?",
#             "How do I bake a cake?",
#             "What is machine learning?"
#         ]
#         embeddings = embedder.embed_batch(texts)
#         print(f"\nBatch embeddings count: {len(embeddings)}")
#         for i, emb in enumerate(embeddings):
#             print(f"  Embedding {i}: {len(emb)} dimensions, norm: {np.linalg.norm(emb):.6f}")