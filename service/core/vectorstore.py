"""
Qdrant Vector Store Module

Manages all interactions with Qdrant for storing and searching vectors.
Provides clean abstractions for collection management and vector operations.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType
)
import uuid
from service.logging_config import get_component_logger

# Initialize logger for vectorstore module
logger = get_component_logger("vectorstore")


class QdrantStore:
    """
    Wrapper for Qdrant vector database operations.

    Features:
    - Collection management (create, delete, list)
    - Payload index creation with tenant isolation support
    - Document storage with flexible metadata (tenant-based multitenancy)
    - Semantic search with tenant isolation and flexible filtering
    - Document deletion

    Args:
        url: Qdrant server URL (default: localhost:6333)
        api_key: Optional API key for cloud Qdrant
    """
    
    def __init__(self, url: str = "http://localhost:6333", api_key: Optional[str] = None):
        if api_key:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(url=url)

        logger.debug(f"QdrantStore initialized", extra={"url": url, "has_api_key": bool(api_key)})
    
    def create_collection(self, collection_name: str, vector_size: int = 768, create_document_index: bool = True) -> bool:
        """
        Create a new collection for storing vectors.

        Args:
            collection_name: Name of the collection (company name, e.g., 'auditcity')
            vector_size: Dimension of vectors (default: 768)
            create_document_index: Whether to create dataset_id index (default: True)

        Returns:
            True if created successfully

        Raises:
            Exception: If collection already exists or creation fails
        """
        try:
            # Check if collection already exists
            collections = self.client.get_collections().collections
            if any(c.name == collection_name for c in collections):
                logger.warning(
                    f"Collection already exists: {collection_name}",
                    extra={"collection_name": collection_name}
                )
                raise Exception(f"Collection '{collection_name}' already exists")

            logger.info(
                f"Creating collection: {collection_name}",
                extra={"collection_name": collection_name, "vector_size": vector_size}
            )

            # Create collection with cosine distance metric
            # Cosine is ideal for normalized embeddings (measures angle, not magnitude)
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )

            # Create payload index for dataset_id
            if create_document_index:
                self.create_payload_index(
                    collection_name=collection_name,
                    field_name="dataset_id",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                logger.info(
                    f"Created dataset_id payload index",
                    extra={"collection_name": collection_name}
                )

            logger.success(
                f"Collection created successfully: {collection_name}",
                extra={"collection_name": collection_name, "vector_size": vector_size}
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to create collection: {collection_name}",
                extra={"collection_name": collection_name, "error": str(e)}
            )
            raise Exception(f"Failed to create collection: {str(e)}")
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection and all its vectors.

        Args:
            collection_name: Name of the collection to delete

        Returns:
            True if deleted successfully

        Raises:
            Exception: If collection doesn't exist or deletion fails
        """
        # Check if collection exists first
        if not self.collection_exists(collection_name):
            raise Exception(f"Collection '{collection_name}' does not exist")

        try:
            self.client.delete_collection(collection_name=collection_name)
            return True
        except Exception as e:
            raise Exception(f"Failed to delete collection: {str(e)}")
    
    def list_collections(self) -> List[str]:
        """
        List all collection names.
        
        Returns:
            List of collection names
        """
        collections = self.client.get_collections().collections
        return [c.name for c in collections]
    
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists.

        Args:
            collection_name: Name of the collection

        Returns:
            True if collection exists
        """
        return collection_name in self.list_collections()

    def create_payload_index(
        self,
        collection_name: str,
        field_name: str,
        field_schema: PayloadSchemaType = PayloadSchemaType.KEYWORD
    ) -> bool:
        """
        Create a payload index for efficient filtering.

        Args:
            collection_name: Collection to create index in
            field_name: Payload field to index (e.g., 'tenant_id', 'business_id')
            field_schema: Field type (KEYWORD for exact match, TEXT for full-text search)

        Returns:
            True if index created successfully

        Raises:
            Exception: If collection doesn't exist or index creation fails
        """
        if not self.collection_exists(collection_name):
            raise Exception(f"Collection '{collection_name}' does not exist")

        try:
            logger.info(
                f"Creating payload index: {field_name}",
                extra={
                    "collection": collection_name,
                    "field": field_name,
                    "schema": field_schema
                }
            )

            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema
            )

            logger.success(
                f"Payload index created: {field_name}",
                extra={"collection": collection_name, "field": field_name}
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to create payload index: {field_name}",
                extra={"collection": collection_name, "field": field_name, "error": str(e)}
            )
            raise Exception(f"Failed to create payload index: {str(e)}")

    def add_document(
        self,
        collection_name: str,
        doc_id: str,
        chunks: List[tuple],  # List of (text, chunk_index)
        embeddings: List[List[float]],
        dataset_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add a document's chunks to the collection.

        This is the core storage operation. Each chunk becomes a point in Qdrant
        with its embedding and payload (text, dataset_id, custom metadata, etc).

        Args:
            collection_name: Target collection (company name, e.g., 'auditcity')
            doc_id: Internal identifier (typically same as dataset_id)
            chunks: List of (chunk_text, chunk_index) tuples
            embeddings: List of embedding vectors (one per chunk)
            dataset_id: Document/dataset name (required - e.g., 'dallas-dentist', 'austin-pizza')
            metadata: Optional flexible metadata dict for filtering (doc_type, rating, category, etc.)

        Returns:
            Number of chunks stored

        Raises:
            ValueError: If chunks and embeddings length mismatch
            Exception: If storage fails
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have same length")

        if not self.collection_exists(collection_name):
            raise Exception(f"Collection '{collection_name}' does not exist")

        try:
            logger.info(
                f"Adding document to collection: {dataset_id}",
                extra={
                    "collection": collection_name,
                    "dataset_id": dataset_id,
                    "chunks_count": len(chunks),
                    "metadata": metadata
                }
            )

            # Prepare points for Qdrant
            points = []
            chunk_count = len(chunks)
            timestamp = datetime.utcnow().isoformat()

            for (chunk_text, chunk_index), embedding in zip(chunks, embeddings):
                # Generate unique point ID
                point_id = str(uuid.uuid4())

                # Build payload with core fields
                payload = {
                    "dataset_id": dataset_id,  # Document/dataset name (always available)
                    "chunk_index": chunk_index,
                    "chunk_count": chunk_count,
                    "text": chunk_text,
                    "created_at": timestamp
                }

                # Merge optional metadata if provided
                if metadata:
                    payload.update(metadata)

                # Create point
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
                points.append(point)

            # Upload to Qdrant
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )

            logger.success(
                f"Document added successfully: {dataset_id}",
                extra={
                    "collection": collection_name,
                    "dataset_id": dataset_id,
                    "points_stored": len(points)
                }
            )

            return len(points)

        except Exception as e:
            logger.error(
                f"Failed to add document: {dataset_id}",
                extra={"collection": collection_name, "dataset_id": dataset_id, "error": str(e)}
            )
            raise Exception(f"Failed to add document: {str(e)}")
    
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        dataset_id: Optional[str] = None,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection.

        Args:
            collection_name: Collection to search (company name, e.g., 'auditcity')
            query_vector: Query embedding vector
            dataset_id: Document/dataset name to search within (optional - e.g., 'dallas-dentist').
                        If not provided or empty, searches entire collection.
            k: Number of results to return
            filters: Optional metadata filters (doc_type, rating, category, etc.)

        Returns:
            List of search results with score, text, and metadata
            Format: [{"score": float, "text": str, "metadata": dict}, ...]

        Raises:
            Exception: If search fails
        """
        if not self.collection_exists(collection_name):
            raise Exception(f"Collection '{collection_name}' does not exist")

        try:
            # Normalize dataset_id: treat empty, whitespace, or placeholder values as None
            if dataset_id:
                dataset_id = dataset_id.strip()
                # Treat empty string or common placeholders as None
                if dataset_id in ("", "string", "null"):
                    dataset_id = None

            logger.debug(
                f"Searching in collection: {collection_name}",
                extra={"collection": collection_name, "k": k, "dataset_id": dataset_id, "filters": filters}
            )

            # Build filter conditions
            must_conditions = []

            # Add dataset_id filter if provided (for document-specific search)
            if dataset_id:
                must_conditions.append(
                    FieldCondition(
                        key="dataset_id",
                        match=MatchValue(value=dataset_id)
                    )
                )

            # Add additional metadata filters if provided
            if filters:
                for key, value in filters.items():
                    # Skip empty or None values
                    if value is not None and value != "":
                        must_conditions.append(
                            FieldCondition(
                                key=key,
                                match=MatchValue(value=value)
                            )
                        )

            # Build query filter (None if no conditions)
            query_filter = Filter(must=must_conditions) if must_conditions else None

            # Execute search
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=k,
                query_filter=query_filter
            )

            # Format results for easy consumption
            formatted_results = []
            for result in results:
                # Extract core metadata fields
                metadata = {
                    "dataset_id": result.payload.get("dataset_id"),
                    "chunk_index": result.payload.get("chunk_index"),
                    "chunk_count": result.payload.get("chunk_count"),
                    "created_at": result.payload.get("created_at")
                }

                # Add all other payload fields as flexible metadata
                for key, value in result.payload.items():
                    if key not in ["text", "chunk_index", "chunk_count", "dataset_id", "created_at"]:
                        metadata[key] = value

                formatted_results.append({
                    "score": result.score,
                    "text": result.payload.get("text", ""),
                    "metadata": metadata
                })

            search_scope = f"document '{dataset_id}'" if dataset_id else "entire collection"
            logger.info(
                f"Search completed in {collection_name} ({search_scope})",
                extra={
                    "collection": collection_name,
                    "dataset_id": dataset_id,
                    "search_scope": search_scope,
                    "results_found": len(formatted_results),
                    "requested": k
                }
            )

            return formatted_results

        except Exception as e:
            logger.error(
                f"Search failed in {collection_name}: {str(e)}",
                extra={"collection": collection_name, "error": str(e)}
            )
            raise Exception(f"Search failed: {str(e)}")
    
    def delete_document(self, collection_name: str, dataset_id: str) -> int:
        """
        Delete all chunks of a document from the collection.

        Args:
            collection_name: Collection containing the document
            dataset_id: Document/dataset ID to delete

        Returns:
            Number of chunks deleted

        Raises:
            Exception: If collection or document doesn't exist, or deletion fails
        """
        if not self.collection_exists(collection_name):
            raise Exception(f"Collection '{collection_name}' does not exist")

        try:
            # First check if document exists by searching for it
            results = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="dataset_id",
                            match=MatchValue(value=dataset_id)
                        )
                    ]
                ),
                limit=1
            )

            # Check if any points were found
            if not results[0]:  # results is a tuple: (points, next_offset)
                raise Exception(f"Document '{dataset_id}' does not exist in collection '{collection_name}'")

            # Delete all points with this dataset_id
            self.client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="dataset_id",
                            match=MatchValue(value=dataset_id)
                        )
                    ]
                )
            )

            # Note: Qdrant doesn't return count of deleted items
            # We return 1 to indicate success
            return 1

        except Exception as e:
            raise Exception(f"Failed to delete document: {str(e)}")
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get information about a collection.

        Args:
            collection_name: Collection name

        Returns:
            Dictionary with collection stats
        """
        if not self.collection_exists(collection_name):
            raise Exception(f"Collection '{collection_name}' does not exist")

        info = self.client.get_collection(collection_name=collection_name)
        return {
            "name": collection_name,
            "vector_count": info.points_count,
            "vector_size": info.config.params.vectors.size,
            "distance": info.config.params.vectors.distance
        }

    def list_datasets(self, collection_name: str) -> List[str]:
        """
        List all unique dataset IDs in a collection.

        Args:
            collection_name: Collection name

        Returns:
            List of unique dataset_ids
        """
        if not self.collection_exists(collection_name):
            raise Exception(f"Collection '{collection_name}' does not exist")

        # Use scroll to get all points and extract unique dataset_ids
        dataset_ids = set()
        offset = None

        while True:
            # Scroll through points in batches of 100
            scroll_result = self.client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False  # Don't need vectors, just payload
            )

            points, next_offset = scroll_result

            # Extract dataset_ids from payloads
            for point in points:
                if point.payload and "dataset_id" in point.payload:
                    dataset_ids.add(point.payload["dataset_id"])

            # Check if there are more points
            if next_offset is None:
                break
            offset = next_offset

        return sorted(list(dataset_ids))


# Example usage
if __name__ == "__main__":
    print("QdrantStore - Vector Database Wrapper")
    print("\nExample usage:")
    print("  store = QdrantStore()")
    print("  store.create_collection('embeddings_768d', vector_size=768)")
    print("  store.add_document('embeddings_768d', 'doc1', chunks, embeddings, tenant_id='company-a', metadata={'category': 'reviews'})")
    print("  results = store.search('embeddings_768d', query_vector, tenant_id='company-a', k=10)")
    print("\nNote: Requires Qdrant server running on localhost:6333")