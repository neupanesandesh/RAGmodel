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
    MatchValue
)
import uuid


class QdrantStore:
    """
    Wrapper for Qdrant vector database operations.
    
    Features:
    - Collection management (create, delete, list)
    - Document storage with metadata
    - Semantic search with optional namespace filtering
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
    
    def create_collection(self, collection_name: str, vector_size: int = 768) -> bool:
        """
        Create a new collection for storing vectors.
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors (default: 768)
        
        Returns:
            True if created successfully
            
        Raises:
            Exception: If collection already exists or creation fails
        """
        try:
            # Check if collection already exists
            collections = self.client.get_collections().collections
            if any(c.name == collection_name for c in collections):
                raise Exception(f"Collection '{collection_name}' already exists")
            
            # Create collection with cosine distance metric
            # Cosine is ideal for normalized embeddings (measures angle, not magnitude)
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            return True
            
        except Exception as e:
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
    
    def add_document(
        self,
        collection_name: str,
        doc_id: str,
        chunks: List[tuple],  # List of (text, chunk_index)
        embeddings: List[List[float]],
        namespace: Optional[str] = None
    ) -> int:
        """
        Add a document's chunks to the collection.
        
        This is the core storage operation. Each chunk becomes a point in Qdrant
        with its embedding and metadata (text, doc_id, namespace, etc).
        
        Args:
            collection_name: Target collection
            doc_id: Unique identifier for the document
            chunks: List of (chunk_text, chunk_index) tuples
            embeddings: List of embedding vectors (one per chunk)
            namespace: Optional namespace for grouping
        
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
            # Prepare points for Qdrant
            points = []
            chunk_count = len(chunks)
            timestamp = datetime.utcnow().isoformat()
            
            for (chunk_text, chunk_index), embedding in zip(chunks, embeddings):
                # Generate unique point ID
                point_id = str(uuid.uuid4())
                
                # Build payload with all metadata
                payload = {
                    "doc_id": doc_id,
                    "chunk_index": chunk_index,
                    "chunk_count": chunk_count,
                    "text": chunk_text,
                    "created_at": timestamp
                }
                
                # Add namespace if provided
                if namespace:
                    payload["namespace"] = namespace
                
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
            
            return len(points)
            
        except Exception as e:
            raise Exception(f"Failed to add document: {str(e)}")
    
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        k: int = 10,
        namespace: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection.
        
        Args:
            collection_name: Collection to search
            query_vector: Query embedding vector
            k: Number of results to return
            namespace: Optional namespace filter
        
        Returns:
            List of search results with score, text, and metadata
            Format: [{"score": float, "text": str, "metadata": dict}, ...]
            
        Raises:
            Exception: If search fails
        """
        if not self.collection_exists(collection_name):
            raise Exception(f"Collection '{collection_name}' does not exist")
        
        try:
            # Build filter if namespace specified
            query_filter = None
            if namespace:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="namespace",
                            match=MatchValue(value=namespace)
                        )
                    ]
                )
            
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
                formatted_results.append({
                    "score": result.score,
                    "text": result.payload.get("text", ""),
                    "metadata": {
                        "doc_id": result.payload.get("doc_id"),
                        "chunk_index": result.payload.get("chunk_index"),
                        "chunk_count": result.payload.get("chunk_count"),
                        "namespace": result.payload.get("namespace"),
                        "created_at": result.payload.get("created_at")
                    }
                })
            
            return formatted_results
            
        except Exception as e:
            raise Exception(f"Search failed: {str(e)}")
    
    def delete_document(self, collection_name: str, doc_id: str) -> int:
        """
        Delete all chunks of a document from the collection.

        Args:
            collection_name: Collection containing the document
            doc_id: Document ID to delete

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
                            key="doc_id",
                            match=MatchValue(value=doc_id)
                        )
                    ]
                ),
                limit=1
            )

            # Check if any points were found
            if not results[0]:  # results is a tuple: (points, next_offset)
                raise Exception(f"Document '{doc_id}' does not exist in collection '{collection_name}'")

            # Delete all points with this doc_id
            self.client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="doc_id",
                            match=MatchValue(value=doc_id)
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


# Example usage
if __name__ == "__main__":
    print("QdrantStore - Vector Database Wrapper")
    print("\nExample usage:")
    print("  store = QdrantStore()")
    print("  store.create_collection('my_collection', vector_size=768)")
    print("  store.add_document('my_collection', 'doc1', chunks, embeddings)")
    print("  results = store.search('my_collection', query_vector, k=10)")
    print("\nNote: Requires Qdrant server running on localhost:6333")