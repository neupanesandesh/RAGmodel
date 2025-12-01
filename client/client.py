"""
Embedding Service Client Library

Clean Python interface for interacting with the Embedding Service.
Install with: pip install embedding-client
"""

from typing import List, Optional, Dict, Any
import requests
from dataclasses import dataclass


@dataclass
class SearchResult:
    """
    A single search result.
    
    Attributes:
        score: Similarity score (0-1, higher is better)
        text: The matched text chunk
        doc_id: Document ID this chunk belongs to
        chunk_index: Position of this chunk in the document
        namespace: Namespace if specified
        metadata: Additional metadata
    """
    score: float
    text: str
    doc_id: str
    chunk_index: int
    chunk_count: int
    namespace: Optional[str]
    created_at: str
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create SearchResult from API response dict."""
        return cls(
            score=data["score"],
            text=data["text"],
            doc_id=data["metadata"]["doc_id"],
            chunk_index=data["metadata"]["chunk_index"],
            chunk_count=data["metadata"]["chunk_count"],
            namespace=data["metadata"].get("namespace"),
            created_at=data["metadata"]["created_at"]
        )


class EmbeddingClient:
    """
    Client for the Embedding Service.
    
    This is the main interface developers use to interact with your service.
    
    Args:
        service_url: URL of the embedding service (e.g., "https://your-service.com")
        timeout: Request timeout in seconds (default: 30)
    
    Example:
        >>> client = EmbeddingClient("https://your-service.com")
        >>> client.create_collection("my_docs")
        >>> client.add_document("my_docs", "doc1", "Your text here...")
        >>> results = client.search("my_docs", "search query", k=10)
    """
    
    def __init__(self, service_url: str, timeout: int = 30):
        self.base_url = service_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[dict] = None,
        params: Optional[dict] = None
    ) -> dict:
        """
        Internal method to make HTTP requests with error handling.
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                json=json_data,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            # Extract error message from response if available
            try:
                error_detail = e.response.json().get("detail", str(e))
            except:
                error_detail = str(e)
            raise Exception(f"API Error: {error_detail}")
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {str(e)}")
    
    def health_check(self) -> dict:
        """
        Check if the service is healthy and operational.
        
        Returns:
            Health status dictionary
        """
        return self._make_request("GET", "/health")
    
    # Collection Operations
    
    def create_collection(self, name: str, vector_size: int = 768) -> dict:
        """
        Create a new collection for storing documents.
        
        Args:
            name: Collection name
            vector_size: Vector dimension (default: 768, recommended for cost)
        
        Returns:
            Success message
            
        Example:
            >>> client.create_collection("customer_support")
        """
        return self._make_request(
            "POST",
            "/collections",
            json_data={"name": name, "vector_size": vector_size}
        )
    
    def list_collections(self) -> List[str]:
        """
        List all available collections.
        
        Returns:
            List of collection names
        """
        response = self._make_request("GET", "/collections")
        return response["collections"]
    
    def get_collection_info(self, collection_name: str) -> dict:
        """
        Get information about a collection.
        
        Args:
            collection_name: Name of the collection
        
        Returns:
            Collection information (vector count, size, etc.)
        """
        return self._make_request("GET", f"/collections/{collection_name}")
    
    def delete_collection(self, collection_name: str) -> dict:
        """
        Delete a collection and all its documents.
        
        Args:
            collection_name: Name of the collection to delete
        
        Returns:
            Success message
        """
        return self._make_request("DELETE", f"/collections/{collection_name}")
    
    # Document Operations
    
    def add_document(
        self,
        collection: str,
        doc_id: str,
        text: str,
        namespace: Optional[str] = None
    ) -> dict:
        """
        Add a document to a collection.

        The text will be automatically chunked, embedded, and stored.

        Args:
            collection: Target collection name
            doc_id: Unique identifier for this document
            text: Document text content
            namespace: Optional namespace for grouping (e.g., "product_docs")

        Returns:
            Information about chunks stored

        Example:
            >>> client.add_document(
            ...     collection="kb",
            ...     doc_id="article_1",
            ...     text="Long article text...",
            ...     namespace="tutorials"
            ... )
        """
        # Build query parameters
        params = {"doc_id": doc_id}
        if namespace:
            params["namespace"] = namespace

        # Send text as raw body (text/plain)
        url = f"{self.base_url}/collections/{collection}/documents/text"

        try:
            response = self.session.post(
                url,
                params=params,
                data=text,
                headers={"Content-Type": "text/plain"},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            try:
                error_detail = e.response.json().get("detail", str(e))
            except:
                error_detail = str(e)
            raise Exception(f"API Error: {error_detail}")

        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {str(e)}")
    
    def delete_document(self, collection: str, doc_id: str) -> dict:
        """
        Delete a document from a collection.
        
        Args:
            collection: Collection name
            doc_id: Document ID to delete
        
        Returns:
            Success message
        """
        return self._make_request(
            "DELETE",
            f"/collections/{collection}/documents/{doc_id}"
        )
    
    # Search Operations
    
    def search(
        self,
        collection: str,
        query: str,
        k: int = 10,
        namespace: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents in a collection.
        
        Args:
            collection: Collection to search
            query: Search query text
            k: Number of results to return (default: 10)
            namespace: Optional namespace filter
        
        Returns:
            List of SearchResult objects, sorted by similarity
            
        Example:
            >>> results = client.search(
            ...     collection="kb",
            ...     query="How do I reset my password?",
            ...     k=5,
            ...     namespace="troubleshooting"
            ... )
            >>> for result in results:
            ...     print(f"Score: {result.score:.3f}")
            ...     print(f"Text: {result.text}")
        """
        response = self._make_request(
            "POST",
            f"/collections/{collection}/search",
            json_data={
                "query": query,
                "k": k,
                "namespace": namespace
            }
        )
        
        # Convert to SearchResult objects
        return [SearchResult.from_dict(r) for r in response["results"]]


# Convenience function for quick setup
def create_client(service_url: str) -> EmbeddingClient:
    """
    Create a client instance.
    
    Args:
        service_url: URL of your embedding service
    
    Returns:
        EmbeddingClient instance
    """
    return EmbeddingClient(service_url)


if __name__ == "__main__":
    print("Embedding Service Client Library")
    print("\nExample usage:")
    print("  from embedding_client import EmbeddingClient")
    print("  client = EmbeddingClient('https://your-service.com')")
    print("  client.create_collection('my_collection')")
    print("  client.add_document('my_collection', 'doc1', 'text...')")
    print("  results = client.search('my_collection', 'query', k=10)")