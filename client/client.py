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
        dataset_id: Dataset identifier this chunk belongs to
        chunk_index: Position of this chunk in the document
        chunk_count: Total number of chunks in the document
        created_at: Timestamp when the document was created
        metadata: Optional user metadata (doc_type, location, etc.)
    """
    score: float
    text: str
    dataset_id: str
    chunk_index: int
    chunk_count: int
    created_at: str
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: dict):
        """Create SearchResult from API response dict."""
        # Extract core fields that are always present
        core_fields = {
            "dataset_id", "chunk_index", "chunk_count", "created_at"
        }

        # Separate user metadata from core fields
        user_metadata = {
            k: v for k, v in data["metadata"].items()
            if k not in core_fields and k != "text"
        }

        return cls(
            score=data["score"],
            text=data["text"],
            dataset_id=data["metadata"]["dataset_id"],
            chunk_index=data["metadata"]["chunk_index"],
            chunk_count=data["metadata"]["chunk_count"],
            created_at=data["metadata"]["created_at"],
            metadata=user_metadata if user_metadata else None
        )


class EmbeddingClient:
    """
    Client for the Embedding Service.

    This is the main interface developers use to interact with your service.

    Args:
        service_url: URL of the embedding service (e.g., "http://localhost:8000")
        timeout: Request timeout in seconds (default: 30)

    Example:
        >>> client = EmbeddingClient("http://localhost:8000")
        >>> client.create_collection("auditcity")
        >>> documents = [
        ...     {"url": "https://example.com/1", "text": "Your text here...", "meta": {"rating": 5}}
        ... ]
        >>> client.add_documents_batch(
        ...     collection="auditcity",
        ...     dataset_id="dallas-dentist",
        ...     documents=documents
        ... )
        >>> results = client.search(
        ...     collection="auditcity",
        ...     dataset_id="dallas-dentist",
        ...     query="search query"
        ... )
    """
    
    def __init__(self, service_url: str, timeout: int = 30, api_key: Optional[str] = None):
        self.base_url = service_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        if api_key:
            self.session.headers["X-API-Key"] = api_key
    
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
    
    def create_collection(self, name: str) -> dict:
        """
        Create a new collection for storing documents.

        Vector size is automatically set based on server configuration.

        Args:
            name: Collection name

        Returns:
            Success message

        Example:
            >>> client.create_collection("customer_support")
        """
        return self._make_request(
            "POST",
            "/collections",
            json_data={"name": name}
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

    def list_datasets(self, collection_name: str) -> List[str]:
        """
        List all dataset IDs in a collection.

        Useful for verifying uploads and checking what datasets exist.

        Args:
            collection_name: Name of the collection

        Returns:
            List of dataset_ids

        Example:
            >>> datasets = client.list_datasets("auditcity")
            >>> print(datasets)
            ['dallas-dentist', 'austin-pizza', 'houston-restaurant']
        """
        response = self._make_request("GET", f"/collections/{collection_name}/datasets")
        return response["datasets"]

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

    def add_documents_batch(
        self,
        collection: str,
        dataset_id: str,
        documents: List[Dict[str, Any]]
    ) -> dict:
        """
        Add multiple preprocessed documents in batch (simple format).

        Each document should be in format: {url, text, meta}
        - One document = one chunk (no auto-chunking)
        - Each chunk stores its own url and metadata
        - Perfect for preprocessed reviews, products, Q&A, etc.

        Args:
            collection: Target collection name
            dataset_id: Unique dataset identifier
            documents: List of documents in {url, text, meta} format

        Returns:
            Upload summary with counts and warnings

        Example:
            >>> # Load preprocessed reviews
            >>> with open("reviews_preprocessed.json", "r") as f:
            ...     docs = json.load(f)
            >>>
            >>> # Upload batch
            >>> response = client.add_documents_batch(
            ...     collection="auditcity",
            ...     dataset_id="lavon-family-dental",
            ...     documents=docs  # List of {url, text, meta}
            ... )
            >>>
            >>> print(f"Uploaded {response['chunks_stored']} documents")
        """
        request_body = {
            "documents": documents
        }

        return self._make_request(
            "POST",
            f"/collections/{collection}/documents/batch/{dataset_id}",
            json_data=request_body
        )

    def delete_document(self, collection: str, dataset_id: str) -> dict:
        """
        Delete a document from a collection.

        Args:
            collection: Collection name
            dataset_id: Dataset identifier to delete

        Returns:
            Success message

        Example:
            >>> client.delete_document("auditcity", "dallas-dentist")
        """
        return self._make_request(
            "DELETE",
            f"/collections/{collection}/documents/{dataset_id}"
        )
    
    # Search Operations

    def search(
        self,
        collection: str,
        query: str,
        dataset_id: Optional[str] = None,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents.

        Args:
            collection: Collection to search
            query: Search query text
            dataset_id: Optional dataset to search within (None = search all datasets)
            k: Number of results to return (default: 5)
            filters: Optional metadata filters (e.g., {"doc_type": "reviews", "location": "Dallas"})

        Returns:
            List of SearchResult objects, sorted by similarity

        Examples:
            # Search entire collection
            >>> results = client.search(
            ...     collection="auditcity",
            ...     query="great service",
            ...     k=10
            ... )

            # Search specific dataset
            >>> results = client.search(
            ...     collection="auditcity",
            ...     dataset_id="dallas-dentist",
            ...     query="professional staff",
            ...     k=5
            ... )

            # Search with metadata filters
            >>> results = client.search(
            ...     collection="auditcity",
            ...     query="amazing experience",
            ...     filters={"doc_type": "reviews", "verified": True},
            ...     k=10
            ... )

            # Access results
            >>> for result in results:
            ...     print(f"Score: {result.score:.3f}")
            ...     print(f"Text: {result.text}")
            ...     print(f"Dataset: {result.dataset_id}")
            ...     if result.metadata:
            ...         print(f"Metadata: {result.metadata}")
        """
        # Build request body
        body = {"query": query}
        if filters:
            body["filters"] = filters

        # Build URL based on whether dataset_id is provided
        if dataset_id:
            # Search specific dataset
            url = f"/collections/{collection}/{dataset_id}/search"
        else:
            # Search entire collection
            url = f"/collections/{collection}/search"

        # Add k as query parameter
        params = {"k": k}

        response = self._make_request(
            "POST",
            url,
            json_data=body,
            params=params
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
    print("  from client import EmbeddingClient")
    print("  ")
    print("  client = EmbeddingClient('http://localhost:8000')")
    print("  ")
    print("  # Create collection")
    print("  client.create_collection('auditcity')")
    print("  ")
    print("  # Upload batch documents")
    print("  documents = [")
    print("      {'url': 'https://example.com/1', 'text': 'Great service...', 'meta': {'rating': 5}},")
    print("      {'url': 'https://example.com/2', 'text': 'Amazing experience!', 'meta': {'rating': 5}}")
    print("  ]")
    print("  client.add_documents_batch(")
    print("      collection='auditcity',")
    print("      dataset_id='dallas-dentist',")
    print("      documents=documents")
    print("  )")
    print("  ")
    print("  # Search specific dataset")
    print("  results = client.search(")
    print("      collection='auditcity',")
    print("      dataset_id='dallas-dentist',")
    print("      query='professional staff'")
    print("  )")
    print("  ")
    print("  # Search entire collection")
    print("  all_results = client.search(")
    print("      collection='auditcity',")
    print("      query='great service',")
    print("      k=10")
    print("  )")