# windsurf_agent/vector_store.py
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import uuid
from dataclasses import dataclass, field
import faiss

from .config import VectorStoreConfig
from .exceptions import VectorStoreError

logger = logging.getLogger(__name__)

@dataclass
class Document:
    """A document with its embedding and metadata."""
    id: str
    text: str
    embedding: np.ndarray
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "embedding": self.embedding.tolist(),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create document from dictionary."""
        return cls(
            id=data["id"],
            text=data["text"],
            embedding=np.array(data["embedding"], dtype=np.float32),
            metadata=data.get("metadata", {})
        )

class SimpleVectorStore:
    """Simple in-memory vector store with FAISS index."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.documents: Dict[str, Document] = {}
        self.index = None
        self.dimension = config.dimension
        self._initialize_index()

    def _initialize_index(self):
        """Initialize the FAISS index."""
        if self.dimension <= 0:
            raise ValueError("Dimension must be a positive integer")
            
        if self.config.similarity_metric == "cosine":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.config.similarity_metric == "euclidean":
            self.index = faiss.IndexFlatL2(self.dimension)
        else:  # dotproduct
            self.index = faiss.IndexFlatIP(self.dimension)

    def add_document(self, text: str, embedding: np.ndarray, metadata: Optional[Dict] = None) -> str:
        """Add a document to the vector store.
        
        Args:
            text: The text content of the document
            embedding: The embedding vector for the document
            metadata: Optional metadata for the document
            
        Returns:
            str: The ID of the added document
            
        Raises:
            ValueError: If the embedding is not 1D or if dimensions don't match
        """
        # Convert embedding to numpy array if it isn't already
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)
        
        # Ensure the embedding is 1D
        if embedding.ndim != 1:
            raise ValueError(f"Expected 1D embedding, got {embedding.ndim}D")
        
        # Initialize index with correct dimension if this is the first document
        if self.index is None:
            self.dimension = len(embedding)
            self._initialize_index()
        # Verify dimension matches for subsequent documents
        elif len(embedding) != self.dimension:
            raise ValueError(
                f"Embedding dimension {len(embedding)} does not match "
                f"index dimension {self.dimension}"
            )
        
        # Ensure metadata is a dictionary
        if metadata is None:
            metadata = {}
        
        # Generate a unique ID for the document
        doc_id = str(uuid.uuid4())
        
        # Create a new document
        doc = Document(
            id=doc_id,
            text=text,
            embedding=embedding,
            metadata=metadata
        )
        
        # Add the document to the store
        self.documents[doc_id] = doc
        
        # Reshape to 2D array as expected by FAISS (n_vectors, dimension)
        embedding_array = embedding.reshape(1, -1)
        self.index.add(embedding_array)
        
        return doc_id

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by its ID."""
        return self.documents.get(doc_id)

    def similarity_search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents to the query embedding.
        
        Args:
            query_embedding: The query embedding vector
            k: Number of results to return
            
        Returns:
            List of (document, similarity_score) tuples, sorted by similarity
        """
        if not self.documents:
            return []
            
        # Reshape query to 2D array as expected by FAISS
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        
        # Search the index
        distances, indices = self.index.search(query_embedding, k=min(k, len(self.documents)))
        
        # Convert to list of (document, score) tuples
        results = []
        doc_list = list(self.documents.values())
        
        for i, idx in enumerate(indices[0]):
            if idx < 0:  # Skip invalid indices
                continue
                
            doc = doc_list[idx]
            score = float(distances[0][i])
            
            # Convert cosine similarity to [0, 1] range if using cosine
            if self.config.similarity_metric == "cosine":
                score = (score + 1) / 2  # Convert from [-1, 1] to [0, 1]
            
            results.append((doc, score))
        
        return results

    def save(self, file_path: Union[str, Path]) -> None:
        """
        Save the vector store to a file.
        
        Args:
            file_path: Path to save the vector store to
        """
        data = {
            'config': self.config.dict(),
            'documents': [doc.to_dict() for doc in self.documents.values()]
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, file_path: Union[str, Path]) -> 'SimpleVectorStore':
        """
        Load a vector store from a file.
        
        Args:
            file_path: Path to load the vector store from
            
        Returns:
            Loaded SimpleVectorStore instance
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Create a new instance with the saved config
        config = VectorStoreConfig(**data['config'])
        store = cls(config)
        
        # Clear the default empty index
        store.documents = {}
        store.index = None
        store._initialize_index()
        
        # Add all documents
        for doc_data in data['documents']:
            doc = Document.from_dict(doc_data)
            store.documents[doc.id] = doc
            store.index.add(np.array([doc.embedding]))
        
        return store

    def add_documents(
        self, 
        texts: List[str], 
        embeddings: List[np.ndarray], 
        metadatas: Optional[List[dict]] = None
    ) -> List[str]:
        """Add multiple documents to the vector store."""
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts must match number of embeddings")
            
        if metadatas is None:
            metadatas = [{}] * len(texts)
        elif len(texts) != len(metadatas):
            raise ValueError("Number of texts must match number of metadata dictionaries")

        doc_ids = []
        for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
            try:
                doc_id = self.add_document(text, embedding, metadata)
                doc_ids.append(doc_id)
            except Exception as e:
                logger.error(f"Failed to add document at index {i}: {str(e)}")
                raise VectorStoreError(f"Failed to add document: {str(e)}") from e

        return doc_ids

    def similarity_search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 4,
        filter_func: Optional[callable] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents to the query embedding.

        Args:
            query_embedding: The query embedding vector
            k: Number of results to return
            filter_func: Optional function to filter documents before returning results
    
        Returns:
            List of (document, similarity_score) tuples, sorted by similarity
        """
        if not self.documents:
            return []
        
        # Initialize results list and get document values
        results = []
        doc_list = list(self.documents.values())
        
        # Convert to numpy array and ensure correct shape for FAISS
        query_embedding = np.asarray(query_embedding, dtype=np.float32).flatten()
        query_embedding = query_embedding.reshape(1, -1)  # Reshape to 2D array as expected by FAISS
    
        # Search the index
        distances, indices = self.index.search(query_embedding, k=min(k, len(self.documents)))
    
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(doc_list):  # Skip invalid indices
                continue
        
            doc = doc_list[idx]
            score = float(distances[0][i])
    
            # Convert cosine similarity to [0, 1] range if using cosine 
            if self.config.similarity_metric == "cosine":
                score = (score + 1) / 2  # Convert from [-1, 1] to [0, 1]
    
            # Apply filter if provided
            if filter_func is None or filter_func(doc):
                results.append((doc, score))
                
                # Early exit if we've found enough results
                if len(results) >= k:
                    break
    
        return results

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self.documents.get(doc_id)

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        if doc_id in self.documents:
            # Rebuild index without the deleted document
            del self.documents[doc_id]
            self._rebuild_index()
            return True
        return False

    def _rebuild_index(self):
        """Rebuild the FAISS index from documents."""
        if not self.documents:
            self.index = None
            return

        embeddings = np.array([doc.embedding for doc in self.documents.values()], dtype=np.float32)
        self._initialize_index()
        self.index.add(embeddings)

    def save(self, file_path: Union[str, Path]):
        """Save the vector store to disk."""
        if not self.documents:
            return

        data = {
            "documents": [doc.to_dict() for doc in self.documents.values()],
            "config": {
                "dimension": self.dimension,
                "similarity_metric": self.config.similarity_metric,
                "collection_name": self.config.collection_name
            }
        }

        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        except Exception as e:
            raise VectorStoreError(f"Failed to save vector store: {str(e)}") from e

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> 'SimpleVectorStore':
        """Load a vector store from disk."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            config = VectorStoreConfig(
                dimension=data["config"]["dimension"],
                similarity_metric=data["config"]["similarity_metric"],
                collection_name=data["config"]["collection_name"]
            )
            
            store = cls(config)
            
            for doc_data in data.get("documents", []):
                doc = Document.from_dict(doc_data)
                store.documents[doc.id] = doc
                store.index.add(np.array([doc.embedding], dtype=np.float32))
                
            return store
            
        except Exception as e:
            raise VectorStoreError(f"Failed to load vector store: {str(e)}") from e

    def __len__(self) -> int:
        """Return the number of documents in the store."""
        return len(self.documents)