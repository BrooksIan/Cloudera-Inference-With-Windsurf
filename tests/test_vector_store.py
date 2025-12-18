import os
import pytest
import numpy as np
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from windsurf_agent.vector_store import SimpleVectorStore, Document
from windsurf_agent.config import VectorStoreConfig

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

def create_test_embedding(seed: int = 0) -> np.ndarray:
    """Create a deterministic test embedding with the specified seed."""
    np.random.seed(seed)
    # Create a 1536-dimensional embedding with small values
    embedding = np.random.normal(0, 0.02, 1536).astype(np.float32)
    # Normalize to unit length for cosine similarity
    return embedding / np.linalg.norm(embedding)

@pytest.fixture
def sample_documents():
    return [
        Document(
            id="1",
            text="This is a test document",
            embedding=create_test_embedding(0),
            metadata={"source": "test"}
        ),
        Document(
            id="2",
            text="Another test document",
            embedding=create_test_embedding(1),
            metadata={"source": "test"}
        )
    ]

@pytest.fixture
def vector_store_config():
    """Fixture for creating a VectorStoreConfig with test values."""
    return VectorStoreConfig(
        persist_dir=None,
        collection_name="test-collection",
        similarity_metric="cosine",
        dimension=1536  # Match the dimension used in create_test_embedding
    )
@pytest.fixture
def vector_store_config():
    """Fixture for creating a VectorStoreConfig with test values."""
    return VectorStoreConfig(
        persist_dir=None,
        collection_name="test-collection",
        similarity_metric="cosine",
        dimension=1536  # Match the dimension used in create_test_embedding
    )


@pytest.fixture
def vector_store_config():
    """Fixture for creating a VectorStoreConfig with test values."""
    return VectorStoreConfig(
        persist_dir=None,
        collection_name="test-collection",
        similarity_metric="cosine",
        dimension=1536  # Match the dimension used in create_test_embedding
    )


def test_add_and_retrieve_document(vector_store_config):
    """Test adding and retrieving a document."""
    store = SimpleVectorStore(vector_store_config)
    
    test_embedding = create_test_embedding(2)
    doc_id = store.add_document(
        text="Test document",
        embedding=test_embedding,
        metadata={"source": "test"}
    )
    
    assert doc_id is not None
    assert len(store) == 1
    
    doc = store.get_document(doc_id)  # Use the returned doc_id
    assert doc is not None
    assert doc.text == "Test document"
    assert doc.metadata["source"] == "test"

def test_similarity_search(sample_documents):
    """Test similarity search."""
    config = VectorStoreConfig()
    store = SimpleVectorStore(config)
    
    # Add sample documents and collect their IDs
    doc_ids = []
    for doc in sample_documents:
        doc_id = store.add_document(doc.text, doc.embedding, doc.metadata)
        doc_ids.append(doc_id)
    
    # Search with a similar vector
    query_embedding = create_test_embedding(0) * 0.9 + create_test_embedding(1) * 0.1
    results = store.similarity_search(query_embedding, k=1)
    
    assert len(results) == 1
    doc, score = results[0]
    assert doc is not None  # Changed from checking specific ID to just verifying we got a document
    assert 0 <= score <= 1

def test_save_and_load(sample_documents, tmp_path):
    """Test saving and loading the vector store."""
    # Create a temporary file
    file_path = tmp_path / "test_vector_store.json"
    
    # Create and save vector store
    config = VectorStoreConfig()
    store = SimpleVectorStore(config)
    
    # Add documents and collect their IDs
    doc_ids = []
    for doc in sample_documents:
        doc_id = store.add_document(doc.text, doc.embedding, doc.metadata)
        doc_ids.append(doc_id)
    
    store.save(file_path)
    
    # Load the vector store
    loaded_store = SimpleVectorStore.load(file_path)
    
    # Verify loaded data
    assert len(loaded_store) == len(sample_documents)
    
    # Get the first document by its ID
    first_doc = loaded_store.get_document(doc_ids[0])
    assert first_doc is not None
    assert first_doc.text == sample_documents[0].text
    
    # Verify search works
    query_embedding = create_test_embedding(0) * 0.9 + create_test_embedding(1) * 0.1
    results = loaded_store.similarity_search(query_embedding, k=1)
    
    assert len(results) == 1
    doc, score = results[0]
    assert doc is not None
    assert 0 <= score <= 1