"""Tests for the main agent class."""
import pytest
from unittest.mock import patch, MagicMock, ANY
import numpy as np

from windsurf_agent.agent import WindsurfAgent
from windsurf_agent.config import Config
from windsurf_agent.exceptions import VectorStoreError, LLMError

@patch('windsurf_agent.agent.WindsurfEmbeddingClient')
@patch('windsurf_agent.agent.WindsurfLLMClient')
@patch('windsurf_agent.agent.SimpleVectorStore')
def test_agent_initialization(mock_vector_store, mock_llm_client, mock_embedding_client, mock_config):
    """Test agent initialization with default config."""
    # Initialize agent with mock config
    agent = WindsurfAgent(mock_config)
    
    # Assert clients were initialized with correct config
    mock_embedding_client.assert_called_once_with(mock_config.embedding)
    mock_llm_client.assert_called_once_with(mock_config.llm)
    mock_vector_store.assert_called_once_with(mock_config.vector_store)
    
    # Assert clients are set as attributes
    assert agent.embedding_client == mock_embedding_client.return_value
    assert agent.llm_client == mock_llm_client.return_value
    assert agent.vector_store == mock_vector_store.return_value

@patch('windsurf_agent.agent.WindsurfEmbeddingClient')
@patch('windsurf_agent.agent.WindsurfLLMClient')
@patch('windsurf_agent.agent.SimpleVectorStore')
def test_embed(mock_vector_store, mock_llm_client, mock_embedding_client, mock_config):
    """Test embedding generation through the agent."""
    # Setup mock embedding client
    mock_embedding = [0.1] * 1536
    mock_embedding_client.return_value.get_embedding.return_value = mock_embedding
    
    # Initialize agent and call embed
    agent = WindsurfAgent(mock_config)
    result = agent.embed("test text")
    
    # Assertions
    assert result == mock_embedding
    agent.embedding_client.get_embedding.assert_called_once_with("test text")

@patch('windsurf_agent.agent.WindsurfEmbeddingClient')
@patch('windsurf_agent.agent.WindsurfLLMClient')
@patch('windsurf_agent.agent.SimpleVectorStore')
def test_add_to_knowledge_base(mock_vector_store, mock_llm_client, mock_embedding_client, mock_config):
    """Test adding documents to the knowledge base."""
    # Setup mocks
    mock_embeddings = [
        [0.1] * 1536,
        [0.2] * 1536
    ]
    mock_doc_ids = ["doc1", "doc2"]
    
    mock_embedding_client.return_value.get_embeddings.return_value = mock_embeddings
    mock_vector_store.return_value.add_documents.return_value = mock_doc_ids
    
    # Initialize agent and add documents
    agent = WindsurfAgent(mock_config)
    texts = ["test text 1", "test text 2"]
    metadatas = [{"source": "test1"}, {"source": "test2"}]
    
    result = agent.add_to_knowledge_base(texts, metadatas)
    
    # Assertions
    assert result == mock_doc_ids
    agent.embedding_client.get_embeddings.assert_called_once_with(texts)
    agent.vector_store.add_documents.assert_called_once_with(
        texts, 
        mock_embeddings, 
        metadatas
    )

@patch('windsurf_agent.agent.WindsurfEmbeddingClient')
@patch('windsurf_agent.agent.WindsurfLLMClient')
@patch('windsurf_agent.agent.SimpleVectorStore')
def test_search(mock_vector_store, mock_llm_client, mock_embedding_client, mock_config):
    """Test searching the knowledge base."""
    # Setup mocks
    from windsurf_agent.vector_store import Document
    
    mock_embedding = [0.1] * 1536
    mock_docs = [
        (Document(id="doc1", text="test doc 1", embedding=np.array([0.1] * 1536), metadata={"source": "test1"}), 0.9),
        (Document(id="doc2", text="test doc 2", embedding=np.array([0.2] * 1536), metadata={"source": "test2"}), 0.8)
    ]
    
    mock_embedding_client.return_value.get_embedding.return_value = mock_embedding
    mock_vector_store.return_value.similarity_search.return_value = mock_docs
    
    # Initialize agent and search
    agent = WindsurfAgent(mock_config)
    results = agent.search("test query", k=2)
    
    # Assertions
    assert len(results) == 2
    assert results[0][0].text == "test doc 1"
    assert results[1][0].text == "test doc 2"
    agent.embedding_client.get_embedding.assert_called_once_with("test query")
    agent.vector_store.similarity_search.assert_called_once_with(
        np.array(mock_embedding, dtype=np.float32),
        k=2,
        filter_func=None
    )

@patch('windsurf_agent.agent.WindsurfEmbeddingClient')
@patch('windsurf_agent.agent.WindsurfLLMClient')
@patch('windsurf_agent.agent.SimpleVectorStore')
def test_generate_response(mock_vector_store, mock_llm_client, mock_embedding_client, mock_config):
    """Test generating a response with context from the knowledge base."""
    # Setup mocks
    mock_embedding = [0.1] * 1536
    mock_docs = [
        (MagicMock(text="Relevant document 1", metadata={"source": "test1"}), 0.9),
        (MagicMock(text="Relevant document 2", metadata={"source": "test2"}), 0.8)
    ]
    
    mock_embedding_client.return_value.get_embedding.return_value = mock_embedding
    mock_vector_store.return_value.similarity_search.return_value = mock_docs
    mock_llm_client.return_value.chat_complete.return_value = "Test response"
    
    # Initialize agent and generate response
    agent = WindsurfAgent(mock_config)
    response = agent.generate_response("test query")
    
    # Assertions
    assert response == "Test response"
    mock_llm_client.return_value.chat_complete.assert_called_once()
    
    # Check that the prompt includes the context from the knowledge base
    call_args = mock_llm_client.return_value.chat_complete.call_args[0][0]
    assert any("Relevant document 1" in msg["content"] for msg in call_args)
    assert any("test query" in msg["content"] for msg in call_args)

@patch('windsurf_agent.agent.WindsurfEmbeddingClient')
@patch('windsurf_agent.agent.WindsurfLLMClient')
@patch('windsurf_agent.agent.SimpleVectorStore')
def test_cleanup(mock_vector_store, mock_llm_client, mock_embedding_client, mock_config):
    """Test cleanup of resources."""
    # Initialize agent and call cleanup
    agent = WindsurfAgent(mock_config)
    agent.cleanup()
    
    # Assert cleanup methods were called
    mock_llm_client.return_value.close.assert_called_once()
    mock_embedding_client.return_value.close.assert_called_once()
    mock_vector_store.return_value.save.assert_called_once()

@patch('windsurf_agent.agent.WindsurfEmbeddingClient')
@patch('windsurf_agent.agent.WindsurfLLMClient')
@patch('windsurf_agent.agent.SimpleVectorStore')
def test_context_manager(mock_vector_store, mock_llm_client, mock_embedding_client, mock_config):
    """Test agent as a context manager."""
    with WindsurfAgent(mock_config) as agent:
        # Test that agent is properly initialized
        assert agent is not None
        assert agent.llm_client is not None
        assert agent.embedding_client is not None
    
    # Assert cleanup methods were called when exiting the context
    mock_llm_client.return_value.close.assert_called_once()
    mock_embedding_client.return_value.close.assert_called_once()
    mock_vector_store.return_value.save.assert_called_once()
