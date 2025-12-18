import os
import pytest
from unittest.mock import Mock, patch
import numpy as np
from windsurf_agent.agent import WindsurfAgent
from windsurf_agent.config import Config
from windsurf_agent.vector_store import Document

@pytest.fixture
def mock_config():
    return Config.from_dict({
        "embedding": {
            "base_url": os.getenv("WINDSURF_EMBEDDING_BASE_URL", "https://test-embedding.windsurf.ai/v1"),
            "api_key": os.getenv("WINDSURF_EMBEDDING_API_KEY", "test_api_key"),
            "model": os.getenv("WINDSURF_EMBEDDING_MODEL", "test-embedding-model"),
            "timeout": int(os.getenv("WINDSURF_EMBEDDING_TIMEOUT", "30")),
            "max_retries": int(os.getenv("WINDSURF_EMBEDDING_MAX_RETRIES", "3"))
        },
        "llm": {
            "base_url": os.getenv("WINDSURF_LLM_BASE_URL", "https://test-llm.windsurf.ai/v1"),
            "api_key": os.getenv("WINDSURF_LLM_API_KEY", "test_api_key"),
            "model": "test-llm-model",
            "temperature": 0.7,
            "max_tokens": 100,
            "timeout": 30,
            "max_retries": 3
        },
        "vector_store": {
            "persist_dir": None,
            "collection_name": "test_collection",
            "similarity_metric": "cosine"
        },
        "log_level": "INFO"
    })

def test_agent_initialization(mock_config):
    with patch('windsurf_agent.agent.WindsurfEmbeddingClient') as mock_embedding, \
         patch('windsurf_agent.agent.WindsurfLLMClient') as mock_llm, \
         patch('windsurf_agent.agent.SimpleVectorStore') as mock_store:
        
        agent = WindsurfAgent(config=mock_config)
        
        # Verify clients were initialized
        mock_embedding.assert_called_once()
        mock_llm.assert_called_once()
        mock_store.assert_called_once()

def test_rag_query(mock_config):
    with patch('windsurf_agent.agent.WindsurfEmbeddingClient') as mock_embedding, \
         patch('windsurf_agent.agent.WindsurfLLMClient') as mock_llm, \
         patch('windsurf_agent.agent.SimpleVectorStore') as mock_store:
        
        # Setup mock embedding client
        mock_embedding.return_value.get_embedding.return_value = [0.1, 0.2, 0.3]
        
        # Setup mock vector store
        mock_doc = Document(
            id="1",
            text="Test document",
            embedding=np.array([0.1, 0.2, 0.3]),
            metadata={"source": "test"}
        )
        mock_store.return_value.similarity_search.return_value = [(mock_doc, 0.9)]
        
        # Setup mock LLM client
        mock_llm.return_value.complete.return_value = "Test answer"
        
        # Initialize agent
        agent = WindsurfAgent(config=mock_config)
        
        # Test RAG query
        result = agent.rag_query("test query")
        
        # Verify results
        assert result["answer"] == "Test answer"
        assert len(result["sources"]) == 1
        assert result["sources"][0]["text"] == "Test document"