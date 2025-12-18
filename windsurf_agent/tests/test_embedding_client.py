import pytest
import responses
from unittest.mock import patch
import numpy as np
from windsurf_agent.embedding_client import WindsurfEmbeddingClient
from windsurf_agent.config import EmbeddingConfig
from windsurf_agent.exceptions import EmbeddingError, APIError

@pytest.fixture
def embedding_config():
    return EmbeddingConfig(
        base_url="https://test-embedding.windsurf.ai/v1",
        api_key="test_api_key",
        model="test-embedding-model",
        timeout=5,
        max_retries=3
    )

@responses.activate
def test_get_embedding_success(embedding_config):
    """Test successful embedding retrieval."""
    mock_response = {
        "data": [{
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
            "index": 0
        }]
    }
    
    responses.add(
        responses.POST,
        "https://test-embedding.windsurf.ai/v1/embeddings",
        json=mock_response,
        status=200
    )
    
    client = WindsurfEmbeddingClient(embedding_config)
    embedding = client.get_embedding("test text")
    
    assert len(embedding) == 5
    assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]

@responses.activate
def test_get_embeddings_success(embedding_config):
    """Test successful batch embedding retrieval."""
    mock_response = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3], "index": 0},
            {"embedding": [0.4, 0.5, 0.6], "index": 1}
        ]
    }
    
    responses.add(
        responses.POST,
        "https://test-embedding.windsurf.ai/v1/embeddings",
        json=mock_response,
        status=200
    )
    
    client = WindsurfEmbeddingClient(embedding_config)
    embeddings = client.get_embeddings(["test text 1", "test text 2"])
    
    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert embeddings[1] == [0.4, 0.5, 0.6]

@responses.activate
def test_get_embedding_api_error(embedding_config):
    """Test API error handling."""
    responses.add(
        responses.POST,
        "https://test-embedding.windsurf.ai/v1/embeddings",
        json={"error": "Invalid request"},
        status=400
    )
    
    client = WindsurfEmbeddingClient(embedding_config)
    # Change this to expect EmbeddingError instead of APIError
    with pytest.raises(EmbeddingError):
        client.get_embedding("test text")

