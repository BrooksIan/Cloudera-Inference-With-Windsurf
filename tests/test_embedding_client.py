import os
import pytest
import numpy as np
import responses
from windsurf_agent.embedding_client import WindsurfEmbeddingClient
from windsurf_agent.config import EmbeddingConfig
from windsurf_agent.exceptions import EmbeddingError, APIError

@pytest.fixture(scope="module")
def embedding_config():
    config = EmbeddingConfig(
        base_url=os.getenv("WINDSURF_EMBEDDING_BASE_URL", ""),
        api_key=os.getenv("WINDSURF_EMBEDDING_API_KEY", "test_api_key"),
        model=os.getenv("WINDSURF_EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5"),
        query_model=os.getenv("WINDSURF_EMBEDDING_QUERY_MODEL", "nvidia/nv-embedqa-e5-v5-query"),
        passage_model=os.getenv("WINDSURF_EMBEDDING_PASSAGE_MODEL", "nvidia/nv-embedqa-e5-v5-passage"),
        timeout=int(os.getenv("WINDSURF_EMBEDDING_TIMEOUT", "30")),
        max_retries=int(os.getenv("WINDSURF_EMBEDDING_MAX_RETRIES", "3"))
    )
    
    print(f"\n=== Embedding Configuration Validation ===")
    print(f"Embedding Base URL: {config.base_url}")
    print(f"Embedding Model: {config.model}")
    print(f"Query Model: {config.query_model}")
    print(f"Passage Model: {config.passage_model}")
    print(f"========================================\n")
    
    return config

@pytest.fixture
def mock_embedding_response():
    return {
        "data": [{
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
            "index": 0
        }]
    }

def test_get_embedding_success(embedding_config, mock_embedding_response):
    with responses.RequestsMock() as rsps:
        # Use the base URL from config to construct the endpoint
        base_url = embedding_config.base_url.rstrip('/')
        if 'endpoints' in base_url.lower():
            # For Cloudera ML endpoints
            url = base_url
            if not url.endswith('/v1'):
                url = f"{url}/v1"
            url = f"{url}/embeddings"
        else:
            # For standard endpoints
            url = f"{base_url}/embeddings"
        
        rsps.add(
            responses.POST,
            url,
            json=mock_embedding_response,
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
    
    # Use the base URL from config to construct the endpoint
    base_url = embedding_config.base_url.rstrip('/')
    if 'endpoints' in base_url.lower():
        # For Cloudera ML endpoints
        url = base_url
        if not url.endswith('/v1'):
            url = f"{url}/v1"
        url = f"{url}/embeddings"
    else:
        # For standard endpoints
        url = f"{base_url}/embeddings"
    
    responses.add(
        responses.POST,
        url,
        json=mock_response,
        status=200
    )
    
    client = WindsurfEmbeddingClient(embedding_config)
    embeddings = client.get_embeddings(["test text 1", "test text 2"])
    
    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert embeddings[1] == [0.4, 0.5, 0.6]

def test_get_embedding_api_error(embedding_config):
    with responses.RequestsMock() as rsps:
        # Use the base URL from config to construct the endpoint
        base_url = embedding_config.base_url.rstrip('/')
        if 'endpoints' in base_url.lower():
            # For Cloudera ML endpoints
            url = base_url
            if not url.endswith('/v1'):
                url = f"{url}/v1"
            url = f"{url}/embeddings"
        else:
            # For standard endpoints
            url = f"{base_url}/embeddings"
        
        rsps.add(
            responses.POST,
            url,
            json={"error": "Invalid request"},
            status=400
        )
        
        client = WindsurfEmbeddingClient(embedding_config)
        with pytest.raises(EmbeddingError):
            client.get_embedding("test text")