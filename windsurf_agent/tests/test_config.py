"""Tests for the configuration module."""
import os
from unittest.mock import patch

import pytest

from windsurf_agent.config import Config, EmbeddingConfig, LLMConfig, VectorStoreConfig

def test_embedding_config_validation():
    """Test embedding config validation."""
    # Test valid config
    config = EmbeddingConfig(
        base_url="https://api.windsurf.ai/embeddings",
        api_key="test-key",
        model="test-model"
    )
    assert config.model == "test-model"
    assert config.timeout == 30  # Default value

    # Test invalid URL
    with pytest.raises(ValueError):
        EmbeddingConfig(
            base_url="not-a-url",
            api_key="test-key",
            model="test-model"
        )

def test_llm_config_validation():
    """Test LLM config validation."""
    # Test valid config
    config = LLMConfig(
        base_url="https://api.windsurf.ai/completions",
        api_key="test-key",
        model="test-model"
    )
    assert config.temperature == 0.7  # Default value
    assert config.max_tokens == 2048  # Default value

    # Test temperature bounds
    with pytest.raises(ValueError):
        LLMConfig(
            base_url="https://api.windsurf.ai/completions",
            api_key="test-key",
            model="test-model",
            temperature=2.0  # Invalid
        )

def test_vector_store_config_validation():
    """Test vector store config validation."""
    # Test valid config
    config = VectorStoreConfig(
        persist_dir="/tmp/test",
        collection_name="test-collection",
        similarity_metric="cosine"
    )
    assert config.similarity_metric == "cosine"

    # Test invalid similarity metric
    with pytest.raises(ValueError):
        VectorStoreConfig(
            persist_dir="/tmp/test",
            collection_name="test-collection",
            similarity_metric="invalid"
        )

def test_config_from_env(monkeypatch):
    """Test loading config from environment variables."""
    env_vars = {
        "WINDSURF_EMBEDDING_URL": "https://api.windsurf.ai/embeddings",
        "WINDSURF_LLM_URL": "https://api.windsurf.ai/completions",
        "WINDSURF_API_KEY": "test-api-key",
        "WINDSURF_EMBEDDING_MODEL": "test-embedding-model",
        "WINDSURF_LLM_MODEL": "test-llm-model",
        "VECTOR_STORE_DIR": "/tmp/vector-store",
        "VECTOR_STORE_COLLECTION": "test-collection",
        "LOG_LEVEL": "DEBUG"
    }
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    
    config = Config.from_env()
    
    assert config.embedding.base_url == "https://api.windsurf.ai/embeddings"
    assert config.llm.base_url == "https://api.windsurf.ai/completions"
    assert config.vector_store.persist_dir == "/tmp/vector-store"
    assert config.log_level == "DEBUG"

def test_config_to_dict():
    """Test converting config to dictionary."""
    config = Config(
        embedding=EmbeddingConfig(
            base_url="https://api.windsurf.ai/embeddings",
            api_key="test-key",
            model="test-embedding-model"
        ),
        llm=LLMConfig(
            base_url="https://api.windsurf.ai/completions",
            api_key="test-key",
            model="test-llm-model"
        ),
        vector_store=VectorStoreConfig(
            persist_dir="/tmp/test",
            collection_name="test-collection"
        )
    )
    
    config_dict = config.dict()
    assert "embedding" in config_dict
    assert "llm" in config_dict
    assert "vector_store" in config_dict
    assert config_dict["embedding"]["model"] == "test-embedding-model"
    assert config_dict["llm"]["model"] == "test-llm-model"
    assert config_dict["vector_store"]["collection_name"] == "test-collection"
