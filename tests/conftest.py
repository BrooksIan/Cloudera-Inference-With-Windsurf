"""Pytest configuration and fixtures for Windsurf Agent tests."""
import os
import pytest
from unittest.mock import Mock, patch
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

from windsurf_agent.config import Config, EmbeddingConfig, LLMConfig, VectorStoreConfig

@pytest.fixture(scope="session")
def mock_config():
    """Create a mock configuration for testing with environment variables."""
    return Config(
        embedding=EmbeddingConfig(
            base_url=os.getenv("WINDSURF_EMBEDDING_BASE_URL", "https://test-embedding.windsurf.ai/v1"),
            api_key=os.getenv("WINDSURF_EMBEDDING_API_KEY", "test-api-key"),
            model=os.getenv("WINDSURF_EMBEDDING_MODEL", "test-embedding-model"),
            timeout=int(os.getenv("WINDSURF_EMBEDDING_TIMEOUT", "30")),
            max_retries=int(os.getenv("WINDSURF_EMBEDDING_MAX_RETRIES", "3"))
        ),
        llm=LLMConfig(
            base_url=os.getenv("WINDSURF_LLM_BASE_URL", "https://test-llm.windsurf.ai/v1"),
            api_key=os.getenv("WINDSURF_LLM_API_KEY", "test-api-key"),
            model=os.getenv("WINDSURF_LLM_MODEL", "test-llm-model"),
            temperature=float(os.getenv("WINDSURF_LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("WINDSURF_LLM_MAX_TOKENS", "2048")),
            timeout=int(os.getenv("WINDSURF_LLM_TIMEOUT", "60")),
            max_retries=int(os.getenv("WINDSURF_LLM_MAX_RETRIES", "3"))
        ),
        vector_store=VectorStoreConfig(
            persist_dir=os.getenv("WINDSURF_VECTOR_STORE_DIR"),
            collection_name=os.getenv("WINDSURF_VECTOR_STORE_COLLECTION", "test-collection"),
            similarity_metric=os.getenv("WINDSURF_VECTOR_STORE_SIMILARITY", "cosine"),
            dimension=int(os.getenv("WINDSURF_VECTOR_STORE_DIMENSION", "1536"))
        )
    )

@pytest.fixture
def sample_embedding():
    """Return a sample embedding vector for testing."""
    return np.random.rand(1536).astype(np.float32)

@pytest.fixture
def sample_document():
    """Return a sample document for testing."""
    return {
        "id": "doc-123",
        "text": "This is a test document.",
        "embedding": np.random.rand(1536).astype(np.float32).tolist(),
        "metadata": {"source": "test"}
    }

@pytest.fixture
def mock_embedding_response():
    """Return a mock embedding API response."""
    return {
        "data": [
            {
                "embedding": [0.1] * 1536,
                "index": 0,
                "object": "embedding"
            }
        ],
        "model": "test-embedding-model",
        "object": "list"
    }

@pytest.fixture
def mock_llm_response():
    """Return a mock LLM completion response."""
    return {
        "id": "cmpl-123",
        "object": "text_completion",
        "created": 1677652288,
        "model": "test-llm-model",
        "choices": [
            {
                "text": "This is a test completion.",
                "index": 0,
                "logprobs": None,
                "finish_reason": "length"
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 10,
            "total_tokens": 15
        }
    }
