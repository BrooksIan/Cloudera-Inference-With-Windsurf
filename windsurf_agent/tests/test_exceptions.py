"""Tests for the exceptions module."""
import pytest

from windsurf_agent.exceptions import (
    WindsurfError,
    ConfigurationError,
    AuthenticationError,
    APIError,
    RateLimitError,
    ValidationError,
    EmbeddingError,
    LLMError,
    VectorStoreError
)

def test_windsurf_error():
    """Test base WindsurfError."""
    with pytest.raises(WindsurfError) as exc_info:
        raise WindsurfError("Test error")
    assert str(exc_info.value) == "Test error"

def test_configuration_error():
    """Test ConfigurationError."""
    with pytest.raises(ConfigurationError) as exc_info:
        raise ConfigurationError("Configuration error")
    assert str(exc_info.value) == "Configuration error"
    assert isinstance(exc_info.value, WindsurfError)

def test_authentication_error():
    """Test AuthenticationError."""
    with pytest.raises(AuthenticationError) as exc_info:
        raise AuthenticationError("Auth failed")
    assert str(exc_info.value) == "Auth failed"
    assert isinstance(exc_info.value, WindsurfError)

def test_api_error():
    """Test APIError with status code and response."""
    with pytest.raises(APIError) as exc_info:
        raise APIError("API error", status_code=404, response={"error": "Not found"})
    assert str(exc_info.value) == "API error"
    assert exc_info.value.status_code == 404
    assert exc_info.value.response == {"error": "Not found"}
    assert isinstance(exc_info.value, WindsurfError)

def test_rate_limit_error():
    """Test RateLimitError."""
    with pytest.raises(RateLimitError) as exc_info:
        raise RateLimitError("Rate limit exceeded")
    assert str(exc_info.value) == "Rate limit exceeded"
    assert isinstance(exc_info.value, APIError)

def test_validation_error():
    """Test ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        raise ValidationError("Validation failed")
    assert str(exc_info.value) == "Validation failed"
    assert isinstance(exc_info.value, WindsurfError)

def test_embedding_error():
    """Test EmbeddingError."""
    with pytest.raises(EmbeddingError) as exc_info:
        raise EmbeddingError("Embedding failed")
    assert str(exc_info.value) == "Embedding failed"
    assert isinstance(exc_info.value, WindsurfError)

def test_llm_error():
    """Test LLMError."""
    with pytest.raises(LLMError) as exc_info:
        raise LLMError("LLM failed")
    assert str(exc_info.value) == "LLM failed"
    assert isinstance(exc_info.value, WindsurfError)

def test_vector_store_error():
    """Test VectorStoreError."""
    with pytest.raises(VectorStoreError) as exc_info:
        raise VectorStoreError("Vector store operation failed")
    assert str(exc_info.value) == "Vector store operation failed"
    assert isinstance(exc_info.value, WindsurfError)
