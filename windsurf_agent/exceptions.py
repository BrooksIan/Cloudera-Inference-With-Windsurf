class WindsurfError(Exception):
    """Base exception for all Windsurf-related errors."""
    pass

class APIError(Exception):
    """Base exception for API errors."""
    pass

class LLMError(APIError):
    """Base exception for LLM-related errors."""
    pass

class EmbeddingError(APIError):
    """Base exception for embedding-related errors."""
    pass

class VectorStoreError(APIError):
    """Base exception for vector store-related errors."""
    pass

class AuthenticationError(APIError):
    """Raised when authentication fails."""
    pass

class RateLimitError(APIError):
    """Raised when rate limit is exceeded."""
    pass

class ValidationError(APIError):
    """Raised when input validation fails."""
    pass
