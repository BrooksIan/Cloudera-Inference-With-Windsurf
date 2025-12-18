class WindsurfError(Exception):
    """Base exception for all Windsurf agent errors."""
    pass

class ConfigurationError(WindsurfError):
    """Raised when there is a configuration error."""
    pass

class AuthenticationError(WindsurfError):
    """Raised when authentication fails."""
    pass

class APIError(WindsurfError):
    """Raised when there is an error with the API call."""
    def __init__(self, message: str, status_code: int = None, response: dict = None):
        self.status_code = status_code
        self.response = response
        super().__init__(message)

class RateLimitError(APIError):
    """Raised when rate limits are exceeded."""
    pass

class ValidationError(WindsurfError):
    """Raised when input validation fails."""
    pass

class EmbeddingError(WindsurfError):
    """Raised when there is an error generating embeddings."""
    pass

class LLMError(WindsurfError):
    """Raised when there is an error with the LLM."""
    pass

class VectorStoreError(WindsurfError):
    """Raised when there is an error with the vector store."""
    pass