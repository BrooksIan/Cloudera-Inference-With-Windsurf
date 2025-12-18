# windsurf_agent/config.py
from pydantic import BaseModel, Field, validator, HttpUrl, field_validator
from typing import Optional, Dict, Any
import os
from pathlib import Path
from urllib.parse import urljoin

class EmbeddingConfig(BaseModel):
    """Configuration for the embedding client."""
    base_url: str = Field(..., description="Base URL for the embedding API")
    api_key: str = Field(..., description="API key for authentication")
    model: str = Field(..., description="Model to use for embeddings")
    timeout: int = Field(30, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum number of retries for failed requests")

    class Config:
        env_prefix = "WINDSURF_EMBEDDING_"

class LLMConfig(BaseModel):
    """Configuration for the LLM client."""
    base_url: str = Field(..., description="Base URL for the LLM API")
    api_key: str = Field(..., description="API key for authentication")
    model: str = Field(..., description="Model to use for completions")
    temperature: float = Field(0.7, description="Temperature for generation")
    max_tokens: int = Field(2048, description="Maximum number of tokens to generate")
    timeout: int = Field(30, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum number of retries for failed requests")

    class Config:
        env_prefix = "WINDSURF_LLM_"

class VectorStoreConfig(BaseModel):
    """Configuration for the vector store."""
    persist_dir: Optional[Path] = Field(None, description="Directory to persist the vector store")
    collection_name: str = Field("documents", description="Name of the collection in the vector store")
    similarity_metric: str = Field("cosine", description="Similarity metric to use (cosine, euclidean, dotproduct)")
    dimension: int = Field(1024, description="Dimension of the embeddings")

    @field_validator('similarity_metric')
    @classmethod
    def validate_similarity_metric(cls, v):
        if v not in ["cosine", "euclidean", "dotproduct"]:
            raise ValueError("similarity_metric must be one of: cosine, euclidean, dotproduct")
        return v

class Config(BaseModel):
    """Main configuration class."""
    embedding: EmbeddingConfig
    llm: LLMConfig
    vector_store: VectorStoreConfig
    log_level: str = Field("INFO", description="Logging level")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables."""
        return cls(
            embedding=EmbeddingConfig(
                base_url=os.getenv("WINDSURF_EMBEDDING_BASE_URL", ""),
                api_key=os.getenv("WINDSURF_EMBEDDING_API_KEY", ""),
                model=os.getenv("WINDSURF_EMBEDDING_MODEL", "text-embedding-3-large"),
                timeout=int(os.getenv("WINDSURF_EMBEDDING_TIMEOUT", "30")),
                max_retries=int(os.getenv("WINDSURF_EMBEDDING_MAX_RETRIES", "3")),
            ),
            llm=LLMConfig(
                base_url=os.getenv("WINDSURF_LLM_BASE_URL", ""),
                api_key=os.getenv("WINDSURF_LLM_API_KEY", ""),
                model=os.getenv("WINDSURF_LLM_MODEL", "gpt-4"),
                temperature=float(os.getenv("WINDSURF_LLM_TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("WINDSURF_LLM_MAX_TOKENS", "2048")),
                timeout=int(os.getenv("WINDSURF_LLM_TIMEOUT", "60")),
                max_retries=int(os.getenv("WINDSURF_LLM_MAX_RETRIES", "3")),
            ),
            vector_store=VectorStoreConfig(
                persist_dir=os.getenv("VECTOR_STORE_DIR"),
                collection_name=os.getenv("VECTOR_STORE_COLLECTION", "documents"),
                similarity_metric=os.getenv("VECTOR_STORE_SIMILARITY", "cosine"),
                dimension=int(os.getenv("VECTOR_STORE_DIMENSION", "1536")),
            ),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )