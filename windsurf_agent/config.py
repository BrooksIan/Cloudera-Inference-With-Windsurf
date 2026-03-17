import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    base_url: str
    api_key: str
    model: str = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout: int = 30
    max_retries: int = 3
    
    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Create config from environment variables."""
        return cls(
            base_url=os.getenv("WINDSURF_LLM_BASE_URL", ""),
            api_key=os.getenv("WINDSURF_LLM_API_KEY", ""),
            model=os.getenv("WINDSURF_LLM_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1.5"),
            temperature=float(os.getenv("WINDSURF_LLM_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("WINDSURF_LLM_MAX_TOKENS", "1024")),
            timeout=int(os.getenv("WINDSURF_LLM_TIMEOUT", "30")),
            max_retries=int(os.getenv("WINDSURF_LLM_MAX_RETRIES", "3"))
        )

@dataclass
class EmbeddingConfig:
    """Configuration for Embedding client."""
    base_url: str
    api_key: str
    model: str = "nvidia/nv-embedqa-e5-v5"
    query_model: str = "nvidia/nv-embedqa-e5-v5-query"
    passage_model: str = "nvidia/nv-embedqa-e5-v5-passage"
    max_retries: int = 3
    timeout: int = 30
    
    @classmethod
    def from_env(cls) -> 'EmbeddingConfig':
        """Create config from environment variables."""
        return cls(
            base_url=os.getenv("WINDSURF_EMBEDDING_BASE_URL", ""),
            api_key=os.getenv("WINDSURF_EMBEDDING_API_KEY", ""),
            model=os.getenv("WINDSURF_EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5"),
            query_model=os.getenv("WINDSURF_EMBEDDING_QUERY_MODEL", "nvidia/nv-embedqa-e5-v5-query"),
            passage_model=os.getenv("WINDSURF_EMBEDDING_PASSAGE_MODEL", "nvidia/nv-embedqa-e5-v5-passage"),
            max_retries=int(os.getenv("WINDSURF_EMBEDDING_MAX_RETRIES", "3")),
            timeout=int(os.getenv("WINDSURF_EMBEDDING_TIMEOUT", "30"))
        )

@dataclass
class VectorStoreConfig:
    """Configuration for Vector Store."""
    dimension: int = 1024
    embedding_dimension: int = 1024
    index_type: str = "faiss"
    similarity_metric: str = "cosine"
    collection_name: str = "default"
    persist_dir: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'VectorStoreConfig':
        """Create config from environment variables."""
        return cls(
            dimension=int(os.getenv("WINDSURF_EMBEDDING_DIMENSION", "1024")),
            embedding_dimension=int(os.getenv("WINDSURF_EMBEDDING_DIMENSION", "1024")),
            index_type=os.getenv("WINDSURF_VECTOR_INDEX_TYPE", "faiss"),
            similarity_metric=os.getenv("WINDSURF_VECTOR_SIMILARITY_METRIC", "cosine"),
            collection_name=os.getenv("WINDSURF_VECTOR_COLLECTION_NAME", "default"),
            persist_dir=os.getenv("WINDSURF_VECTOR_PERSIST_DIR")
        )

@dataclass
class Config:
    """Main configuration class containing all sub-configurations."""
    llm: LLMConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create config from environment variables."""
        return cls(
            llm=LLMConfig.from_env(),
            embedding=EmbeddingConfig.from_env(),
            vector_store=VectorStoreConfig.from_env()
        )
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """Create config from dictionary."""
        return cls(
            llm=LLMConfig(**config_dict.get("llm", {})),
            embedding=EmbeddingConfig(**config_dict.get("embedding", {})),
            vector_store=VectorStoreConfig(**config_dict.get("vector_store", {}))
        )
