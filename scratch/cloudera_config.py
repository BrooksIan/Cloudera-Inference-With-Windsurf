import os
from windsurf_agent.config import LLMConfig, Config

class ClouderaLLMConfig(LLMConfig):
    """Configuration that enforces Cloudera-hosted models."""
    
    def __init__(self, **data):
        super().__init__(**data)
        # Validate that the base URL points to a Cloudera endpoint
        if not self._is_valid_cloudera_endpoint():
            raise ValueError("LLM endpoint must be a Cloudera-hosted service")

    def _is_valid_cloudera_endpoint(self) -> bool:
        """Check if the base URL is a valid Cloudera endpoint."""
        # List of allowed Cloudera domains
        allowed_domains = [
            'cloudera.com',
            'cloudera.site',  # For Cloudera Machine Learning deployments
            'cdp.cloudera.com',
            'cloudera-ml.ai',
            'cloudera-ml.cloud'  # Add other Cloudera domains as needed
        ]
        
        try:
            from urllib.parse import urlparse
            domain = urlparse(self.base_url).netloc.lower()
            return any(allowed in domain for allowed in allowed_domains)
        except Exception:
            return False

class ClouderaConfig(Config):
    """Configuration that enforces Cloudera-hosted services."""
    
    @classmethod
    def from_env(cls) -> 'ClouderaConfig':
        """Create configuration from environment variables with Cloudera validation."""
        config = super().from_env()
        
        # Override with Cloudera-specific config
        return cls(
            llm=ClouderaLLMConfig(
                base_url=os.getenv("WINDSURF_LLM_BASE_URL", ""),
                api_key=os.getenv("WINDSURF_LLM_API_KEY", ""),
                model=os.getenv("WINDSURF_LLM_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1"),
                temperature=float(os.getenv("WINDSURF_LLM_TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("WINDSURF_LLM_MAX_TOKENS", "2048")),
                timeout=int(os.getenv("WINDSURF_LLM_TIMEOUT", "60")),
                max_retries=int(os.getenv("WINDSURF_LLM_MAX_RETRIES", "3")),
            ),
            # Keep other configs as is
            embedding=config.embedding,
            vector_store=config.vector_store
        )

def enforce_cloudera_models():
    """Monkey patch the config to ensure only Cloudera models are used."""
    import windsurf_agent.config as config_module
    config_module.Config = ClouderaConfig
    config_module.LLMConfig = ClouderaLLMConfig
    print("Enforced Cloudera-hosted models only policy")