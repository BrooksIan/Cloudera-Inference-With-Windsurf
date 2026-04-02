import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np

from .config import Config
from .ClouderaLLMClient import ClouderaLLMClient
from .vector_store import SimpleVectorStore, Document
from .exceptions import WindsurfError, EmbeddingError, LLMError, VectorStoreError

logger = logging.getLogger(__name__)

class ClouderaOnlyAgent:
    """Cloudera AI-only agent class - NO WINDSURF MODELS ALLOWED"""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the Cloudera-only agent."""
        self.config = config or Config.from_env()
        self._setup_cloudera_clients()
    
    def _setup_cloudera_clients(self):
        """Set up Cloudera clients ONLY."""
        try:
            # ONLY use ClouderaLLMClient - no Windsurf clients
            self.llm_client = ClouderaLLMClient()
            
            # Disable embedding client for now or use Cloudera-only
            self.embedding_client = None  # Disabled to prevent Windsurf usage
            
            # Use simple vector store
            self.vector_store = SimpleVectorStore(self.config.vector_store)
            
            logger.info("ClouderaOnlyAgent initialized successfully - NO WINDSURF MODELS")
        except Exception as e:
            logger.error(f"Failed to initialize ClouderaOnlyAgent: {str(e)}")
            raise
    
    def _validate_cloudera_model(self, model: str) -> None:
        """Validate that ONLY Cloudera models are used."""
        allowed_models = [
            "goes---nemotron-v1-5-49b-throughput",
            "nvidia/llama-3.3-nemotron-super-49b-v1.5",
            "goes---e5-embedding"
        ]
        
        if model not in allowed_models:
            raise LLMError(
                f"Model '{model}' is BLOCKED. Only Cloudera AI models allowed: "
                f"{', '.join(allowed_models)}"
            )
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Cloudera AI ONLY."""
        try:
            # Validate model
            model = kwargs.get('model', self.llm_client.model)
            self._validate_cloudera_model(model)
            
            # Convert to messages format
            messages = [{"role": "user", "content": prompt}]
            
            # Use ONLY Cloudera client
            response = self.llm_client.chat_completion(
                messages=messages,
                model=model,
                **kwargs
            )
            
            return ''.join(response)
        except Exception as e:
            logger.error(f"Cloudera AI generation failed: {str(e)}")
            raise LLMError(f"Cloudera AI generation failed: {str(e)}") from e
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat using Cloudera AI ONLY."""
        try:
            # Validate model
            model = kwargs.get('model', self.llm_client.model)
            self._validate_cloudera_model(model)
            
            # Use ONLY Cloudera client
            response = self.llm_client.chat_completion(
                messages=messages,
                model=model,
                **kwargs
            )
            
            return ''.join(response)
        except Exception as e:
            logger.error(f"Cloudera AI chat failed: {str(e)}")
            raise LLMError(f"Cloudera AI chat failed: {str(e)}") from e
