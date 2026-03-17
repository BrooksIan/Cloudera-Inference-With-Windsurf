from openai import OpenAI
import os
from pathlib import Path
from typing import Dict, List, Optional, Generator, Union
import logging
from .config import LLMConfig
from .exceptions import APIError, AuthenticationError, RateLimitError

logger = logging.getLogger(__name__)

class ClouderaLLMClient:
    """Client for interacting with Cloudera ML LLM endpoints."""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize the client with configuration.
        
        Args:
            config: Optional LLMConfig instance. If not provided, loads from environment.
        """
        self.config = config or LLMConfig.from_env()
        self.client = self._initialize_client()
        
    def _initialize_client(self):
        """Initialize the OpenAI client with proper configuration."""
        return OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout
        )
        
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Generate a streaming chat completion."""
        try:
            response = self.client.chat.completions.create(
                model=model or self.config.model,
                messages=messages,
                stream=True,
                **self._get_request_params(**kwargs)
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            raise APIError(f"Chat completion failed: {str(e)}") from e
    
    def complete(
        self,
        prompt: str,
        system_message: str = "You are a helpful AI assistant.",
        **kwargs
    ) -> Generator[str, None, None]:
        """Generate a completion for a single prompt."""
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        return self.chat(messages, **kwargs)
        
    def _get_request_params(self, **overrides) -> dict:
        """Get default request parameters with overrides."""
        params = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": 0.7,
        }
        params.update(overrides)
        return params
