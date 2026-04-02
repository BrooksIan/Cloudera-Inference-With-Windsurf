# windsurf_agent/embedding_client.py
import logging
import json
from typing import List, Optional
import numpy as np
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryCallState,
    before_sleep_log
)

from .config import EmbeddingConfig
from .exceptions import EmbeddingError, APIError, RateLimitError
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class WindsurfEmbeddingClient:
    """Client for interacting with Windsurf's embedding API."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        })

    # In embedding_client.py, find the _make_request method and update it:
    def _make_request(self, endpoint: str, payload: dict) -> Any:
        """Make a request to the API with retries and error handling."""
        # Use the same URL construction as LLM client
        base_url = self.config.base_url.strip()
        if not base_url:
            raise ValueError("Base URL is not configured")
            
        # Ensure base_url has a scheme
        if not base_url.startswith(('http://', 'https://')):
            base_url = f"https://{base_url}"
            
        # Remove any trailing slashes from base_url and leading slashes from endpoint
        base_url = base_url.rstrip('/')
        endpoint = endpoint.lstrip('/')
        
        # Construct the full URL
        url = f"{base_url}/{endpoint}" if endpoint else base_url
        
        logger.info(f"Making request to: {url}")
        logger.info(f"Request payload: {payload}")
        
        @retry(
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((requests.exceptions.RequestException, APIError)),
            before_sleep=before_sleep_log(logger, logging.WARNING)
        )
        def _request_with_retry():
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    timeout=self.config.timeout
                )
                
                if response.status_code == 429:
                    raise RateLimitError("Rate limit exceeded")
                
                response.raise_for_status()
                return response.json()
            
            except requests.exceptions.RequestException as e:
                raise APIError(f"API request failed: {str(e)}") from e

        return _request_with_retry()

    def get_embedding(self, text: str, input_type: str = None) -> List[float]:
        """Get embedding for a single text.
        
        Args:
            text: The text to get an embedding for
            input_type: Type of input ('query' or 'passage'). If None, will be determined automatically
                      based on the endpoint type.
            
        Returns:
            List[float]: The embedding vector
            
        Raises:
            ValueError: If the input text is empty
            APIError: If there's an error from the API
            EmbeddingError: For other errors during embedding retrieval
        """
        if not text.strip():
            raise ValueError("Text cannot be empty")
            
        try:
            # Get embeddings for the single text
            embeddings = self.get_embeddings([text], input_type=input_type)
            return embeddings[0] if embeddings else []
        except Exception as e:
            raise EmbeddingError(f"Failed to get embedding: {str(e)}") from e

    def get_embeddings(self, texts: List[str], input_type: str = None) -> List[List[float]]:
        """Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to get embeddings for
            input_type: Type of input, either 'passage' or 'query'
            
        Returns:
            List of embedding vectors, one for each input text
        """
        try:
            is_cloudera_ml = 'endpoints' in self.config.base_url.lower()
            
            # Create payload with model and input
            # Select appropriate model based on input_type for Cloudera ML
            if is_cloudera_ml and input_type:
                if input_type == 'query':
                    model = self.config.query_model
                elif input_type == 'passage':
                    model = self.config.passage_model
                else:
                    model = self.config.model
            else:
                model = self.config.model
                
            payload = {
                "input": texts,
                "model": model
            }
            
            # Add input_type if specified
            if input_type:
                payload["input_type"] = input_type
            
            # For Cloudera ML, we don't need to append anything to the URL
            endpoint = "" if is_cloudera_ml else "embeddings"
            
            # The endpoint is empty for Cloudera ML as we handle the full URL in _make_request
            endpoint = "" if is_cloudera_ml else "embeddings"
            response = self._make_request(endpoint, payload)
            
            # Handle different response formats
            if isinstance(response, dict):
                # Case 1: Response has 'data' field with list of embeddings
                if "data" in response and isinstance(response["data"], list):
                    # If the data items have 'embedding' field
                    if response["data"] and isinstance(response["data"][0], dict) and "embedding" in response["data"][0]:
                        return [item["embedding"] for item in response["data"]]
                    # If the data items are the embeddings themselves
                    elif response["data"] and isinstance(response["data"][0], list):
                        return response["data"]
                # Case 2: Response has 'embeddings' field with list of embeddings
                elif "embeddings" in response and isinstance(response["embeddings"], list):
                    return response["embeddings"]
                # Case 3: Response is the embedding dictionary itself
                elif "embedding" in response and isinstance(response["embedding"], list):
                    return [response["embedding"]]
            # Case 4: Response is a list of embeddings
            elif isinstance(response, list):
                # If it's a list of lists, assume they're embeddings
                if all(isinstance(item, list) for item in response):
                    return response
                # If it's a list of dicts with 'embedding' field
                elif all(isinstance(item, dict) and "embedding" in item for item in response):
                    return [item["embedding"] for item in response]
                
            logger.error(f"Unexpected response format: {response}")
            raise EmbeddingError(f"Unexpected response format from API: {response}")
            
        except Exception as e:
            logger.error(f"Failed to get embeddings: {str(e)}")
            if not isinstance(e, (APIError, EmbeddingError)):
                raise EmbeddingError(f"Failed to get embeddings: {str(e)}") from e
            raise

    def get_embedding_np(self, text: str) -> np.ndarray:
        """Get embedding as a NumPy array."""
        return np.array(self.get_embedding(text))

    def get_embeddings_np(self, texts: List[str]) -> np.ndarray:
        """Get embeddings as a NumPy array."""
        return np.array(self.get_embeddings(texts))

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'session'):
            self.session.close()