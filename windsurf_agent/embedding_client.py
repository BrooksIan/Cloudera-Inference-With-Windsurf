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
        
        # Set headers - include Cloudera-specific headers if needed
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        
        # Add Cloudera-specific headers for ML endpoints
        if 'endpoints' in config.base_url.lower():
            headers.update({
                "X-Requested-By": "cascade-agent",
                "X-XSRF-Header": "true"
            })
            
        self.session.headers.update(headers)

    # In embedding_client.py, find the _make_request method and update it:
    def _make_request(self, endpoint: str, payload: dict) -> Any:
        """Make a request to the API with retries and error handling."""
        # For Cloudera ML endpoints, use the base URL as-is and include model in the payload
        is_cloudera_ml = 'endpoints' in self.config.base_url.lower()
        logger.info(f"Is Cloudera ML endpoint: {is_cloudera_ml}")
        logger.info(f"Base URL: {self.config.base_url}")
        
        if is_cloudera_ml:
            # For Cloudera ML, append /embeddings to the base URL
            url = f"{self.config.base_url.rstrip('/')}/embeddings"
            logger.info(f"Using Cloudera ML URL: {url}")
            
            # Ensure the payload has the required input_type for Cloudera ML
            if 'input_type' not in payload:
                payload['input_type'] = 'passage'  # Default to 'passage' if not specified
        else:
            # For standard endpoints, append the endpoint to the base URL
            url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
            logger.info(f"Using standard endpoint URL: {url}")
        
        logger.info(f"Using URL: {url}")
        logger.info(f"Full request payload: {json.dumps(payload, indent=2)}")
        
        logger.info(f"Making request to: {url}")
        logger.info(f"Request headers: {self.session.headers}")
        logger.info(f"Request payload: {json.dumps(payload, indent=2)}")
        
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
                logger.info(f"Response status: {response.status_code}")
                logger.info(f"Response headers: {dict(response.headers)}")
                logger.info(f"Response content: {response.text[:500]}...")
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                error_msg = f"{getattr(e.response, 'status_code', 'No status')} {getattr(e.response, 'reason', 'No reason')}"
                if hasattr(e.response, 'text'):
                    error_msg += f" - {e.response.text}"
                logger.error(f"API request failed: {error_msg}")
                raise APIError(f"API request failed: {error_msg}") from e

        try:
            return _request_with_retry()
        except Exception as e:
            logger.error(f"Request failed after retries: {str(e)}")
            raise

    def _before_retry(self, retry_state: RetryCallState) -> None:
        """Log before retrying a failed request."""
        if retry_state.outcome and retry_state.outcome.exception():
            logger.warning(
                f"Retrying after {retry_state.attempt_number} attempts: "
                f"{str(retry_state.outcome.exception())}"
            )

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