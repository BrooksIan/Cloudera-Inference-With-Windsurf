import json
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryCallState
)

from windsurf_agent.config import LLMConfig
from windsurf_agent.exceptions import LLMError, APIError, RateLimitError

logger = logging.getLogger(__name__)

class KnoxLLMClient:
    """Client for interacting with Knox-protected LLM API."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "X-Requested-By": "windsurf-agent"  # Required by Knox
        })
        
        # Add additional Knox-specific headers if needed
        self.session.headers.update(self._get_knox_headers())
    
    def _get_knox_headers(self) -> Dict[str, str]:
        """Get Knox-specific headers."""
        return {
            "X-Requested-By": "windsurf-agent",
            "X-XSRF-Header": "true"
        }

    def _make_request(self, endpoint: str, payload: dict, stream: bool = False) -> Any:
        """Make a request to the API with retries and error handling."""
        # Ensure base_url is properly formatted
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
        
        # Log the URL being called (without sensitive query parameters)
        logger.debug(f"Making request to: {url}")
        logger.debug(f"Request headers: {self.session.headers}")
        
        @retry(
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((requests.exceptions.RequestException, APIError)),
            before_sleep=self._before_retry
        )
        def _request_with_retry():
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    stream=stream,
                    timeout=self.config.timeout
                )
                
                # Log response status and headers for debugging
                logger.debug(f"Response status: {response.status_code}")
                logger.debug(f"Response headers: {response.headers}")
                
                if response.status_code == 401:
                    logger.error("Authentication failed. Please check your API key and permissions.")
                    logger.error(f"Response body: {response.text}")
                
                if response.status_code == 429:
                    raise RateLimitError("Rate limit exceeded")
                
                response.raise_for_status()
                
                if stream:
                    return self._handle_streaming_response(response)
                return response.json()
            
            except requests.exceptions.RequestException as e:
                if hasattr(e, 'response') and e.response is not None:
                    logger.error(f"Request failed with status {e.response.status_code}: {e.response.text}")
                raise APIError(f"API request failed: {str(e)}") from e

        return _request_with_retry()

    def _handle_streaming_response(self, response) -> AsyncGenerator[str, None]:
        """Handle streaming response from the API."""
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    line = line[6:]  # Remove 'data: ' prefix
                    if line == '[DONE]':
                        break
                    try:
                        data = json.loads(line)
                        yield data
                    except json.JSONDecodeError:
                        continue

    def _before_retry(self, retry_state: RetryCallState) -> None:
        """Log before retrying a failed request."""
        logger.warning(
            f"Retrying {retry_state.fn.__name__} after {retry_state.attempt_number} "
            f"attempts: {str(retry_state.outcome.exception())}"
        )

    def complete(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate a completion for the given prompt."""
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        try:
            response = self._make_request(
                endpoint="/completions",
                payload={
                    "model": self.config.model,
                    "prompt": prompt,
                    "temperature": temperature if temperature is not None else self.config.temperature,
                    "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
                    **kwargs
                }
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            raise LLMError(f"Failed to generate completion: {str(e)}") from e

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncGenerator[Dict[str, Any], None]]:
        """Generate a chat completion."""
        if not messages:
            raise ValueError("Messages cannot be empty")

        try:
            response = self._make_request(
                endpoint="/chat/completions",
                payload={
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": temperature if temperature is not None else self.config.temperature,
                    "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
                    "stream": stream,
                    **kwargs
                },
                stream=stream
            )
            
            if stream:
                return response
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise LLMError(f"Failed to generate chat completion: {str(e)}") from e

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'session'):
            self.session.close()
