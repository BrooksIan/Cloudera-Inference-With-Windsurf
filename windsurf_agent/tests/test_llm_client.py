"""Tests for the LLM client."""
import json
import pytest
from unittest.mock import patch, MagicMock, call
import requests

from windsurf_agent.llm_client import WindsurfLLMClient
from windsurf_agent.exceptions import LLMError, APIError, RateLimitError

@patch('windsurf_agent.llm_client.requests.Session')
def test_complete_success(mock_session_class, mock_config, mock_llm_response):
    """Test successful text completion."""
    # Setup mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_llm_response
    
    # Configure mock session
    mock_session = MagicMock()
    mock_session.post.return_value = mock_response
    mock_session_class.return_value = mock_session
    
    # Initialize client and make request
    client = WindsurfLLMClient(mock_config.llm)
    result = client.complete("Test prompt")
    
    # Assertions
    assert result == "This is a test completion."
    mock_session.post.assert_called_once()
    
    # Check request payload
    request_args = mock_session.post.call_args[1]
    assert request_args['json']['prompt'] == "Test prompt"
    assert request_args['json']['model'] == mock_config.llm.model
    assert request_args['json']['temperature'] == mock_config.llm.temperature
    assert request_args['json']['max_tokens'] == mock_config.llm.max_tokens

@patch('windsurf_agent.llm_client.requests.Session')
def test_complete_empty_prompt(mock_session_class, mock_config):
    """Test completion with empty prompt raises ValueError."""
    client = WindsurfLLMClient(mock_config.llm)
    with pytest.raises(ValueError):
        client.complete("")

@patch('windsurf_agent.llm_client.requests.Session')
def test_complete_rate_limit(mock_session_class, mock_config):
    """Test handling of rate limiting."""
    # Setup mock to return 429
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_session = MagicMock()
    mock_session.post.return_value = mock_response
    mock_session_class.return_value = mock_session
    
    client = WindsurfLLMClient(mock_config.llm)
    with pytest.raises(RateLimitError):
        client.complete("test prompt")

@patch('windsurf_agent.llm_client.requests.Session')
def test_complete_http_error(mock_session_class, mock_config):
    """Test handling of HTTP errors."""
    # Setup mock to raise HTTP error
    mock_session = MagicMock()
    mock_session.post.side_effect = requests.exceptions.RequestException("Connection error")
    mock_session_class.return_value = mock_session
    
    client = WindsurfLLMClient(mock_config.llm)
    with pytest.raises(APIError):
        client.complete("test prompt")

@patch('windsurf_agent.llm_client.requests.Session')
def test_chat_completion(mock_session_class, mock_config):
    """Test chat completion with messages."""
    # Setup mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "chat-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": mock_config.llm.model,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "This is a test chat completion."
                },
                "finish_reason": "stop",
                "index": 0
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "total_tokens": 20
        }
    }
    
    # Configure mock session
    mock_session = MagicMock()
    mock_session.post.return_value = mock_response
    mock_session_class.return_value = mock_session
    
    # Initialize client and make request
    client = WindsurfLLMClient(mock_config.llm)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    result = client.chat_complete(messages)
    
    # Assertions
    assert result == "This is a test chat completion."
    mock_session.post.assert_called_once()
    
    # Check request payload
    request_args = mock_session.post.call_args[1]
    assert request_args['json']['messages'] == messages
    assert request_args['json']['model'] == mock_config.llm.model

@patch('windsurf_agent.llm_client.requests.Session')
def test_stream_completion(mock_session_class, mock_config):
    """Test streaming completion."""
    # Setup streaming response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_lines.return_value = [
        b'data: {"choices": [{"text": "Hello"}]}',
        b'data: {"choices": [{"text": " world"}]}',
        b'data: [DONE]'
    ]
    
    # Configure mock session
    mock_session = MagicMock()
    mock_session.post.return_value = mock_response
    mock_session_class.return_value = mock_session
    
    # Initialize client and make request
    client = WindsurfLLMClient(mock_config.llm)
    
    # Test with callback
    callback_results = []
    def callback(chunk):
        callback_results.append(chunk)
    
    result = client.complete("Say hello", stream=True, callback=callback)
    
    # Assertions
    assert result == "Hello world"
    assert len(callback_results) == 2
    assert callback_results[0] == "Hello"
    assert callback_results[1] == " world"
