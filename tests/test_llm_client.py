import os
import pytest
import json
import responses
from windsurf_agent.llm_client import WindsurfLLMClient
from windsurf_agent.config import LLMConfig
from windsurf_agent.exceptions import LLMError

@pytest.fixture
def llm_config():
    config = LLMConfig(
        base_url=os.getenv("WINDSURF_LLM_BASE_URL", ""),
        api_key=os.getenv("WINDSURF_LLM_API_KEY", "test_api_key"),
        model=os.getenv("WINDSURF_LLM_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1.5"),
        temperature=0.7,
        max_tokens=100,
        timeout=30,
        max_retries=3
    )
    
    print(f"\n=== LLM Configuration Validation ===")
    print(f"LLM Base URL: {config.base_url}")
    print(f"LLM Model: {config.model}")
    print(f"Temperature: {config.temperature}")
    print(f"Max Tokens: {config.max_tokens}")
    print(f"===================================\n")
    
    return config

@pytest.fixture
def mock_completion_response():
    return {
        "choices": [{
            "text": "This is a test completion.",
            "finish_reason": "stop",
            "index": 0
        }]
    }

def test_complete_success(llm_config, mock_completion_response):
    with responses.RequestsMock() as rsps:
        # Use the actual base URL from config
        completions_url = f"{llm_config.base_url.rstrip('/')}/completions"
        rsps.add(
            responses.POST,
            completions_url,
            json=mock_completion_response,
            status=200
        )
        
        client = WindsurfLLMClient(llm_config)
        response = client.complete("test prompt")
        
        assert response == "This is a test completion."

def test_chat_success(llm_config):
    mock_response = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "This is a test chat response."
            }
        }]
    }
    
    with responses.RequestsMock() as rsps:
        # Use the actual base URL from config
        chat_url = f"{llm_config.base_url.rstrip('/')}/chat/completions"
        rsps.add(
            responses.POST,
            chat_url,
            json=mock_response,
            status=200
        )
        
        client = WindsurfLLMClient(llm_config)
        messages = [{"role": "user", "content": "Hello"}]
        response = client.chat(messages)
        
        assert response == "This is a test chat response."