import pytest
import json
import responses
from windsurf_agent.llm_client import WindsurfLLMClient
from windsurf_agent.config import LLMConfig
from windsurf_agent.exceptions import LLMError

@pytest.fixture
def llm_config():
    return LLMConfig(
        base_url="https://test-llm.windsurf.ai/v1",
        api_key="test_api_key",
        model="test-llm-model",
        temperature=0.7,
        max_tokens=100,
        timeout=30,
        max_retries=3
    )

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
        rsps.add(
            responses.POST,
            "https://test-llm.windsurf.ai/v1/completions",
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
        rsps.add(
            responses.POST,
            "https://test-llm.windsurf.ai/v1/chat/completions",
            json=mock_response,
            status=200
        )
        
        client = WindsurfLLMClient(llm_config)
        messages = [{"role": "user", "content": "Hello"}]
        response = client.chat(messages)
        
        assert response == "This is a test chat response."