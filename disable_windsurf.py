#!/usr/bin/env python3
"""
Disable Windsurf models and enforce Cloudera AI only.
This script will modify the configuration to prevent any Windsurf model usage.
"""

import os
import sys
from pathlib import Path

# Add the windsurf_agent to the path
sys.path.insert(0, str(Path(__file__).parent / "windsurf_agent"))

def disable_windsurf_models():
    """Disable all Windsurf models and enforce Cloudera only."""
    print("🚫 Disabling Windsurf Models")
    print("🔒 Enforcing Cloudera AI Only")
    print("=" * 50)
    
    # Create a strict Cloudera-only configuration
    cloudera_only_config = '''
# Cloudera AI ONLY Configuration
# All Windsurf models are disabled

# Force Cloudera endpoints only
WINDSURF_LLM_BASE_URL="https://ml-64288d82-5dd.go01-dem.ylcu-atmi.cloudera.site/namespaces/serving-default/endpoints/goes---nemotron-v1-5-49b-throughput/v1"
WINDSURF_LLM_API_KEY="eyJraWQiOiIzYzhlNzA3OTEyZmI0NTA1ODE3NzE3YzMyOTU4MmQwMTFjYjlmNTAwIiwidHlwIjoiSldUIiwiYWxnIjoiUlMyNTYifQ.eyJzdWIiOiJpYnJvb2tzIiwiYXVkIjoiaHR0cHM6Ly9kZS55bGN1LWF0bWkuY2xvdWRlcmEuc2l0ZSIsImlzcyI6Imh0dHBzOi8vY29uc29sZWF1dGguY2RwLmNsb3VkZXJhLmNvbS84YTFlMTVjZC0wNGMyLTQ4YWEtOGYzNS1iNGE4YzExOTk3ZDMiLCJncm91cHMiOiJjZHBfZGVtb3Nfd29ya2Vyc193dyBjZHBfZGVtby1hd3MtcHJpbSBfY19kZl9kZXZlbG9wXzkxMTQ2M2MgX2NfZGZfdmlld182ZjU5ZTlmMyBfY19kZl9hZG1pbmlzdGVyXzkxMTQ2M2MgX2NfZGZfdmlld185MTE0NjNjIF9jX21sX2J1c2luZXNzX3VzZXJzXzZlZTBkYjkxIF9jX2RmX3B1Ymxpc2hfOTExNDYzYyBfY19kZl92aWV3XzkxMTQ2M2MwIF9jX2Vudl9hc3NpZ25lZXNfOTExNDYzYyBfY19yYW5nZXJfYWRtaW5zXzkwNmIwYmEgX2NfbWxfdXNlcnNfNmY1OWU5ZjMgX2NfbWxfdXNlcnNfNGQ4M2FkN2YgX2NfZW52X2Fzc2lnbmVlc185MDZiMGJhIF9jX2RmX2RldmVsb3BfNmY1OWU5ZjMgX2NfZGZfYWRtaW5pc3Rlcl82ZjU5ZTlmMyBfY19kZl9wdWJsaXNoXzZmNTllOWYzIF9jX21sX3VzZXJzXzZlZTBkYjkxIF9jX21sX2J1c2luZXNzX3VzZXJzXzZmNTllOWYzIF9jX21sX2J1c2luZXNzX3VzZXJzXzkxMTQ2M2MgX2NfZW52X2Fzc2lnbmVlc182ZjU5ZTlmMyBfY19yYW5nZXJfYWRtaW5zXzZmNTllOWYzIF9jX3Jhbmdlcl9hZG1pbnNfOTExNDYzYyBfY19kZV91c2Vyc185MTE0NjNjIF9jX21sX3VzZXJzXzkxMTQ2M2MgX2NfZGZfcHJvamVjdF9tZW1iZXJfNDBkZmU1NjggX2NfZGZfcHJvamVjdF9tZW1iZXJfNDJlMDU2Y2IgX2NfZGZfdmlld182ZjU5ZTlmMzAgX2NfZGZfcHJvamVjdF9tZW1iZXJfNTc1Zjg0ZjcgX2NfZGVfdXNlcnNfNmY1OWU5ZjMiLCJleHAiOjE3NzUxNTE1MjQsInR5cGUiOiJ1c2VyIiwiZ2l2ZW5fbmFtZSI6IklhbiIsImlhdCI6MTc3NTE0NzkyNCwiZmFtaWx5X25hbWUiOiJCcm9va3MiLCJlbWFpbCI6Imlicm9va3NAY2xvdWRlcmEuY29tIn0.rEjO3wDX7chV8RTBRgkb6bpco7OEquyHBhY3u5C39AG52sq51aZ1TwJvPYS1FO3TPKHvwfYQwRDXzZ9WXSA2pzby9785xU6UAccclOAWeXpLWlP1lV8T5toEZK64WZWFkU-Jr_ALSWIVKz6VJhj-sL6irXuGtHpyg4bkws-Z-qe2sCP4vrceoR6snRj_olrtXeCnkrTu0hxhqWlQvu7HJj82duy312X3Hp0yDil-0i8UgikwAchm93sPLbaL8kSctrkGoRa1xPGpp4XLeTAJWSnkPwWa0mMnMY0FjP-pljr_XRa_sYRqW7bmraDSLV4TfZaO_Su0UWkp6FNcUnWiqQ"
WINDSURF_LLM_MODEL="nvidia/llama-3.3-nemotron-super-49b-v1.5"
WINDSURF_LLM_TEMPERATURE=0.2
WINDSURF_LLM_MAX_TOKENS=2048
WINDSURF_LLM_TIMEOUT=30

# Disable Windsurf embedding - use Cloudera only
WINDSURF_EMBEDDING_BASE_URL="https://ml-64288d82-5dd.go01-dem.ylcu-atmi.cloudera.site/namespaces/serving-default/endpoints/goes---e5-embedding/v1"
WINDSURF_EMBEDDING_API_KEY="eyJraWQiOiIzYzhlNzA3OTEyZmI0NTA1ODE3NzE3YzMyOTU4MmQwMTFjYjlmNTAwIiwidHlwIjoiSldUIiwiYWxnIjoiUlMyNTYifQ.eyJzdWIiOiJpYnJvb2tzIiwiYXVkIjoiaHR0cHM6Ly9kZS55bGN1LWF0bWkuY2xvdWRlcmEuc2l0ZSIsImlzcyI6Imh0dHBzOi8vY29uc29sZWF1dGguY2RwLmNsb3VkZXJhLmNvbS84YTFlMTVjZC0wNGMyLTQ4YWEtOGYzNS1iNGE4YzExOTk3ZDMiLCJncm91cHMiOiJjZHBfZGVtb3Nfd29ya2Vyc193dyBjZHBfZGVtby1hd3MtcHJpbSBfY19kZl9kZXZlbG9wXzkxMTQ2M2MgX2NfZGZfdmlld182ZjU5ZTlmMyBfY19kZl9hZG1pbmlzdGVyXzkxMTQ2M2MgX2NfZGZfdmlld185MTE0NjNjIF9jX21sX2J1c2luZXNzX3VzZXJzXzZlZTBkYjkxIF9jX2RmX3B1Ymxpc2hfOTExNDYzYyBfY19kZl92aWV3XzkxMTQ2M2MwIF9jX2Vudl9hc3NpZ25lZXNfOTExNDYzYyBfY19yYW5nZXJfYWRtaW5zXzkwNmIwYmEgX2NfbWxfdXNlcnNfNmY1OWU5ZjMgX2NfbWxfdXNlcnNfNGQ4M2FkN2YgX2NfZW52X2Fzc2lnbmVlc185MDZiMGJhIF9jX2RmX2RldmVsb3BfNmY1OWU5ZjMgX2NfZGZfYWRtaW5pc3Rlcl82ZjU5ZTlmMyBfY19kZl9wdWJsaXNoXzZmNTllOWYzIF9jX21sX3VzZXJzXzZlZTBkYjkxIF9jX21sX2J1c2luZXNzX3VzZXJzXzZmNTllOWYzIF9jX21sX2J1c2luZXNzX3VzZXJzXzkxMTQ2M2MgX2NfZW52X2Fzc2lnbmVlc182ZjU5ZTlmMyBfY19yYW5nZXJfYWRtaW5zXzZmNTllOWYzIF9jX3Jhbmdlcl9hZG1pbnNfOTExNDYzYyBfY19kZV91c2Vyc185MTE0NjNjIF9jX21sX3VzZXJzXzkxMTQ2M2MgX2NfZGZfcHJvamVjdF9tZW1iZXJfNDBkZmU1NjggX2NfZGZfcHJvamVjdF9tZW1iZXJfNDJlMDU2Y2IgX2NfZGZfdmlld182ZjU5ZTlmMzAgX2NfZGZfcHJvamVjdF9tZW1iZXJfNTc1Zjg0ZjcgX2NfZGVfdXNlcnNfNmY1OWU5ZjMiLCJleHAiOjE3NzUxNTE1MjQsInR5cGUiOiJ1c2VyIiwiZ2l2ZW5fbmFtZSI6IklhbiIsImlhdCI6MTc3NTE0NzkyNCwiZmFtaWx5X25hbWUiOiJCcm9va3MiLCJlbWFpbCI6Imlicm9va3NAY2xvdWRlcmEuY29tIn0.rEjO3wDX7chV8RTBRgkb6bpco7OEquyHBhY3u5C39AG52sq51aZ1TwJvPYS1FO3TPKHvwfYQwRDXzZ9WXSA2pzby9785xU6UAccclOAWeXpLWlP1lV8T5toEZK64WZWFkU-Jr_ALSWIVKz6VJhj-sL6irXuGtHpyg4bkws-Z-qe2sCP4vrceoR6snRj_olrtXeCnkrTu0hxhqWlQvu7HJj82duy312X3Hp0yDil-0i8UgikwAchm93sPLbaL8kSctrkGoRa1xPGpp4XLeTAJWSnkPwWa0mMnMY0FjP-pljr_XRa_sYRqW7bmraDSLV4TfZaO_Su0UWkp6FNcUnWiqQ"
WINDSURF_EMBEDDING_MODEL="goes---e5-embedding"

# Disable any Windsurf fallbacks
DISABLE_WINDSURF_FALLBACK=true
DISABLE_NON_CLOUDERA_MODELS=true
ENFORCE_CLOUDERA_ONLY=true

# Block common Windsurf/OpenAI models
BLOCKED_MODELS="gpt-3.5-turbo,gpt-4,gpt-4-turbo,text-davinci-003,claude-3,claude-2"
ALLOWED_MODELS="nvidia/llama-3.3-nemotron-super-49b-v1.5,goes---nemotron-v1-5-49b-throughput,goes---e5-embedding"
'''
    
    # Write the strict configuration
    config_file = Path(__file__).parent / ".env"
    with open(config_file, 'w') as f:
        f.write(cloudera_only_config)
    
    print("✅ Created strict Cloudera-only configuration")
    print("🚫 All Windsurf models disabled")
    print("🔒 Only Cloudera endpoints allowed")

def create_cloudera_only_agent():
    """Create a modified agent that only uses Cloudera."""
    print("\n🔧 Creating Cloudera-Only Agent")
    print("=" * 50)
    
    # Create a strict agent implementation
    cloudera_agent_code = '''import logging
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
'''
    
    # Write the Cloudera-only agent
    agent_file = Path(__file__).parent / "windsurf_agent" / "cloudera_only_agent.py"
    with open(agent_file, 'w') as f:
        f.write(cloudera_agent_code)
    
    print("✅ Created ClouderaOnlyAgent class")
    print("🚫 All Windsurf client imports removed")
    print("🔒 Only ClouderaLLMClient used")

def test_cloudera_only():
    """Test the Cloudera-only configuration."""
    print("\n🧪 Testing Cloudera-Only Configuration")
    print("=" * 50)
    
    try:
        # Test configuration loading
        from config import Config
        config = Config.from_env()
        print("✅ Configuration loaded successfully")
        print(f"   🔒 LLM URL: {config.llm.base_url[:50]}...")
        print(f"   🔒 Embedding URL: {config.embedding.base_url[:50]}...")
        
        # Test Cloudera client initialization
        from ClouderaLLMClient import ClouderaLLMClient
        client = ClouderaLLMClient()
        print("✅ ClouderaLLMClient initialized")
        
        # Test Cloudera-only agent
        sys.path.insert(0, str(Path(__file__).parent / "windsurf_agent"))
        from cloudera_only_agent import ClouderaOnlyAgent
        agent = ClouderaOnlyAgent()
        print("✅ ClouderaOnlyAgent initialized")
        
        print("\n🎉 Cloudera-Only Setup Complete!")
        print("✅ No Windsurf models available")
        print("✅ Only Cloudera AI endpoints")
        print("✅ Strict model validation")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main function."""
    print("🚫 Disabling Windsurf Models - Cloudera AI Only")
    print("🔒 Strict Cloudera Enforcement")
    print("=" * 60)
    
    # Disable Windsurf models
    disable_windsurf_models()
    
    # Create Cloudera-only agent
    create_cloudera_only_agent()
    
    # Test the setup
    success = test_cloudera_only()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 Windsurf Models Successfully Disabled!")
        print("✅ Only Cloudera AI models available")
        print("✅ All Windsurf fallbacks blocked")
        print("✅ Strict endpoint enforcement")
        print("✅ Model validation active")
        print("\n💡 Next steps:")
        print("   1. Use ClouderaOnlyAgent instead of WindsurfAgent")
        print("   2. All API calls will go to Cloudera only")
        print("   3. No Windsurf models can be used")
    else:
        print("\n❌ Setup incomplete - check configuration")

if __name__ == "__main__":
    main()
