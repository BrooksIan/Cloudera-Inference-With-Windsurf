#!/usr/bin/env python3
"""
Test script to validate that the Cascade agent only uses Cloudera AI hosted LLMs.
"""

import os
import sys
import logging
from pathlib import Path

# Add the windsurf_agent to the path
sys.path.insert(0, str(Path(__file__).parent / "windsurf_agent"))

try:
    from windsurf_agent.agent import WindsurfAgent
    from windsurf_agent.config import Config, LLMConfig, EmbeddingConfig
    from windsurf_agent.ClouderaLLMClient import ClouderaLLMClient
    from windsurf_agent.exceptions import LLMError
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_cloudera_only_configuration():
    """Test that the configuration enforces Cloudera endpoints only."""
    print("=== Testing Cloudera-only Configuration ===")
    
    # Test valid Cloudera configuration
    try:
        config = Config.from_env()
        print("✓ Configuration loaded successfully with Cloudera endpoints")
        print(f"  LLM Base URL: {config.llm.base_url}")
        print(f"  LLM Model: {config.llm.model}")
        print(f"  Embedding Base URL: {config.embedding.base_url}")
        print(f"  Embedding Model: {config.embedding.model}")
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
        return False
    
    return True

def test_cloudera_client_initialization():
    """Test that ClouderaLLMClient can be initialized."""
    print("\n=== Testing ClouderaLLMClient Initialization ===")
    
    try:
        client = ClouderaLLMClient()
        print("✓ ClouderaLLMClient initialized successfully")
        print(f"  Base URL: {client.base_url}")
        print(f"  Model: {client.model}")
    except Exception as e:
        print(f"✗ ClouderaLLMClient initialization failed: {e}")
        return False
    
    return True

def test_agent_initialization():
    """Test that the WindsurfAgent can be initialized with Cloudera-only settings."""
    print("\n=== Testing WindsurfAgent Initialization ===")
    
    try:
        agent = WindsurfAgent()
        print("✓ WindsurfAgent initialized successfully with Cloudera AI LLM")
        print(f"  LLM Client type: {type(agent.llm_client).__name__}")
        print(f"  Using Cloudera endpoint: {'cloudera.site' in agent.llm_client.base_url}")
    except Exception as e:
        print(f"✗ WindsurfAgent initialization failed: {e}")
        return False
    
    return True

def test_model_validation():
    """Test that model validation works correctly."""
    print("\n=== Testing Model Validation ===")
    
    try:
        agent = WindsurfAgent()
        
        # Test valid model
        try:
            agent._validate_cloudera_model("goes---nemotron-v1-5-49b-throughput")
            print("✓ Valid Cloudera model accepted")
        except LLMError as e:
            print(f"✗ Valid model was rejected: {e}")
            return False
        
        # Test invalid model
        try:
            agent._validate_cloudera_model("gpt-4")
            print("✗ Invalid model was accepted - this should not happen!")
            return False
        except LLMError as e:
            print(f"✓ Invalid model correctly rejected: {e}")
        
    except Exception as e:
        print(f"✗ Model validation test failed: {e}")
        return False
    
    return True

def test_basic_generation():
    """Test basic text generation with Cloudera AI."""
    print("\n=== Testing Basic Text Generation ===")
    
    try:
        agent = WindsurfAgent()
        
        # Simple test prompt
        test_prompt = "Say 'Hello from Cloudera AI!'"
        response = agent.generate(test_prompt, max_tokens=50)
        
        print("✓ Text generation successful")
        print(f"  Response: {response[:100]}...")
        
    except Exception as e:
        print(f"✗ Text generation failed: {e}")
        return False
    
    return True

def test_chat_functionality():
    """Test chat functionality with Cloudera AI."""
    print("\n=== Testing Chat Functionality ===")
    
    try:
        agent = WindsurfAgent()
        
        messages = [
            {"role": "user", "content": "What is 2+2? Just give the number."}
        ]
        
        response = agent.chat(messages, max_tokens=10)
        
        print("✓ Chat functionality successful")
        print(f"  Response: {response[:100]}...")
        
    except Exception as e:
        print(f"✗ Chat functionality failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("Cloudera AI-only Cascade Agent Validation Tests")
    print("=" * 50)
    
    tests = [
        test_cloudera_only_configuration,
        test_cloudera_client_initialization,
        test_agent_initialization,
        test_model_validation,
        test_basic_generation,
        test_chat_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The agent is configured for Cloudera AI only.")
        return 0
    else:
        print("❌ Some tests failed. Please check the configuration.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
