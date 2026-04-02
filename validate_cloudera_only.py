#!/usr/bin/env python3
"""
Simple validation script to check Cloudera-only configuration.
"""

import os
import sys
from pathlib import Path

# Add the windsurf_agent to the path
sys.path.insert(0, str(Path(__file__).parent / "windsurf_agent"))

def validate_cloudera_endpoints():
    """Validate that environment variables point to Cloudera endpoints only."""
    print("=== Validating Cloudera Endpoints ===")
    
    # Check LLM endpoint
    llm_base_url = os.getenv("WINDSURF_LLM_BASE_URL", "")
    if not llm_base_url:
        print("✗ WINDSURF_LLM_BASE_URL not set")
        return False
    
    if "cloudera.site" not in llm_base_url:
        print(f"✗ WINDSURF_LLM_BASE_URL is not a Cloudera endpoint: {llm_base_url}")
        return False
    
    print(f"✓ LLM Base URL is Cloudera endpoint: {llm_base_url}")
    
    # Check embedding endpoint
    embedding_base_url = os.getenv("WINDSURF_EMBEDDING_BASE_URL", "")
    if not embedding_base_url:
        print("✗ WINDSURF_EMBEDDING_BASE_URL not set")
        return False
    
    if "cloudera.site" not in embedding_base_url:
        print(f"✗ WINDSURF_EMBEDDING_BASE_URL is not a Cloudera endpoint: {embedding_base_url}")
        return False
    
    print(f"✓ Embedding Base URL is Cloudera endpoint: {embedding_base_url}")
    
    return True

def validate_models():
    """Validate that models are Cloudera AI hosted models."""
    print("\n=== Validating Models ===")
    
    # Allowed Cloudera models
    allowed_models = [
        "goes---nemotron-v1-5-49b-throughput",
        "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        "goes---e5-embedding",
        "nvidia/nv-embedqa-e5-v5",
        "nvidia/nv-embedqa-e5-v5-query",
        "nvidia/nv-embedqa-e5-v5-passage"
    ]
    
    llm_model = os.getenv("WINDSURF_LLM_MODEL", "")
    if llm_model not in allowed_models:
        print(f"✗ LLM model not in allowed list: {llm_model}")
        print(f"  Allowed models: {', '.join(allowed_models)}")
        return False
    
    print(f"✓ LLM model is allowed: {llm_model}")
    
    embedding_model = os.getenv("WINDSURF_EMBEDDING_MODEL", "")
    if embedding_model not in allowed_models:
        print(f"✗ Embedding model not in allowed list: {embedding_model}")
        print(f"  Allowed models: {', '.join(allowed_models)}")
        return False
    
    print(f"✓ Embedding model is allowed: {embedding_model}")
    
    return True

def validate_code_changes():
    """Validate that code changes enforce Cloudera-only usage."""
    print("\n=== Validating Code Changes ===")
    
    # Check agent.py uses ClouderaLLMClient
    agent_file = Path(__file__).parent / "windsurf_agent" / "agent.py"
    if not agent_file.exists():
        print("✗ agent.py not found")
        return False
    
    agent_content = agent_file.read_text()
    
    if "from .ClouderaLLMClient import ClouderaLLMClient" not in agent_content:
        print("✗ agent.py doesn't import ClouderaLLMClient")
        return False
    
    if "self.llm_client = ClouderaLLMClient()" not in agent_content:
        print("✗ agent.py doesn't use ClouderaLLMClient")
        return False
    
    if "_validate_cloudera_model" not in agent_content:
        print("✗ agent.py doesn't have model validation")
        return False
    
    print("✓ agent.py correctly configured for Cloudera-only usage")
    
    # Check config.py enforces Cloudera endpoints
    config_file = Path(__file__).parent / "windsurf_agent" / "config.py"
    if not config_file.exists():
        print("✗ config.py not found")
        return False
    
    config_content = config_file.read_text()
    
    if "cloudera.site" not in config_content:
        print("✗ config.py doesn't enforce Cloudera endpoints")
        return False
    
    print("✓ config.py enforces Cloudera endpoints only")
    
    # Check ClouderaLLMClient exists and validates endpoints
    cloudera_client_file = Path(__file__).parent / "windsurf_agent" / "ClouderaLLMClient.py"
    if not cloudera_client_file.exists():
        print("✗ ClouderaLLMClient.py not found")
        return False
    
    client_content = cloudera_client_file.read_text()
    
    if "cloudera.site" not in client_content:
        print("✗ ClouderaLLMClient doesn't validate Cloudera endpoints")
        return False
    
    print("✓ ClouderaLLMClient validates Cloudera endpoints")
    
    return True

def main():
    """Run all validations."""
    print("Cloudera AI-only Configuration Validation")
    print("=" * 50)
    
    # Load environment variables
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip('"\'')
                    os.environ[key] = value
    
    validations = [
        validate_cloudera_endpoints,
        validate_models,
        validate_code_changes
    ]
    
    passed = 0
    total = len(validations)
    
    for validation in validations:
        try:
            if validation():
                passed += 1
        except Exception as e:
            print(f"✗ Validation {validation.__name__} failed: {e}")
    
    print("\n" + "=" * 50)
    print(f"Validation Results: {passed}/{total} validations passed")
    
    if passed == total:
        print("🎉 All validations passed! The agent is configured for Cloudera AI only.")
        return 0
    else:
        print("❌ Some validations failed. Please check the configuration.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
