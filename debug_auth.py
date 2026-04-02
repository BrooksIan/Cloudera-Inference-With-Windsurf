#!/usr/bin/env python3
"""
Debug script to test LLM authentication and API connectivity.
"""

import os
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"✅ Loaded environment from {env_path}")

def test_llm_auth():
    """Test LLM authentication with debug information."""
    print("🔍 Testing LLM Authentication...")
    print("=" * 50)
    
    # Get configuration
    base_url = os.getenv("WINDSURF_LLM_BASE_URL")
    api_key = os.getenv("WINDSURF_LLM_API_KEY")
    model = os.getenv("WINDSURF_LLM_MODEL")
    
    print(f"Base URL: {base_url}")
    print(f"Model: {model}")
    print(f"API Key (first 20 chars): {api_key[:20]}..." if api_key else "❌ No API key found")
    print()
    
    if not base_url or not api_key:
        print("❌ Missing required configuration")
        return False
    
    # Test with a simple completion request
    payload = {
        "model": model,
        "prompt": "Hello",
        "max_tokens": 10,
        "temperature": 0.1
    }
    
    # Try the completions endpoint
    completions_url = f"{base_url.rstrip('/')}/completions"
    
    # Test different authentication formats
    auth_formats = [
        ("Bearer token", {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}),
        ("Token only", {"Authorization": api_key, "Content-Type": "application/json"}),
        ("X-API-Key", {"X-API-Key": api_key, "Content-Type": "application/json"}),
        ("Custom Auth", {"Authorization": f"Token {api_key}", "Content-Type": "application/json"})
    ]
    
    for format_name, headers in auth_formats:
        print(f"Testing {format_name}...")
        try:
            response = requests.post(completions_url, headers=headers, json=payload, timeout=10)
            print(f"  {format_name}: {response.status_code}")
            if response.status_code == 200:
                print(f"  ✅ SUCCESS with {format_name}!")
                return True
            elif response.status_code != 401:
                print(f"  Response: {response.text[:200]}...")
        except Exception as e:
            print(f"  {format_name}: Error - {e}")
    
    return False

def test_embedding_auth():
    """Test embedding authentication."""
    print("\n🔍 Testing Embedding Authentication...")
    print("=" * 50)
    
    # Get configuration
    base_url = os.getenv("WINDSURF_EMBEDDING_BASE_URL")
    api_key = os.getenv("WINDSURF_EMBEDDING_API_KEY")
    model = os.getenv("WINDSURF_EMBEDDING_MODEL")
    
    print(f"Base URL: {base_url}")
    print(f"Model: {model}")
    print(f"API Key (first 20 chars): {api_key[:20]}..." if api_key else "❌ No API key found")
    print()
    
    if not base_url or not api_key:
        print("❌ Missing required configuration")
        return False
    
    # Test with a simple embedding request
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "input": "Hello world",
        "input_type": "passage"  # Required for this embedding model
    }
    
    # Try the embeddings endpoint
    embeddings_url = f"{base_url.rstrip('/')}/embeddings"
    print(f"Testing endpoint: {embeddings_url}")
    
    try:
        response = requests.post(embeddings_url, headers=headers, json=payload, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Authentication successful!")
            result = response.json()
            if "data" in result and len(result["data"]) > 0:
                embedding = result["data"][0]["embedding"]
                print(f"Embedding dimension: {len(embedding)}")
            return True
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Authentication Debug Script")
    print("=" * 60)
    
    llm_success = test_llm_auth()
    embedding_success = test_embedding_auth()
    
    print(f"\n📊 Results:")
    print(f"LLM Authentication: {'✅ PASS' if llm_success else '❌ FAIL'}")
    print(f"Embedding Authentication: {'✅ PASS' if embedding_success else '❌ FAIL'}")
