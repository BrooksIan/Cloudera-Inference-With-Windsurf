#!/usr/bin/env python3
"""
Debug script to test the Cloudera LLM endpoint directly.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import requests
import json

# Load environment variables from project root .env file
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"✅ Loaded environment from {env_path}")

def test_llm_endpoint():
    """Test the LLM endpoint directly with different approaches."""
    
    # Get configuration
    base_url = os.getenv("WINDSURF_LLM_BASE_URL")
    api_key = os.getenv("WINDSURF_LLM_API_KEY")
    model = os.getenv("WINDSURF_LLM_MODEL")
    
    print(f"Base URL: {base_url}")
    print(f"Model: {model}")
    print(f"API Key: {api_key[:50]}...")
    
    # Test different endpoints
    endpoints_to_test = [
        "",  # Direct to base URL
        "/v1",  # Add v1
        "/completions",  # Standard completions
        "/chat/completions",  # Standard chat
        "/generate",  # Alternative
        "/predict",  # Alternative
    ]
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    for endpoint in endpoints_to_test:
        url = f"{base_url}{endpoint}"
        print(f"\n🔍 Testing: {url}")
        
        # Try different payload formats
        payloads = [
            # OpenAI-style completion
            {
                "model": model,
                "prompt": "What is 2+2?",
                "temperature": 0.7,
                "max_tokens": 100
            },
            # OpenAI-style chat
            {
                "model": model,
                "messages": [
                    {"role": "user", "content": "What is 2+2?"}
                ],
                "temperature": 0.7,
                "max_tokens": 100
            },
            # Simple format
            {
                "input": "What is 2+2?",
                "model": model
            },
            # HuggingFace style
            {
                "inputs": "What is 2+2?",
                "parameters": {
                    "temperature": 0.7,
                    "max_new_tokens": 100
                }
            }
        ]
        
        for i, payload in enumerate(payloads):
            try:
                print(f"  Payload {i+1}: {json.dumps(payload, indent=2)}")
                
                response = requests.post(url, headers=headers, json=payload, timeout=10)
                
                print(f"  Status: {response.status_code}")
                print(f"  Response: {response.text[:200]}...")
                
                if response.status_code == 200:
                    print("  ✅ SUCCESS!")
                    return response.json()
                    
            except Exception as e:
                print(f"  Error: {e}")
    
    print("\n❌ No successful endpoint found")
    return None

def test_endpoint_info():
    """Try to get information about the endpoint."""
    
    base_url = os.getenv("WINDSURF_LLM_BASE_URL")
    api_key = os.getenv("WINDSURF_LLM_API_KEY")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Try GET requests to common info endpoints
    info_endpoints = ["/", "/v1", "/info", "/models", "/health"]
    
    for endpoint in info_endpoints:
        url = f"{base_url}{endpoint}"
        print(f"\n🔍 Testing GET: {url}")
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            print(f"  Status: {response.status_code}")
            print(f"  Response: {response.text[:200]}...")
            
            if response.status_code == 200:
                print("  ✅ SUCCESS!")
                
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    print("🚀 Cloudera LLM Endpoint Debug")
    print("=" * 40)
    
    test_endpoint_info()
    test_llm_endpoint()
