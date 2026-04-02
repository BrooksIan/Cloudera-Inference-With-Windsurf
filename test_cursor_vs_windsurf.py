#!/usr/bin/env python3
"""
Test script to compare authentication differences between Cursor AI and Windsurf approaches.
"""

import os
import requests
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"✅ Loaded environment from {env_path}")

def test_cursor_style_auth():
    """Test authentication like Cursor AI might do it."""
    print("🔍 Testing Cursor AI Style Authentication...")
    print("=" * 60)
    
    base_url = os.getenv("WINDSURF_LLM_BASE_URL")
    api_key = os.getenv("WINDSURF_LLM_API_KEY")
    model = os.getenv("WINDSURF_LLM_MODEL")
    
    print(f"Base URL: {base_url}")
    print(f"Model: {model}")
    print(f"API Key (first 20): {api_key[:20]}..." if api_key else "❌ No API key")
    
    # Test 1: OpenAI client style (like Windsurf)
    print("\n📋 Test 1: OpenAI Client Style (Windsurf approach)")
    try:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            default_headers={
                "X-Requested-By": "cascade-agent",
                "X-XSRF-Header": "true",
            }
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10,
            temperature=0.1
        )
        print("✅ OpenAI client style: SUCCESS")
        print(f"Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"❌ OpenAI client style: {e}")
    
    # Test 2: Direct HTTP with different auth headers
    print("\n📋 Test 2: Direct HTTP with Various Headers")
    
    test_cases = [
        ("Standard Bearer", {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}),
        ("No Bearer prefix", {"Authorization": api_key, "Content-Type": "application/json"}),
        ("X-API-Key", {"X-API-Key": api_key, "Content-Type": "application/json"}),
        ("Custom format", {"Authorization": f"Token {api_key}", "Content-Type": "application/json"}),
        ("With User-Agent", {
            "Authorization": f"Bearer {api_key}", 
            "Content-Type": "application/json",
            "User-Agent": "Cursor/0.1.0"
        }),
        ("With CSRF headers", {
            "Authorization": f"Bearer {api_key}", 
            "Content-Type": "application/json",
            "X-Requested-By": "cursor",
            "X-XSRF-Header": "true"
        })
    ]
    
    # Test completions endpoint
    completions_url = f"{base_url.rstrip('/')}/completions"
    payload = {
        "model": model,
        "prompt": "Hello",
        "max_tokens": 10,
        "temperature": 0.1
    }
    
    for test_name, headers in test_cases:
        print(f"  Testing {test_name}...")
        try:
            response = requests.post(completions_url, headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                print(f"  ✅ {test_name}: SUCCESS")
                return True
            else:
                print(f"  ❌ {test_name}: {response.status_code}")
                if response.status_code != 401:
                    print(f"     Response: {response.text[:100]}...")
        except Exception as e:
            print(f"  ❌ {test_name}: Error - {e}")
    
    # Test 3: Chat completions endpoint
    print("\n📋 Test 3: Chat Completions Endpoint")
    chat_url = f"{base_url.rstrip('/')}/chat/completions"
    chat_payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10,
        "temperature": 0.1
    }
    
    for test_name, headers in test_cases[:3]:  # Test first 3 cases
        print(f"  Testing chat endpoint with {test_name}...")
        try:
            response = requests.post(chat_url, headers=headers, json=chat_payload, timeout=10)
            if response.status_code == 200:
                print(f"  ✅ Chat {test_name}: SUCCESS")
                return True
            else:
                print(f"  ❌ Chat {test_name}: {response.status_code}")
        except Exception as e:
            print(f"  ❌ Chat {test_name}: Error - {e}")
    
    return False

def test_embedding_differences():
    """Test embedding authentication differences."""
    print("\n🔍 Testing Embedding Authentication Differences...")
    print("=" * 60)
    
    base_url = os.getenv("WINDSURF_EMBEDDING_BASE_URL")
    api_key = os.getenv("WINDSURF_EMBEDDING_API_KEY")
    model = os.getenv("WINDSURF_EMBEDDING_MODEL")
    
    print(f"Embedding Base URL: {base_url}")
    print(f"Embedding Model: {model}")
    
    # Test different endpoint paths
    endpoints = [
        "/embeddings",
        "/v1/embeddings", 
        "/api/v1/embeddings"
    ]
    
    headers_variants = [
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        {"Authorization": api_key, "Content-Type": "application/json"},
        {"X-API-Key": api_key, "Content-Type": "application/json"}
    ]
    
    payload = {
        "model": model,
        "input": "Hello world"
    }
    
    for endpoint in endpoints:
        url = f"{base_url.rstrip('/')}{endpoint}"
        print(f"\nTesting endpoint: {url}")
        
        for i, headers in enumerate(headers_variants):
            header_name = ["Bearer", "No Bearer", "X-API-Key"][i]
            print(f"  Headers: {header_name}")
            
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=10)
                if response.status_code == 200:
                    print(f"  ✅ SUCCESS with {header_name} headers!")
                    result = response.json()
                    if "data" in result:
                        print(f"  Embedding dimension: {len(result['data'][0]['embedding'])}")
                    return True
                else:
                    print(f"  ❌ {header_name}: {response.status_code}")
                    if "Token has expired" in response.text:
                        print(f"  Note: Token expired - need fresh token")
                    elif response.status_code != 401:
                        print(f"  Response: {response.text[:100]}...")
            except Exception as e:
                print(f"  ❌ {header_name}: Error - {e}")
    
    return False

def analyze_token_format():
    """Analyze the JWT token to understand its structure."""
    print("\n🔍 Analyzing Token Format...")
    print("=" * 60)
    
    api_key = os.getenv("WINDSURF_LLM_API_KEY")
    if not api_key:
        print("❌ No API key found")
        return
    
    try:
        # Try to decode JWT header and payload
        import base64
        import json
        
        # Split token
        parts = api_key.split('.')
        if len(parts) >= 2:
            # Decode header
            header_padding = '=' * (-len(parts[0]) % 4)
            header_data = base64.urlsafe_b64decode(parts[0] + header_padding)
            header = json.loads(header_data)
            print(f"Token Header: {json.dumps(header, indent=2)}")
            
            # Decode payload
            payload_padding = '=' * (-len(parts[1]) % 4)
            payload_data = base64.urlsafe_b64decode(parts[1] + payload_padding)
            payload = json.loads(payload_data)
            print(f"Token Payload: {json.dumps(payload, indent=2)}")
            
            # Check expiration
            if 'exp' in payload:
                import time
                exp_time = payload['exp']
                current_time = int(time.time())
                print(f"Token expires: {exp_time} ({time.ctime(exp_time)})")
                print(f"Current time: {current_time} ({time.ctime(current_time)})")
                print(f"Token expired: {exp_time < current_time}")
        else:
            print("❌ Token doesn't appear to be in JWT format")
            
    except Exception as e:
        print(f"❌ Error analyzing token: {e}")

if __name__ == "__main__":
    print("🚀 Cursor AI vs Windsurf Authentication Comparison")
    print("=" * 70)
    
    # Analyze token first
    analyze_token_format()
    
    # Test LLM authentication
    llm_success = test_cursor_style_auth()
    
    # Test embedding authentication  
    embedding_success = test_embedding_differences()
    
    print(f"\n📊 Results:")
    print(f"LLM Authentication: {'✅ PASS' if llm_success else '❌ FAIL'}")
    print(f"Embedding Authentication: {'✅ PASS' if embedding_success else '❌ FAIL'}")
    
    if not llm_success and not embedding_success:
        print("\n💡 Possible causes:")
        print("1. Token has expired (most likely)")
        print("2. Different authentication format expected")
        print("3. Different endpoint paths required")
        print("4. Additional headers needed (User-Agent, CSRF, etc.)")
        print("5. Model name format differences")
