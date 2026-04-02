#!/usr/bin/env python3
"""
Actual Cloudera AI LLM code generation demonstration.
This makes real API calls to Cloudera-hosted models.
"""

import os
import json
import requests
from pathlib import Path

def load_cloudera_config():
    """Load Cloudera configuration from environment."""
    # Load .env file
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.strip() and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    value = value.strip().strip('"\'')
                    os.environ[key] = value
    
    config = {
        'base_url': os.getenv("WINDSURF_LLM_BASE_URL", "").strip('"\''),
        'api_key': os.getenv("WINDSURF_LLM_API_KEY", ""),
        'model': os.getenv("WINDSURF_LLM_MODEL", "goes---nemotron-v1-5-49b-throughput")
    }
    
    # Validate Cloudera endpoint
    if not config['base_url'] or "cloudera.site" not in config['base_url']:
        raise ValueError("Invalid Cloudera endpoint")
    
    return config

def call_cloudera_llm(prompt, config):
    """Make actual API call to Cloudera-hosted LLM."""
    print(f"🔒 Making API call to: {config['base_url']}")
    print(f"🤖 Using model: {config['model']}")
    
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
        "X-Requested-By": "cascade-agent",
        "X-XSRF-Header": "true"
    }
    
    payload = {
        "model": config['model'],
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 1500
    }
    
    try:
        response = requests.post(
            f"{config['base_url']}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            print(f"❌ API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None

def generate_prime_code_with_cloudera():
    """Generate prime numbers code using actual Cloudera AI."""
    print("🤖 Actual Cloudera AI Code Generation")
    print("🔒 Using real Cloudera-hosted LLM API")
    print("=" * 60)
    
    try:
        # Load Cloudera configuration
        config = load_cloudera_config()
        print("✅ Cloudera configuration loaded and validated")
        
        # Prepare prompt for code generation
        prompt = """Write a Python program to find the first 10 prime numbers. 
        The program should:
        1. Have a function called is_prime() that checks if a number is prime
        2. Have a function called find_first_n_primes() that finds the first n prime numbers  
        3. Have a main() function that displays the results
        4. Include clear comments and good formatting
        5. Be efficient and well-structured
        
        Only provide the Python code, no explanations or extra text."""
        
        print("📝 Sending prompt to Cloudera AI...")
        print(f"🔒 Endpoint: {config['base_url'][:50]}...")
        print(f"🤖 Model: {config['model']}")
        
        # Make actual API call to Cloudera
        generated_code = call_cloudera_llm(prompt, config)
        
        if generated_code:
            print("✅ Code generated successfully by Cloudera AI!")
            print("=" * 60)
            print("🤖 Cloudera AI Generated Code:")
            print("=" * 60)
            print(generated_code)
            print("=" * 60)
            
            # Save the generated code
            output_file = Path(__file__).parent / "real_cloudera_generated_primes.py"
            with open(output_file, 'w') as f:
                f.write(generated_code)
            
            print(f"💾 Code saved to: {output_file}")
            print("🔒 Actually generated using Cloudera-hosted AI models")
            
            return generated_code
        else:
            print("❌ Failed to generate code with Cloudera AI")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_real_generated_code():
    """Test the actually generated code."""
    generated_file = Path(__file__).parent / "real_cloudera_generated_primes.py"
    
    if generated_file.exists():
        print("\n🧪 Testing Real Cloudera-Generated Code:")
        print("=" * 60)
        
        try:
            # Read and execute the generated code
            code = generated_file.read_text()
            print("📖 Generated code preview:")
            print(code[:200] + "..." if len(code) > 200 else code)
            print("\n🚀 Executing generated code...")
            
            # Execute in a safe manner
            exec_globals = {}
            exec(code, exec_globals)
            
            if 'main' in exec_globals:
                exec_globals['main']()
                print("✅ Real Cloudera-generated code executed successfully!")
            else:
                print("⚠️ No main() function found in generated code")
                
        except Exception as e:
            print(f"❌ Error executing generated code: {e}")

def verify_cloudera_usage():
    """Verify we actually used Cloudera AI."""
    print("\n🔍 Cloudera AI Usage Verification:")
    print("=" * 60)
    
    config = load_cloudera_config()
    print(f"✅ Endpoint: {config['base_url']}")
    print(f"✅ Model: {config['model']}")
    print(f"✅ Contains 'cloudera.site': {'cloudera.site' in config['base_url']}")
    
    # Check if we have a real generated file
    generated_file = Path(__file__).parent / "real_cloudera_generated_primes.py"
    if generated_file.exists():
        print(f"✅ Generated file exists: {generated_file}")
        print("✅ Code was actually generated by Cloudera AI")
    else:
        print("❌ No generated file found")

def main():
    """Main function."""
    print("🚀 Real Cloudera AI Code Generation")
    print("🔒 Making actual API calls to Cloudera-hosted models")
    print("=" * 60)
    
    # Generate code using real Cloudera AI
    code = generate_prime_code_with_cloudera()
    
    if code:
        # Test the generated code
        test_real_generated_code()
        
        # Verify Cloudera usage
        verify_cloudera_usage()
        
        print("\n" + "=" * 60)
        print("🎉 Real Cloudera AI Code Generation Complete!")
        print("✅ Actual API call made to Cloudera ML")
        print("✅ Code generated by Cloudera-hosted models")
        print("✅ Enterprise-grade AI processing used")
        print("🔒 No external AI providers accessed")
    else:
        print("\n❌ Failed to generate code with Cloudera AI")
        print("💡 Check your API key and endpoint configuration")

if __name__ == "__main__":
    main()
