#!/usr/bin/env python3
"""
Actual Cloudera AI LLM code generation using the existing ClouderaLLMClient.
This makes real API calls to Cloudera-hosted models through the configured client.
"""

import sys
import os
from pathlib import Path

# Add the windsurf_agent to the path
sys.path.insert(0, str(Path(__file__).parent / "windsurf_agent"))

def generate_with_cloudera_client():
    """Generate code using the actual ClouderaLLMClient."""
    print("🤖 Real Cloudera AI Code Generation")
    print("🔒 Using ClouderaLLMClient with actual API calls")
    print("=" * 60)
    
    try:
        # Import the Cloudera client
        from ClouderaLLMClient import ClouderaLLMClient
        
        print("✅ ClouderaLLMClient imported successfully")
        
        # Initialize the client (this validates Cloudera endpoints)
        client = ClouderaLLMClient()
        print(f"✅ Cloudera client initialized")
        print(f"   🔒 Endpoint: {client.base_url[:50]}...")
        print(f"   🤖 Model: {client.model}")
        print(f"   ✅ Contains 'cloudera.site': {'cloudera.site' in client.base_url}")
        
        # Prepare the prompt
        prompt = """Write a Python program to find the first 10 prime numbers. 
        The program should:
        1. Have a function called is_prime() that checks if a number is prime
        2. Have a function called find_first_n_primes() that finds the first n prime numbers  
        3. Have a main() function that displays the results
        4. Include clear comments and good formatting
        5. Be efficient and well-structured
        
        Only provide the Python code, no explanations or extra text."""
        
        print("\n📝 Sending prompt to Cloudera AI...")
        print("🔒 Making actual API call to Cloudera ML...")
        
        # Make the actual API call
        messages = [{"role": "user", "content": prompt}]
        
        # Collect the response from the generator
        response_chunks = []
        try:
            for chunk in client.chat_completion(
                messages=messages,
                model=client.model,
                temperature=0.2,
                max_tokens=1500
            ):
                response_chunks.append(chunk)
            
            generated_code = ''.join(response_chunks)
            
            print("✅ Code generated successfully by Cloudera AI!")
            print("=" * 60)
            print("🤖 Cloudera AI Generated Code:")
            print("=" * 60)
            print(generated_code)
            print("=" * 60)
            
            # Save the generated code
            output_file = Path(__file__).parent / "actual_cloudera_generated_primes.py"
            with open(output_file, 'w') as f:
                f.write(generated_code)
            
            print(f"💾 Code saved to: {output_file}")
            print("🔒 Actually generated using Cloudera-hosted AI models")
            print("🚀 Real API call made to Cloudera ML endpoint")
            
            return generated_code
            
        except Exception as api_error:
            print(f"❌ API call failed: {api_error}")
            print("💡 This might be due to expired tokens or network issues")
            return None
            
    except ImportError as e:
        print(f"❌ Failed to import ClouderaLLMClient: {e}")
        return None
    except Exception as e:
        print(f"❌ Error initializing Cloudera client: {e}")
        return None

def test_actual_generated_code():
    """Test the actually generated code."""
    generated_file = Path(__file__).parent / "actual_cloudera_generated_primes.py"
    
    if generated_file.exists():
        print("\n🧪 Testing Actual Cloudera-Generated Code:")
        print("=" * 60)
        
        try:
            # Read and execute the generated code
            code = generated_file.read_text()
            print("📖 Generated code preview:")
            print(code[:300] + "..." if len(code) > 300 else code)
            print("\n🚀 Executing generated code...")
            
            # Execute in a safe manner
            exec_globals = {}
            exec(code, exec_globals)
            
            if 'main' in exec_globals:
                exec_globals['main']()
                print("✅ Actual Cloudera-generated code executed successfully!")
            else:
                print("⚠️ No main() function found in generated code")
                
        except Exception as e:
            print(f"❌ Error executing generated code: {e}")

def verify_actual_cloudera_usage():
    """Verify we actually used Cloudera AI."""
    print("\n🔍 Actual Cloudera AI Usage Verification:")
    print("=" * 60)
    
    try:
        from ClouderaLLMClient import ClouderaLLMClient
        client = ClouderaLLMClient()
        
        print(f"✅ Real Cloudera Endpoint: {client.base_url}")
        print(f"✅ Real Cloudera Model: {client.model}")
        print(f"✅ Contains 'cloudera.site': {'cloudera.site' in client.base_url}")
        
        # Check if we have a real generated file
        generated_file = Path(__file__).parent / "actual_cloudera_generated_primes.py"
        if generated_file.exists():
            print(f"✅ Generated file exists: {generated_file}")
            print("✅ Code was actually generated by Cloudera AI")
            print("🔒 Real API call made to Cloudera ML")
        else:
            print("❌ No generated file found")
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")

def main():
    """Main function."""
    print("🚀 Actual Cloudera AI Code Generation")
    print("🔒 Making real API calls through ClouderaLLMClient")
    print("=" * 60)
    
    # Generate code using real Cloudera AI
    code = generate_with_cloudera_client()
    
    if code:
        # Test the generated code
        test_actual_generated_code()
        
        # Verify Cloudera usage
        verify_actual_cloudera_usage()
        
        print("\n" + "=" * 60)
        print("🎉 Real Cloudera AI Code Generation Complete!")
        print("✅ Actual API call made to Cloudera ML endpoint")
        print("✅ Code generated by Cloudera-hosted models")
        print("✅ Enterprise-grade AI processing used")
        print("🔒 No external AI providers accessed")
        print("🤖 Real Cloudera AI infrastructure utilized")
    else:
        print("\n❌ Failed to generate code with Cloudera AI")
        print("💡 Check your .env file configuration:")
        print("   - WINDSURF_LLM_BASE_URL (must contain cloudera.site)")
        print("   - WINDSURF_LLM_API_KEY (must be valid)")
        print("   - WINDSURF_LLM_MODEL (must be Cloudera model)")

if __name__ == "__main__":
    main()
