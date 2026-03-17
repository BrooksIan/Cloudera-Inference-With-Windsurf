#!/usr/bin/env python3
"""
Script to count the first 20 prime numbers using Cloudera hosted models.
This demonstrates using Cloudera's LLM service for computational tasks.
"""

import os
from dotenv import load_dotenv
from windsurf_agent.ClouderaLLMClient import ClouderaLLMClient


def get_prime_code_from_cloudera():
    """Use Cloudera LLM to generate Python code for finding prime numbers."""
    
    # Load environment variables
    load_dotenv()
    
    # Initialize Cloudera LLM client
    client = ClouderaLLMClient()
    
    # Prompt the LLM to write code for finding prime numbers
    prompt = """Write a Python function that finds and returns the first 20 prime numbers. 
    The function should be called 'find_first_20_primes()' and should return a list of integers.
    Only provide the function code, no explanations or additional text."""
    
    messages = [
        {"role": "system", "content": "You are a Python programmer that provides clean, functional code."},
        {"role": "user", "content": prompt}
    ]
    
    print("Asking Cloudera hosted model to write Python code for prime numbers...")
    print("Generated code: ", end="", flush=True)
    
    # Get response from Cloudera model
    response_text = ""
    for chunk in client.chat_completion(messages, stream=False):
        response_text += chunk
    
    print(response_text)
    
    return response_text.strip()


def verify_primes_locally(primes):
    """Verify the prime numbers using a simple local algorithm."""
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    print("\nVerifying primes locally...")
    valid_primes = []
    for i, num in enumerate(primes, 1):
        if is_prime(num):
            valid_primes.append(num)
            print(f"{i:2d}. {num} ✓")
        else:
            print(f"{i:2d}. {num} ❌ (not prime)")
    
    return valid_primes


def execute_generated_code(code_text):
    """Execute the generated code and return the prime numbers."""
    try:
        # Create a local namespace to execute the code
        namespace = {}
        
        # Execute the generated code
        exec(code_text, namespace)
        
        # Call the generated function
        if 'find_first_20_primes' in namespace:
            primes = namespace['find_first_20_primes']()
            print(f"\n✓ Successfully executed LLM-generated code")
            print(f"Found {len(primes)} prime numbers")
            return primes
        else:
            print("❌ Generated code doesn't contain 'find_first_20_primes()' function")
            return None
            
    except Exception as e:
        print(f"❌ Error executing generated code: {e}")
        return None


def main():
    """Main function to demonstrate Cloudera LLM code generation for prime numbers."""
    print("=" * 60)
    print("Prime Number Code Generation using Cloudera Hosted Models")
    print("=" * 60)
    
    # Get code from Cloudera model
    generated_code = get_prime_code_from_cloudera()
    
    if generated_code:
        # Execute the generated code
        primes = execute_generated_code(generated_code)
        
        if primes:
            # Verify locally
            verified_primes = verify_primes_locally(primes)
            
            print(f"\nResults:")
            print(f"- Primes found by LLM code: {len(primes)}")
            print(f"- Verified primes: {len(verified_primes)}")
            print(f"- First prime: {verified_primes[0] if verified_primes else 'N/A'}")
            print(f"- Last prime: {verified_primes[-1] if verified_primes else 'N/A'}")
            
            if len(verified_primes) == 20:
                print("✓ LLM successfully generated working code for 20 prime numbers!")
            else:
                print("⚠ Generated code found incorrect number of primes")
        else:
            print("Failed to execute generated code")
    else:
        print("Failed to get code from Cloudera model")


if __name__ == "__main__":
    main()
