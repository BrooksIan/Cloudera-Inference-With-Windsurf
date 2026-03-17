#!/usr/bin/env python3
"""
Complete Python script to find the first 20 prime numbers.
Based on code generation from Cloudera hosted LLM.
"""

# Enforce Cloudera-hosted models only
try:
    from scratch.cloudera_config import enforce_cloudera_models
    enforce_cloudera_models()
except ImportError:
    print("Warning: Could not import enforce_cloudera_models - running without Cloudera enforcement")

def find_first_20_primes():
    """Find and return the first 20 prime numbers."""
    primes = []
    num = 2
    
    while len(primes) < 20:
        # Check if num is prime
        is_prime = True
        
        if num < 2:
            is_prime = False
        else:
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0:
                    is_prime = False
                    break
        
        if is_prime:
            primes.append(num)
        
        num += 1
    
    return primes


def main():
    """Main function to display the first 20 prime numbers."""
    print("First 20 Prime Numbers")
    print("=" * 30)
    
    primes = find_first_20_primes()
    
    for i, prime in enumerate(primes, 1):
        print(f"{i:2d}. {prime}")
    
    print(f"\nTotal primes found: {len(primes)}")
    print(f"Last prime: {primes[-1]}")


if __name__ == "__main__":
    main()
