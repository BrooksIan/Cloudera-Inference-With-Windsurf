"""
Basic Data Processing Script
Demonstrates fundamental Python operations including:
- List comprehensions
- Functions with type hints
- Basic file I/O
- Error handling
"""
from typing import List, Tuple
import random
import json


def generate_random_numbers(count: int, min_val: int = 1, max_val: int = 100) -> List[int]:
    """Generate a list of random integers within a specified range."""
    return [random.randint(min_val, max_val) for _ in range(count)]


def process_numbers(numbers: List[int]) -> Tuple[float, int, int]:
    """Calculate statistics for a list of numbers."""
    if not numbers:
        raise ValueError("Input list cannot be empty")
    
    average = sum(numbers) / len(numbers)
    minimum = min(numbers)
    maximum = max(numbers)
    
    return average, minimum, maximum


def save_results(data: dict, filename: str = 'results.json') -> None:
    """Save processed results to a JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {filename}")
    except IOError as e:
        print(f"Error saving file: {e}")


def main():
    # Generate and process random numbers
    numbers = generate_random_numbers(20)
    print(f"Generated numbers: {numbers}")
    
    try:
        avg, min_val, max_val = process_numbers(numbers)
        results = {
            'numbers': numbers,
            'statistics': {
                'average': round(avg, 2),
                'minimum': min_val,
                'maximum': max_val,
                'count': len(numbers)
            },
            'sorted_ascending': sorted(numbers),
            'sorted_descending': sorted(numbers, reverse=True)
        }
        
        print("\nStatistics:")
        print(f"- Average: {results['statistics']['average']}")
        print(f"- Minimum: {results['statistics']['minimum']}")
        print(f"- Maximum: {results['statistics']['maximum']}")
        
        # Save results to file
        save_results(results, 'number_analysis.json')
        
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
