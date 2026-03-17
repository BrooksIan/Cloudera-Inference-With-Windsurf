# examples/code_generation.py
import os
from dotenv import load_dotenv
from pathlib import Path
from windsurf_agent.ClouderaLLMClient import ClouderaLLMClient

def load_environment():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent / '.env'
    if not env_path.exists():
        raise FileNotFoundError(f"Could not find .env file at {env_path}")
    load_dotenv(env_path)
    print(f"Environment loaded from: {env_path}")

def generate_code(description: str, language: str = "python") -> str:
    """
    Generate code based on the provided description.
    
    Args:
        description: A description of what the code should do
        language: The programming language to generate code in
        
    Returns:
        The generated code as a string
    """
    # Load environment variables
    load_environment()
    
    # Initialize the client
    client = ClouderaLLMClient(
        base_url=os.getenv("WINDSURF_LLM_BASE_URL"),
        api_key=os.getenv("WINDSURF_LLM_API_KEY")
    )
    
    # Create a system message
    system_message = f"""You are an expert {language} programming assistant. 
    Generate clean, efficient, and well-documented code based on the user's description.
    Include type hints and docstrings.
    Return only the code block without any additional explanation or markdown formatting.
    """
    
    # Create the user's prompt
    user_prompt = f"""Generate a {language} function that:
    {description}
    
    Requirements:
    - Include proper error handling
    - Add type hints
    - Include a detailed docstring
    - Follow the language's style guide
    - Make it production-ready
    """
    
    # Generate the completion
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]
    
    print(f"Generating {language} code for: {description[:100]}...\n")
    
    # Stream the response
    full_response = ""
    print("Generated code:\n")
    try:
        for chunk in client.chat_completion(messages, stream=True):
            print(chunk, end="", flush=True)
            full_response += chunk
    except Exception as e:
        print(f"\nError generating code: {str(e)}")
        return ""
    
    return full_response

def main():
    # Example usage
    example_description = """
    Takes a list of numbers and returns a dictionary with the sum, average, min, and max values.
    Handle empty lists and non-numeric values appropriately.
    """
    
    # Generate the code
    code = generate_code(example_description, "python")
    
    # Save to file
    if code:
        output_file = "generated_code.py"
        with open(output_file, "w") as f:
            f.write(code)
        print(f"\n\nCode saved to {output_file}")

if __name__ == "__main__":
    main()