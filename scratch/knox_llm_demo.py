import os
import logging
from dotenv import load_dotenv
from pathlib import Path

# Import the KnoxLLMClient
from knox_llm_client import KnoxLLMClient
from windsurf_agent.config import LLMConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_llm_client() -> KnoxLLMClient:
    """Initialize and return an LLM client with configuration from environment variables."""
    # Load environment variables from .env file
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path, override=True)
    
    # Debug: Print environment variables
    print("\nEnvironment Variables:")
    print(f"WINDSURF_LLM_API_KEY: {'*' * 20 if os.getenv('WINDSURF_LLM_API_KEY') else 'Not set'}")
    print(f"WINDSURF_LLM_BASE_URL: {os.getenv('WINDSURF_LLM_BASE_URL')}")
    print(f"WINDSURF_LLM_MODEL: {os.getenv('WINDSURF_LLM_MODEL')}")
    
    # Create configuration
    config = LLMConfig(
        base_url=os.getenv('WINDSURF_LLM_BASE_URL', ''),
        api_key=os.getenv('WINDSURF_LLM_API_KEY', ''),
        model=os.getenv('WINDSURF_LLM_MODEL', 'nvidia/llama-3.3-nemotron-super-49b-v1.5'),
        temperature=float(os.getenv('WINDSURF_LLM_TEMPERATURE', '0.7')),
        max_tokens=int(os.getenv('WINDSURF_LLM_MAX_TOKENS', '2048')),
        timeout=int(os.getenv('WINDSURF_LLM_TIMEOUT', '60')),
        max_retries=int(os.getenv('WINDSURF_LLM_MAX_RETRIES', '3'))
    )
    
    if not config.base_url or 'your-cloudera-ml-endpoint' in config.base_url:
        raise ValueError("LLM base URL is not properly configured. Please check your .env file.")
    
    return KnoxLLMClient(config)

def run_chat_example(llm: KnoxLLMClient):
    """Run a simple chat example."""
    print("\n=== Chat Example ===")
    print("Sending chat request...")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, can you tell me a short joke?"}
    ]
    
    try:
        response = llm.chat(messages)
        print(f"\nResponse: {response}")
    except Exception as e:
        logger.error(f"Error in chat example: {str(e)}", exc_info=True)
        raise

def run_completion_example(llm: KnoxLLMClient):
    """Run a simple completion example."""
    print("\n=== Completion Example ===")
    print("Sending completion request...")
    
    prompt = "Once upon a time"
    
    try:
        response = llm.complete(
            prompt=prompt,
            max_tokens=50
        )
        print(f"\nPrompt: {prompt}")
        print(f"Completion: {response}")
    except Exception as e:
        logger.error(f"Error in completion example: {str(e)}", exc_info=True)
        raise

def main():
    """Main function to run the demo."""
    try:
        # Initialize the LLM client
        print("Initializing LLM client...")
        llm = setup_llm_client()
        
        # Run examples
        run_chat_example(llm)
        run_completion_example(llm)
        
        print("\nDemo completed successfully!")
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
