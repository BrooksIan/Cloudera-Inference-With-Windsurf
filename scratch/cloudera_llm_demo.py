"""
Cloudera AI LLM Demo

This script demonstrates how to use the Cloudera AI LLM models with the Windsurf agent.
It shows both chat and completion examples using the configured Cloudera AI endpoints.
"""

import os
import logging
from dotenv import load_dotenv
from windsurf_agent.llm_client import WindsurfLLMClient as LLMClient
from windsurf_agent.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_llm_client() -> LLMClient:
    """Initialize and return an LLM client with configuration from environment variables."""
    try:
        # Get the absolute path to the .env file
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
        
        # Load environment variables from .env file with override
        loaded = load_dotenv(env_path, override=True)
        
        # Debug: Print environment file info
        print(f"\nLoading .env file: {env_path}")
        print(f"File exists: {os.path.exists(env_path)}")
        print(f"Environment variables loaded: {loaded}")
        
        # Debug: Print environment variables (safely)
        print("\nEnvironment Variables:")
        print(f"WINDSURF_LLM_API_KEY: {'*' * 20 if os.getenv('WINDSURF_LLM_API_KEY') else 'Not set'}")
        print(f"WINDSURF_LLM_BASE_URL: {os.getenv('WINDSURF_LLM_BASE_URL', 'Not set')}")
        print(f"WINDSURF_LLM_MODEL: {os.getenv('WINDSURF_LLM_MODEL', 'Not set')}")
        
        # Verify required environment variables
        required_vars = ['WINDSURF_LLM_API_KEY', 'WINDSURF_LLM_BASE_URL', 'WINDSURF_LLM_MODEL']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Create configuration from environment variables
        config = Config.from_env()
        
        # Validate configuration
        if not config.llm.base_url or 'your-cloudera-ml-endpoint' in config.llm.base_url:
            raise ValueError("LLM base URL is not properly configured. Please check your .env file.")
            
        if not config.llm.api_key or 'your-llm-api-key' in config.llm.api_key:
            raise ValueError("LLM API key is not properly configured. Please check your .env file.")
        
        # Log the configuration (without sensitive data)
        logger.info(f"Initializing LLM client with model: {config.llm.model}")
        logger.info(f"Using base URL: {config.llm.base_url}")
        
        # Print configuration details
        print("\nConfiguration:")
        print(f"Model: {config.llm.model}")
        print(f"Base URL: {config.llm.base_url}")
        print(f"Temperature: {config.llm.temperature}")
        print(f"Max Tokens: {config.llm.max_tokens}")
        print(f"Timeout: {config.llm.timeout}s")
        print(f"Max Retries: {config.llm.max_retries}")
        
        return LLMClient(config.llm)
        
    except Exception as e:
        logger.error(f"Failed to initialize LLM client: {str(e)}", exc_info=True)
        raise

def run_chat_example(llm: LLMClient):
    """Run a simple chat example with the LLM."""
    print("\n=== Chat Example ===")
    
    # Define the conversation
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Provide concise and accurate responses."},
        {"role": "user", "content": "What are the key benefits of using Cloudera AI for machine learning?"}
    ]
    
    try:
        print("\nSending chat message...")
        response = llm.chat(messages)
        
        # Handle the response
        if isinstance(response, str):
            print("\nAssistant:", response)
        elif isinstance(response, dict) and "content" in response:
            print("\nAssistant:", response["content"])
        else:
            print("\nReceived unexpected response format:", response)
            
    except Exception as e:
        logger.error(f"Error in chat example: {str(e)}", exc_info=True)

def run_completion_example(llm: LLMClient):
    """Run a completion example with the LLM."""
    print("\n=== Completion Example ===")
    
    prompt = """
    Write a short Python function that calculates the factorial of a number.
    Include type hints and a docstring.
    """
    
    try:
        print("\nSending completion request...")
        response = llm.complete(
            prompt=prompt,
            max_tokens=300,
            temperature=0.7
        )
        
        print("\nCompletion result:")
        print(response)
        
    except Exception as e:
        logger.error(f"Error in completion example: {str(e)}", exc_info=True)

def main():
    """Main function to run the demo."""
    try:
        print("Starting Cloudera AI LLM Demo...")
        
        # Initialize the LLM client
        llm = setup_llm_client()
        
        # Run examples
        run_chat_example(llm)
        run_completion_example(llm)
        
        print("\nDemo completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        print("\nAn error occurred. Please check the logs for more details.")

if __name__ == "__main__":
    main()
