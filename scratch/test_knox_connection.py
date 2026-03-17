"""
Test script for Knox-protected LLM endpoint
"""
import os
import sys
import logging
from dotenv import load_dotenv
from windsurf_agent.config import Config

# Add parent directory to path to import knox_llm_client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scratch.knox_llm_client import KnoxLLMClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_knox_client() -> KnoxLLMClient:
    """Initialize and return a Knox LLM client with configuration from environment variables."""
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
        
        # Create configuration from environment variables
        config = Config.from_env()
        
        # Print configuration
        print("\nLLM Configuration:")
        print(f"Base URL: {config.llm.base_url}")
        print(f"Model: {config.llm.model}")
        print(f"Timeout: {config.llm.timeout}")
        print(f"Max Retries: {config.llm.max_retries}")
        
        return KnoxLLMClient(config.llm)
        
    except Exception as e:
        logger.error(f"Failed to initialize Knox LLM client: {str(e)}", exc_info=True)
        raise

def test_knox_connection():
    """Test connection to Knox-protected LLM endpoint."""
    try:
        client = setup_knox_client()
        
        # Test a simple completion request
        print("\nTesting Knox LLM connection...")
        
        # Simple test request
        test_prompt = "Hello, this is a test. Please respond with 'Connection successful!'"
        
        print(f"\nSending test prompt: {test_prompt}")
        
        response = client.complete(
            prompt=test_prompt,
            temperature=0.7,
            max_tokens=50
        )
        
        print("\nResponse:")
        print(response)
        
        return True
        
    except Exception as e:
        logger.error(f"Knox connection test failed: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    print("=== Knox LLM Connection Tester ===\n")
    success = test_knox_connection()
    
    if success:
        print("\n✅ Knox LLM connection test completed successfully!")
    else:
        print("\n❌ Knox LLM connection test failed. Please check the error messages above.")
