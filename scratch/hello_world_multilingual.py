#!/usr/bin/env python3
"""
A script that demonstrates using a Cloudera-hosted LLM to generate
"Hello, World!" in multiple languages.

This script enforces the use of Cloudera-hosted LLMs and demonstrates
how to properly configure and use the Windsurf LLM client.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file in project root
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"Loaded environment variables from {env_path}")
else:
    print(f"No .env file found at {env_path}. Using system environment variables.")

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log which environment variables are loaded
loaded_vars = {k: '***' if 'key' in k.lower() or 'secret' in k.lower() or 'token' in k.lower() else v 
              for k, v in os.environ.items() if k.startswith(('WINDSURF_', 'LOG_'))}
logger.debug(f"Loaded environment variables: {loaded_vars}")

try:
    from windsurf_agent.llm_client import WindsurfLLMClient
except ImportError as e:
    logger.error("Failed to import Windsurf LLM client. Make sure windsurf_agent is installed.")
    raise

# Handle cloudera_config import
try:
    from cloudera_config import enforce_cloudera_models
except ImportError:
    try:
        # Try relative import if running as a module
        from .cloudera_config import enforce_cloudera_models
    except ImportError:
        logger.error("Failed to import cloudera_config. Make sure it's in your PYTHONPATH.")
        raise

def get_hello_world(api_key: str, base_url: str, model: str = "nvidia/llama-3.3-nemotron-super-49b-v1") -> str:
    """
    Get 'Hello, World!' in multiple languages using Cloudera LLM.
    
    Args:
        api_key: API key for authentication
        base_url: Base URL for the LLM API
        model: Model to use for generation (default: nvidia/llama-3.3-nemotron-super-49b-v1)
        
    Returns:
        str: Formatted response from the LLM or error message
    """
    try:
        logger.info("Enforcing Cloudera models only policy")
        enforce_cloudera_models()
        
        logger.info("Initializing Windsurf LLM client")
        from windsurf_agent.config import LLMConfig
        
        # Create config from environment variables
        config = LLMConfig(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=0.3,
            max_tokens=2048,
            timeout=60,
            max_retries=3
        )
        client = WindsurfLLMClient(config=config)
        
        prompt = """Please provide "Hello, World!" in three different languages. 
Format your response as a numbered list with the language name and translation.
Example:
1. Spanish: ¡Hola Mundo!
2. French: Bonjour le monde!
3. Japanese: こんにちは世界！"""
        
        logger.info("Sending request to LLM")
        response = client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Lower temperature for more predictable output
            model=model
        )
        
        if not response or not response.strip():
            raise ValueError("Received empty response from the model")
            
        logger.info("Successfully received response from LLM")
        return response.strip()
        
    except Exception as e:
        logger.error(f"Error in get_hello_world: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

def get_environment_vars() -> Dict[str, str]:
    """
    Get and validate required environment variables.
    
    Returns:
        Dict containing the required environment variables
        
    Raises:
        ValueError: If any required environment variables are missing
    """
    env_vars = {
        'base_url': os.getenv("WINDSURF_LLM_BASE_URL"),
        'api_key': os.getenv("WINDSURF_LLM_API_KEY"),
        'model': os.getenv("WINDSURF_LLM_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1")
    }
    
    missing = [k.upper() for k, v in env_vars.items() if v is None and k != 'model']
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
    logger.debug(f"Environment variables loaded: {', '.join(k for k in env_vars if env_vars[k])}")
    return env_vars

def create_example_env() -> Path:
    """Create an example .env file in the project root if it doesn't exist.
    
    Returns:
        Path: Path to the created .env file, or None if it already exists
    """
    env_path = Path(__file__).parent.parent / '.env'
    if not env_path.exists():
        with open(env_path, 'w') as f:
            f.write("""# Windsurf LLM Configuration
WINDSURF_LLM_BASE_URL=your-cloudera-ml-endpoint
WINDSURF_LLM_API_KEY=your-api-key
WINDSURF_LLM_MODEL=nvidia/llama-3.3-nemotron-super-49b-v1

# Optional: Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
# LOG_LEVEL=INFO
""")
        print(f"Created example .env file at {env_path}")
        return env_path
    return None

def print_usage() -> None:
    """Print usage instructions."""
    print("\nUsage:")
    print("1. Create a .env file with your configuration:")
    print("   - The script will create an example .env file if one doesn't exist")
    print("   - Edit the .env file with your actual credentials")
    print("2. Run the script:")
    print("    python -m scratch.hello_world_multilingual")
    print("\nFor debugging, you can enable verbose output by adding to .env:")
    print("    LOG_LEVEL=DEBUG")

def main() -> int:
    """
    Main function to run the script.
    
    Returns:
        int: Exit code (0 for success, non-zero for errors)
    """
    try:
        # Create example .env if it doesn't exist
        env_file = create_example_env()
        if env_file:
            print(f"Please edit {env_file} with your configuration and run the script again.")
            return 1
            
        # Set log level from environment or default to INFO
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, log_level, logging.INFO))
        
        logger.info("Starting Hello World Multilingual script")
        
        # Get environment variables
        try:
            env_vars = get_environment_vars()
        except ValueError as e:
            logger.error(str(e))
            print_usage()
            return 1
        
        # Get and display the translations
        print("\n=== Generating 'Hello, World!' in multiple languages ===\n")
        
        result = get_hello_world(
            api_key=env_vars['api_key'],
            base_url=env_vars['base_url'],
            model=env_vars['model']
        )
        
        print(result)
        print("\n" + "="*60 + "\n")
        print("Note: These translations were generated by a Cloudera-hosted LLM.")
        
        logger.info("Script completed successfully")
        return 0
        
    except Exception as e:
        logger.critical(f"Unexpected error: {str(e)}", exc_info=True)
        print(f"\nError: {str(e)}\n")
        print("Please check the logs for more details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
