import os
import logging
import json
from dotenv import load_dotenv
from windsurf_agent.llm_client import WindsurfLLMClient as LLMClient
from windsurf_agent.config import LLMConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
load_dotenv()

def main():
    try:
        # Verify environment variables
        api_key = os.getenv("WINDSURF_LLM_API_KEY")
        base_url = os.getenv("WINDSURF_LLM_BASE_URL")
        
        print("\nConfiguration:")
        print(f"- API Key: {'*' * 8 + api_key[-4:] if api_key else 'Not set'}")
        print(f"- Base URL: {base_url or 'Not set'}")
        
        if not api_key or not base_url:
            raise ValueError("Missing required environment variables. Please check your .env file.")
        
        # Print the first few characters of the API key for verification
        logger.info(f"Using API key starting with: {api_key[:8]}...")
            
        logger.info("Initializing LLM client...")
        
        # Initialize the LLM client with configuration
        config = LLMConfig(
            api_key=api_key,
            base_url=base_url,
            model="nvidia/llama-3.3-nemotron-super-49b-v1"
        )
        
        llm_client = LLMClient(config)
        
        # Create a simple chat prompt
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello, World!' in a creative way."}
        ]
        
        logger.info("Sending request to LLM...")
        
        # Print the request being sent
        request_data = {
            "model": "nvidia/llama-3.3-nemotron-super-49b-v1",
            "messages": messages,
            "max_tokens": 50,
            "temperature": 0.7
        }
        print("\nSending request to LLM:")
        print(json.dumps(request_data, indent=2))
        
        # Get the response from the LLM
        response = llm_client.chat(
            messages=messages,
            max_tokens=50,
            temperature=0.7
        )
        
        print("\nRaw response:")
        print(response)
        
        # Replace the response handling section with this:

        # Print the response
        if response:
            if hasattr(response, 'choices') and len(response.choices) > 0:
                print("\nLLM says:")
                print("-" * 50)
                print(response.choices[0].message.content)
                print("-" * 50)
            elif isinstance(response, str):
                print("\nLLM says (raw response):")
                print("-" * 50)
                print(response)
                print("-" * 50)
            else:
                logger.error("Unexpected response format from LLM")
                logger.error(f"Response type: {type(response)}")
                logger.error(f"Response content: {response}")
        else:
            logger.error("No response received from the LLM")
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        print("\nTroubleshooting Tips:")
        print("1. Verify your WINDSURF_LLM_API_KEY is set correctly in your .env file")
        print("2. Check that WINDSURF_LLM_BASE_URL points to the correct endpoint")
        print("3. Ensure your API key has the necessary permissions")
        print("4. Verify your network connection and that the service is available")

if __name__ == "__main__":
    main()
