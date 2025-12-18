import os
import logging
from dotenv import load_dotenv
from windsurf_agent.llm_client import WindsurfLLMClient as LLMClient
from windsurf_agent.config import LLMConfig
from windsurf_agent.config import Config
print("LLM API Key:", os.getenv("WINDSURF_LLM_API_KEY") is not None)
print("LLM Base URL:", os.getenv("WINDSURF_LLM_BASE_URL"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_llm_client() -> LLMClient:
    """Initialize and return an LLM client with configuration from environment variables."""
    config = LLMConfig(
        api_key=os.getenv("WINDSURF_LLM_API_KEY", "your-api-key-here"),
        base_url=os.getenv("WINDSURF_LLM_BASE_URL", "https://api.windsurf.ai/v1"),
        model=os.getenv("WINDSURF_LLM_MODEL", "gpt-3.5-turbo"),
        temperature=0.7,
        max_tokens=1000,
        timeout=30
    )
    return LLMClient(config)

def run_chat_example():
    """Run a simple chat example with the LLM."""
    try:
        # Initialize the LLM client
        config = Config.from_env()
        llm = LLMClient(config=config.llm)
        
        # Define the conversation
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a short joke about programming."}
        ]
        
        print("Sending chat message...")
        response = llm.chat(messages)
        
        # Handle the response
        if isinstance(response, str):
            print("\nAssistant:", response)
            assistant_response = response
        elif isinstance(response, dict) and "content" in response:
            print("\nAssistant:", response["content"])
            assistant_response = response["content"]
        else:
            print("\nAssistant:", str(response))
            assistant_response = str(response)
        
        # Continue the conversation
        messages.append({"role": "assistant", "content": assistant_response})
        messages.append({"role": "user", "content": "That's funny! Can you explain why it's funny?"})
        
        print("\nSending follow-up message...")
        response = llm.chat(messages)
        
        # Handle the follow-up response
        if isinstance(response, str):
            print("\nAssistant:", response)
        elif isinstance(response, dict) and "content" in response:
            print("\nAssistant:", response["content"])
        else:
            print("\nAssistant:", str(response))
            
    except Exception as e:
        logger.error(f"Error in chat example: {str(e)}")
        raise

def run_completion_example():
    """Run a completion example with the LLM."""
    try:
        # Initialize the LLM client
        config = Config.from_env()
        llm = LLMClient(config=config.llm)
        
        prompt = """
        Write a short product description for a new smartwatch with the following features:
        - Heart rate monitoring
        - Sleep tracking
        - 7-day battery life
        - Waterproof up to 50 meters
        - Built-in GPS
        
        Product description:
        """
        
        print("Sending completion request...")
        response = llm.complete(prompt)
        
        # Handle the response
        if isinstance(response, str):
            print("\nGenerated description:", response)
        elif isinstance(response, dict) and "content" in response:
            print("\nGenerated description:", response["content"])
        else:
            print("\nGenerated description:", str(response))
            
    except Exception as e:
        logger.error(f"Error in completion example: {str(e)}")
        raise

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    print("=" * 80)
    print("LLM CHAT EXAMPLE")
    print("=" * 80)
    run_chat_example()
    
    print("\n" + "=" * 80)
    print("LLM COMPLETION EXAMPLE")
    print("=" * 80)
    run_completion_example()

if __name__ == "__main__":
    main()
