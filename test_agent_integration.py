import os
import sys
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def test_agent_integration():
    """Test the agent integration with Cloudera LLM."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Verify required environment variables
        required_vars = [
            'WINDSURF_LLM_BASE_URL',
            'WINDSURF_LLM_API_KEY'
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        logger.info("All required environment variables are set")
        
        # Import after verifying environment
        from windsurf_agent import AgentFactory
        
        logger.info("Creating code agent...")
        code_agent = AgentFactory.create_agent('code')
        
        # Test a simple code generation
        prompt = """
        Create a Python function that calculates the nth Fibonacci number.
        The function should be efficient and handle edge cases.
        Include type hints and a docstring.
        """
        
        logger.info("Generating code...")
        code = code_agent.generate_code(prompt)
        
        logger.info("\nGenerated code:")
        print("\n" + "="*80)
        print(code)
        print("="*80 + "\n")
        
        logger.info("Test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_agent_integration()
    sys.exit(0 if success else 1)
