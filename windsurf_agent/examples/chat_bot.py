import os
import logging
from dotenv import load_dotenv
from windsurf_agent.agent import WindsurfAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize the agent
    agent = WindsurfAgent()
    
    print("Windsurf Chat Bot")
    print("Type 'quit' to exit\n")
    
    # Chat loop
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break
            
        # Get response using chat
        messages = [{"role": "user", "content": user_input}]
        response = agent.chat(messages)
        
        print(f"Assistant: {response}\n")

if __name__ == "__main__":
    main()