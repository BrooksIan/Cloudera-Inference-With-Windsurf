import os
from dotenv import load_dotenv
from windsurf_agent import ClouderaLLMClient

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize client
    client = ClouderaLLMClient()
    
    # Stream response
    print("AI: ", end="", flush=True)
    for chunk in client.complete("Tell me about Cloudera ML"):
        print(chunk, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    main()
