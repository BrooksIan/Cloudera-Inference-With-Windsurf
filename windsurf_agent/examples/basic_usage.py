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
    
    # Add some documents to the knowledge base
    documents = [
        "The capital of France is Paris.",
        "The Eiffel Tower is located in Paris.",
        "France is known for its wine and cheese.",
        "The official language of France is French."
    ]
    
    print("Adding documents to knowledge base...")
    agent.add_to_knowledge_base(documents)
    
    # Ask a question
    question = "What is the capital of France?"
    print(f"\nQuestion: {question}")
    
    # Get answer using RAG
    result = agent.rag_query(question)
    
    print("\nAnswer:")
    print(result["answer"])
    
    print("\nSources:")
    for i, source in enumerate(result["sources"], 1):
        print(f"{i}. {source['text']} (Score: {source['score']:.3f})")

if __name__ == "__main__":
    main()