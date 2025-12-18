import os
import logging
from dotenv import load_dotenv
from windsurf_agent.agent import WindsurfAgent
from windsurf_agent.llm_client import WindsurfLLMClient as LLMClient
from windsurf_agent.config import Config, LLMConfig
from windsurf_agent.vector_store import SimpleVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedRAGExample:
    def __init__(self):
        """Initialize the advanced RAG example with an agent and LLM client."""
        # Load environment variables
        load_dotenv()
        
        # Create a custom config
        config = Config.from_env()
        
        # Update the embedding model based on the environment
        # Using the base model ID that worked in the curl test
        config.embedding.model = os.getenv("WINDSURF_EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")
        
        # Initialize the agent with the custom config
        self.agent = WindsurfAgent(config=config)
        
        # Store query model for search
        # For now, using the same model for both passage and query
        self.query_model = os.getenv("WINDSURF_EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")
        
        # Initialize a direct LLM client for comparison
        llm_config = LLMConfig(
            base_url=os.getenv("WINDSURF_LLM_BASE_URL", "https://api.windsurf.ai/v1"),
            api_key=os.getenv("WINDSURF_LLM_API_KEY", "your-api-key-here"),
            model=os.getenv("WINDSURF_LLM_MODEL", "gpt-3.5-turbo"),
            temperature=0.7,
            max_tokens=1000
        )
        self.llm = LLMClient(llm_config)
        
        # Sample knowledge base
        self.knowledge_base = [
            "The Windsurf Agent is a powerful tool for building AI applications.",
            "It supports Retrieval-Augmented Generation (RAG) for more accurate responses.",
            "The agent can be customized with different LLM providers and configurations.",
            "Windsurf supports both chat and completion APIs for different use cases.",
            "The framework includes built-in support for document processing and vector storage.",
            "You can extend the agent with custom tools and integrations.",
            "The latest version includes improved error handling and performance optimizations.",
        ]
    
    def reset_vector_store(self):
        """Reset the vector store with the correct dimension."""
        # Explicitly set the dimension to 1024
        self.agent.config.vector_store.dimension = 1024
        # Reinitialize the vector store with the updated config
        self.agent.vector_store = SimpleVectorStore(self.agent.config.vector_store)
        logger.info(f"Vector store reset with dimension: {self.agent.vector_store.dimension}")

    def search(self, query: str, k: int = 5, **kwargs):
        """Override search to use query model for search."""
        # Save the original model
        original_model = self.agent.embedding_client.model
        
        try:
            # Use query model for search
            self.agent.embedding_client.model = self.query_model
            return self.agent.search(query, k=k, **kwargs)
        finally:
            # Restore the original model
            self.agent.embedding_client.model = original_model

    def setup_knowledge_base(self):
        """Initialize the knowledge base with sample documents."""
        print("\nSetting up knowledge base...")
    
        # Reset the vector store to ensure correct dimension
        self.reset_vector_store()

        # Create metadata for each document
        metadatas = [{"source": f"doc_{i+1}"} for i in range(len(self.knowledge_base))]
    
        # Add documents with metadata
        self.agent.add_to_knowledge_base(
            texts=self.knowledge_base,
            metadatas=metadatas
        )
        print(f"Knowledge base initialized with {len(self.knowledge_base)} documents.")
    
    def ask_question(self, question: str, use_rag: bool = True):
        """Ask a question using either RAG or direct LLM.
        
        Args:
            question: The question to ask
            use_rag: If True, use RAG. If False, use direct LLM.
        """
        print("\n" + "=" * 80)
        print(f"QUESTION ({'RAG' if use_rag else 'Direct LLM'})")
        print("=" * 80)
        print(f"\nQ: {question}")
        
        try:
            if use_rag:
                # Use RAG for answering
                result = self.agent.rag_query(question)
                print("\nA:", result["answer"])
                
                if result["sources"]:
                    print("\nSources:")
                    for i, source in enumerate(result["sources"], 1):
                        print(f"{i}. {source['text']} (Score: {source['score']:.3f})")
            else:
                # Use direct LLM for answering
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question}
                ]
                response = self.llm.chat(messages)
                # Handle both string and dictionary responses
                if isinstance(response, dict) and 'content' in response:
                    print("\nA:", response["content"])
                else:
                    print("\nA:", str(response))
                print("\n(No sources - Direct LLM response)")
                
        except Exception as e:
            logger.error(f"Error while getting answer: {str(e)}")
            raise

def main():
    example = AdvancedRAGExample()
    
    # Set up the knowledge base
    example.setup_knowledge_base()
    
    # Example questions to demonstrate RAG vs direct LLM
    questions = [
        "What is the Windsurf Agent?",
        "What are the key features of the latest version?",
        "How can I extend the agent's functionality?"
    ]
    
    for question in questions:
        # First, ask using RAG
        example.ask_question(question, use_rag=True)
        
        # Then, ask the same question using direct LLM for comparison
        example.ask_question(question, use_rag=False)

if __name__ == "__main__":
    main()