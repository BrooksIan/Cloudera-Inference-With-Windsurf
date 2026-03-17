#!/usr/bin/env python3
"""
Demo script for using Cloudera AI models with Cascade.
This script demonstrates how to use both the LLM and embedding models.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from windsurf_agent.config import Config
from windsurf_agent.llm_client import WindsurfLLMClient
from windsurf_agent.embedding_client import WindsurfEmbeddingClient
from windsurf_agent.vector_store import SimpleVectorStore, Document

# Load environment variables from project root .env file
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"✅ Loaded environment from {env_path}")
else:
    print(f"❌ .env file not found at {env_path}")
    # Try loading from current directory
    load_dotenv()

def test_llm_completion():
    """Test the Cloudera LLM model for text completion."""
    print("🤖 Testing Cloudera LLM Model...")
    print("=" * 50)
    
    # Load configuration
    config = Config.from_env()
    
    # Create LLM client
    llm_client = WindsurfLLMClient(config.llm)
    
    # Test completion
    prompt = "What is machine learning? Explain in simple terms."
    print(f"Prompt: {prompt}")
    print("-" * 30)
    
    try:
        response = llm_client.complete(prompt)
        print(f"Response: {response}")
        print("✅ LLM test successful!")
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
    
    print()

def test_llm_chat():
    """Test the Cloudera LLM model for chat completion."""
    print("💬 Testing Cloudera LLM Chat...")
    print("=" * 50)
    
    # Load configuration
    config = Config.from_env()
    
    # Create LLM client
    llm_client = WindsurfLLMClient(config.llm)
    
    # Test chat
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What are the benefits of using Cloudera for AI/ML workloads?"}
    ]
    
    print("Chat messages:")
    for msg in messages:
        print(f"  {msg['role']}: {msg['content']}")
    print("-" * 30)
    
    try:
        response = llm_client.chat(messages)
        print(f"Assistant: {response}")
        print("✅ Chat test successful!")
    except Exception as e:
        print(f"❌ Chat test failed: {e}")
    
    print()

def test_embeddings():
    """Test the Cloudera embedding model."""
    print("🔍 Testing Cloudera Embedding Model...")
    print("=" * 50)
    
    # Load configuration
    config = Config.from_env()
    
    # Create embedding client
    embedding_client = WindsurfEmbeddingClient(config.embedding)
    
    # Test single embedding
    text = "Cloudera provides enterprise data platform solutions."
    print(f"Text: {text}")
    print("-" * 30)
    
    try:
        embedding = embedding_client.get_embedding(text, input_type="passage")
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        print("✅ Single embedding test successful!")
    except Exception as e:
        print(f"❌ Single embedding test failed: {e}")
    
    print()
    
    # Test batch embeddings
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Cloudera Data Platform supports various AI/ML workloads."
    ]
    
    print("Testing batch embeddings...")
    print(f"Number of texts: {len(texts)}")
    print("-" * 30)
    
    try:
        embeddings = embedding_client.get_embeddings(texts, input_type="passage")
        print(f"Batch embeddings shape: {len(embeddings)} x {len(embeddings[0])}")
        print("✅ Batch embedding test successful!")
    except Exception as e:
        print(f"❌ Batch embedding test failed: {e}")
    
    print()

def test_rag_pipeline():
    """Test a simple RAG (Retrieval-Augmented Generation) pipeline."""
    print("🔗 Testing RAG Pipeline...")
    print("=" * 50)
    
    # Load configuration
    config = Config.from_env()
    
    # Create clients
    llm_client = WindsurfLLMClient(config.llm)
    embedding_client = WindsurfEmbeddingClient(config.embedding)
    
    # Sample documents
    documents = [
        "Cloudera Data Platform (CDP) is a unified data platform that combines analytics and machine learning.",
        "CDP supports various workloads including data engineering, analytics, and ML/AI.",
        "The platform includes components like Cloudera Machine Learning (CML) for data science workloads.",
        "CML provides tools for model training, deployment, and monitoring in enterprise environments.",
        "Cloudera offers both on-premises and cloud deployment options for flexibility."
    ]
    
    print("Creating document embeddings...")
    
    try:
        # Create vector store
        vector_config = config.vector_store
        vector_store = SimpleVectorStore(vector_config)
        
        # Add documents to vector store
        doc_ids = []
        for i, doc_text in enumerate(documents):
            embedding = embedding_client.get_embedding(doc_text, input_type="passage")
            doc_id = vector_store.add_document(
                text=doc_text,
                embedding=embedding,
                metadata={"source": f"doc_{i+1}"}
            )
            doc_ids.append(doc_id)
        
        print(f"Added {len(doc_ids)} documents to vector store")
        
        # Query the vector store
        query = "What machine learning capabilities does Cloudera offer?"
        query_embedding = embedding_client.get_embedding(query, input_type="query")
        
        # Search for relevant documents
        results = vector_store.similarity_search(query_embedding, k=2)
        
        print(f"\nQuery: {query}")
        print("Relevant documents:")
        for doc, score in results:
            print(f"  Score: {score:.3f} - {doc.text[:100]}...")
        
        # Generate response using retrieved context
        context = "\n".join([doc.text for doc, _ in results])
        
        rag_prompt = f"""Based on the following context, answer the question:

Context:
{context}

Question: {query}

Answer:"""
        
        response = llm_client.complete(rag_prompt)
        print(f"\nRAG Response: {response}")
        print("✅ RAG pipeline test successful!")
        
    except Exception as e:
        print(f"❌ RAG pipeline test failed: {e}")
    
    print()

def main():
    """Run all demo tests."""
    print("🚀 Cloudera AI Models Demo with Cascade")
    print("=" * 60)
    print()
    
    # Check environment variables
    required_vars = [
        "WINDSURF_LLM_BASE_URL",
        "WINDSURF_LLM_API_KEY", 
        "WINDSURF_LLM_MODEL",
        "WINDSURF_EMBEDDING_BASE_URL",
        "WINDSURF_EMBEDDING_API_KEY",
        "WINDSURF_EMBEDDING_MODEL"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please ensure your .env file is properly configured.")
        return
    
    print("✅ Environment variables configured")
    print()
    
    # Run tests
    test_llm_completion()
    test_llm_chat()
    test_embeddings()
    test_rag_pipeline()
    
    print("🎉 Demo completed!")

if __name__ == "__main__":
    main()
