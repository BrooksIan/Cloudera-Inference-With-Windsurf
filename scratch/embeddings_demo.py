#!/usr/bin/env python3
"""
Simple demo script for using Cloudera Embedding Model with Cascade.
This script demonstrates how to use the embedding model for semantic search.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from windsurf_agent.config import Config
from windsurf_agent.embedding_client import WindsurfEmbeddingClient
from windsurf_agent.vector_store import SimpleVectorStore, Document
import numpy as np

# Load environment variables from project root .env file
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"✅ Loaded environment from {env_path}")
else:
    print(f"❌ .env file not found at {env_path}")
    load_dotenv()

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def test_embeddings():
    """Test the Cloudera embedding model and semantic search."""
    print("🔍 Testing Cloudera Embedding Model for Semantic Search")
    print("=" * 60)
    
    # Load configuration
    config = Config.from_env()
    
    # Create embedding client
    embedding_client = WindsurfEmbeddingClient(config.embedding)
    
    # Sample documents about Cloudera and AI/ML
    documents = [
        "Cloudera Data Platform (CDP) is a unified hybrid data platform",
        "CDP provides enterprise-grade data management and analytics",
        "Machine learning workloads can be deployed on Cloudera's platform",
        "Cloudera supports both on-premises and cloud deployments",
        "Data scientists use Cloudera Machine Learning for model development",
        "The platform includes tools for data engineering and streaming",
        "Cloudera offers security and governance for enterprise data",
        "AI and ML models can be trained and deployed using CDP",
        "Real-time analytics is supported through Cloudera's streaming capabilities",
        "The platform integrates with popular ML frameworks like TensorFlow and PyTorch"
    ]
    
    print(f"Processing {len(documents)} documents...")
    print("-" * 40)
    
    # Create embeddings for all documents
    document_embeddings = []
    for i, doc in enumerate(documents):
        try:
            embedding = embedding_client.get_embedding(doc)
            document_embeddings.append(embedding)
            print(f"✅ Document {i+1}: Embedded (dimension: {len(embedding)})")
        except Exception as e:
            print(f"❌ Document {i+1}: Failed - {e}")
            return
    
    print(f"\n✅ Successfully embedded {len(document_embeddings)} documents")
    print(f"Embedding dimension: {len(document_embeddings[0])}")
    
    # Test semantic search
    print("\n🔎 Testing Semantic Search")
    print("=" * 30)
    
    queries = [
        "What machine learning capabilities does Cloudera offer?",
        "How does Cloudera handle cloud deployments?",
        "What security features are available?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 30)
        
        try:
            # Create embedding for the query
            query_embedding = embedding_client.get_embedding(query)
            
            # Calculate similarities
            similarities = []
            for i, doc_embedding in enumerate(document_embeddings):
                similarity = cosine_similarity(query_embedding, doc_embedding)
                similarities.append((i, similarity, documents[i]))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Show top 3 results
            print("Top 3 most relevant documents:")
            for rank, (doc_idx, similarity, doc_text) in enumerate(similarities[:3], 1):
                print(f"  {rank}. [Score: {similarity:.3f}] {doc_text[:80]}...")
                
        except Exception as e:
            print(f"❌ Query failed: {e}")

def test_embedding_similarity():
    """Test embedding similarity between related and unrelated texts."""
    print("\n🔗 Testing Embedding Similarity")
    print("=" * 30)
    
    # Load configuration
    config = Config.from_env()
    
    # Create embedding client
    embedding_client = WindsurfEmbeddingClient(config.embedding)
    
    # Test pairs of texts
    test_pairs = [
        ("Cloudera provides data platform solutions", "CDP offers enterprise data management"),
        ("Machine learning requires training data", "Deep learning uses neural networks"),
        ("The weather is nice today", "Cloudera supports hybrid cloud deployments")
    ]
    
    for i, (text1, text2) in enumerate(test_pairs, 1):
        try:
            emb1 = embedding_client.get_embedding(text1)
            emb2 = embedding_client.get_embedding(text2)
            
            similarity = cosine_similarity(emb1, emb2)
            
            print(f"\nPair {i}:")
            print(f"  Text 1: {text1}")
            print(f"  Text 2: {text2}")
            print(f"  Similarity: {similarity:.3f}")
            
            if similarity > 0.7:
                print("  → Highly related")
            elif similarity > 0.4:
                print("  → Somewhat related")
            else:
                print("  → Not related")
                
        except Exception as e:
            print(f"❌ Pair {i} failed: {e}")

def main():
    """Run the embedding demo."""
    print("🚀 Cloudera Embedding Model Demo with Cascade")
    print("=" * 50)
    
    # Check environment variables
    required_vars = [
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
    test_embeddings()
    test_embedding_similarity()
    
    print("\n🎉 Embedding demo completed!")
    print("\n💡 This demonstrates how to use Cloudera's embedding model for:")
    print("   • Document embedding and storage")
    print("   • Semantic search capabilities")
    print("   • Text similarity analysis")
    print("   • Building RAG (Retrieval-Augmented Generation) systems")

if __name__ == "__main__":
    main()
