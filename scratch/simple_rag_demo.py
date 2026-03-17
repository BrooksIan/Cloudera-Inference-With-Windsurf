#!/usr/bin/env python3
"""
Simple RAG (Retrieval-Augmented Generation) demo using Cloudera Embeddings.
This demonstrates how to build a Q&A system using Cascade with Cloudera models.
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

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class SimpleRAGSystem:
    """A simple RAG system using Cloudera embeddings."""
    
    def __init__(self):
        # Load configuration
        config = Config.from_env()
        
        # Create embedding client
        self.embedding_client = WindsurfEmbeddingClient(config.embedding)
        
        # Create vector store
        self.vector_store = SimpleVectorStore(config.vector_store)
        
        # Knowledge base
        self.documents = []
        print("🤖 RAG System initialized")
    
    def add_documents(self, texts):
        """Add documents to the knowledge base."""
        print(f"📚 Adding {len(texts)} documents to knowledge base...")
        
        for i, text in enumerate(texts):
            try:
                # Create embedding
                embedding = self.embedding_client.get_embedding(text)
                
                # Add to vector store
                doc_id = self.vector_store.add_document(
                    text=text,
                    embedding=embedding,
                    metadata={"source": f"doc_{i+1}", "length": len(text)}
                )
                
                self.documents.append(text)
                print(f"  ✅ Document {i+1} added")
                
            except Exception as e:
                print(f"  ❌ Document {i+1} failed: {e}")
        
        print(f"📊 Knowledge base now contains {len(self.documents)} documents")
    
    def search(self, query, top_k=3):
        """Search for relevant documents."""
        try:
            # Create query embedding
            query_embedding = self.embedding_client.get_embedding(query)
            
            # Search vector store
            results = self.vector_store.similarity_search(query_embedding, k=top_k)
            
            return results
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []
    
    def ask_question(self, question):
        """Ask a question and get relevant context."""
        print(f"\n❓ Question: {question}")
        print("-" * 50)
        
        # Search for relevant documents
        results = self.search(question, top_k=3)
        
        if not results:
            print("❌ No relevant documents found")
            return None
        
        # Display results
        print("🔍 Relevant documents found:")
        context_parts = []
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"  {i}. [Score: {score:.3f}] {doc.text[:100]}...")
            context_parts.append(f"Document {i}: {doc.text}")
        
        # Combine context
        context = "\n\n".join(context_parts)
        
        return context
    
    def interactive_qa(self):
        """Interactive Q&A session."""
        print("\n🎯 Interactive Q&A Session")
        print("=" * 40)
        print("Type 'quit' to exit")
        print()
        
        while True:
            question = input("❓ Ask a question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            # Get relevant context
            context = self.ask_question(question)
            
            if context:
                print(f"\n📝 Context for answering:")
                print("-" * 30)
                print(context[:300] + "..." if len(context) > 300 else context)
                print("\n💡 You can now use this context with any LLM to generate a complete answer!")

def main():
    """Run the RAG demo."""
    print("🚀 Simple RAG System Demo with Cloudera Embeddings")
    print("=" * 55)
    
    # Initialize RAG system
    rag = SimpleRAGSystem()
    
    # Sample knowledge base about Cloudera
    knowledge_base = [
        "Cloudera Data Platform (CDP) is a unified hybrid data platform that combines analytics and machine learning capabilities.",
        "CDP provides enterprise-grade security and governance for data across on-premises and cloud environments.",
        "Machine learning workloads can be deployed on Cloudera using Cloudera Machine Learning (CML) component.",
        "CML supports popular ML frameworks including TensorFlow, PyTorch, and Scikit-learn.",
        "Data scientists can use CML for model training, experimentation, and deployment in enterprise environments.",
        "The platform supports real-time data processing through Apache Spark and Flink integrations.",
        "Cloudera offers both self-service and managed deployment options for flexibility.",
        "Data governance features include data cataloging, lineage tracking, and access control.",
        "The platform integrates with major cloud providers like AWS, Azure, and Google Cloud.",
        "Cloudera's streaming analytics capabilities enable real-time data processing and insights."
    ]
    
    # Add documents to knowledge base
    rag.add_documents(knowledge_base)
    
    print("\n" + "=" * 55)
    print("🎯 Sample Questions to Try:")
    print("  • What machine learning capabilities does Cloudera offer?")
    print("  • How does Cloudera handle data security?")
    print("  • What cloud platforms does Cloudera support?")
    print("  • What tools are available for data scientists?")
    print()
    
    # Interactive Q&A
    rag.interactive_qa()

if __name__ == "__main__":
    main()
