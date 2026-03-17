# Cloudera AI Models: Complete Demo Examples

*This document contains detailed, runnable demo scripts for Cloudera AI models with Cascade. For the main tutorial and concepts, see the technical blog.*

---

## 🚀 Demo 1: Semantic Search with Embeddings

### Overview
Demonstrates how to build a semantic search engine that understands meaning beyond keywords using Cloudera's embedding model.

### Code

```python
# scratch/embeddings_demo.py
import os
from pathlib import Path
from dotenv import load_dotenv
from windsurf_agent.config import Config
from windsurf_agent.embedding_client import WindsurfEmbeddingClient
import numpy as np

def semantic_search_demo():
    """Demonstrate semantic search capabilities."""
    
    # Load configuration
    config = Config.from_env()
    embedding_client = WindsurfEmbeddingClient(config.embedding)
    
    # Sample documents
    documents = [
        "Cloudera Data Platform (CDP) is a unified hybrid data platform",
        "Machine learning workloads can be deployed on Cloudera's platform",
        "Data scientists use Cloudera Machine Learning for model development",
        "The platform supports real-time data processing through Apache Spark",
        "Cloudera offers both on-premises and cloud deployment options"
    ]
    
    # Create embeddings for all documents
    print("🔍 Creating document embeddings...")
    document_embeddings = []
    for doc in documents:
        embedding = embedding_client.get_embedding(doc)
        document_embeddings.append(embedding)
    
    # Test semantic search
    queries = [
        "What machine learning capabilities does Cloudera offer?",
        "How does Cloudera handle cloud deployments?",
        "What data processing features are available?"
    ]
    
    for query in queries:
        query_embedding = embedding_client.get_embedding(query)
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(document_embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, similarity, documents[i]))
        
        # Sort and display results
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nQuery: {query}")
        print("Top results:")
        for rank, (idx, score, doc) in enumerate(similarities[:3], 1):
            print(f"  {rank}. [Score: {score:.3f}] {doc[:80]}...")

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

if __name__ == "__main__":
    semantic_search_demo()
```

### Running the Demo

```bash
python scratch/embeddings_demo.py
```

### Expected Output

```
🔍 Creating document embeddings...
✅ Document 1: Embedded (dimension: 1024)
✅ Document 2: Embedded (dimension: 1024)
...

Query: What machine learning capabilities does Cloudera offer?
Top results:
  1. [Score: 0.838] Machine learning workloads can be deployed on Cloudera's platform...
  2. [Score: 0.821] Data scientists use Cloudera Machine Learning for model development...
  3. [Score: 0.779] The platform supports real-time data processing through Apache Spark...
```

---

## 🤖 Demo 2: Interactive RAG System

### Overview
Build a complete Retrieval-Augmented Generation system that can answer questions using your knowledge base.

### Code

```python
# scratch/simple_rag_demo.py
import os
from pathlib import Path
from dotenv import load_dotenv
from windsurf_agent.config import Config
from windsurf_agent.embedding_client import WindsurfEmbeddingClient
from windsurf_agent.vector_store import SimpleVectorStore, Document
import numpy as np

class SimpleRAGSystem:
    """A complete RAG system using Cloudera embeddings and LLM."""
    
    def __init__(self):
        config = Config.from_env()
        self.embedding_client = WindsurfEmbeddingClient(config.embedding)
        self.vector_store = SimpleVectorStore(config.vector_store)
        self.documents = []
    
    def add_documents(self, texts):
        """Add documents to the knowledge base."""
        print(f"📚 Adding {len(texts)} documents to knowledge base...")
        
        for i, text in enumerate(texts):
            embedding = self.embedding_client.get_embedding(text)
            doc_id = self.vector_store.add_document(
                text=text,
                embedding=embedding,
                metadata={"source": f"doc_{i+1}"}
            )
            self.documents.append(text)
            print(f"  ✅ Document {i+1} added")
    
    def search(self, query, top_k=3):
        """Search for relevant documents."""
        query_embedding = self.embedding_client.get_embedding(query)
        results = self.vector_store.similarity_search(query_embedding, k=top_k)
        return results
    
    def ask_question(self, question):
        """Ask a question and get relevant context."""
        print(f"\n❓ Question: {question}")
        print("-" * 50)
        
        results = self.search(question, top_k=3)
        
        if not results:
            print("❌ No relevant documents found")
            return None
        
        print("🔍 Relevant documents found:")
        context_parts = []
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"  {i}. [Score: {score:.3f}] {doc.text[:100]}...")
            context_parts.append(f"Document {i}: {doc.text}")
        
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
    
    rag = SimpleRAGSystem()
    
    # Sample knowledge base
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
    
    rag.add_documents(knowledge_base)
    
    print("\n" + "=" * 55)
    print("🎯 Sample Questions to Try:")
    print("  • What machine learning capabilities does Cloudera offer?")
    print("  • How does Cloudera handle data security?")
    print("  • What cloud platforms does Cloudera support?")
    print("  • What tools are available for data scientists?")
    print()
    
    rag.interactive_qa()

if __name__ == "__main__":
    main()
```

### Running the Demo

```bash
python scratch/simple_rag_demo.py
```

### Sample Interaction

```
🚀 Simple RAG System Demo with Cloudera Embeddings
=======================================================
🤖 RAG System initialized
📚 Adding 10 documents to knowledge base...
  ✅ Document 1 added
  ✅ Document 2 added
  ...
📊 Knowledge base now contains 10 documents

=======================================================
🎯 Sample Questions to Try:
  • What machine learning capabilities does Cloudera offer?
  • How does Cloudera handle data security?
  • What cloud platforms does Cloudera support?
  • What tools are available for data scientists?

🎯 Interactive Q&A Session
========================================
Type 'quit' to exit

❓ Ask a question: What cloud platforms does Cloudera support?

❓ Question: What cloud platforms does Cloudera support?
--------------------------------------------------
🔍 Relevant documents found:
  1. [Score: 0.870] The platform integrates with major cloud providers like AWS, Azure, and Google Cloud...
  2. [Score: 0.865] Cloudera offers both self-service and managed deployment options for flexibility...
  3. [Score: 0.852] The platform supports real-time data processing through Apache Spark and Flink integrations...

📝 Context for answering:
------------------------------
Document 1: The platform integrates with major cloud providers like AWS, Azure, and Google Cloud...
Document 2: Cloudera offers both self-service and managed deployment options for flexibility...
Document 3: The platform supports real-time data processing through Apache Spark and Flink integrations...

💡 You can now use this context with any LLM to generate a complete answer!
```

---

## 🎯 Demo 3: Complete AI Application

### Overview
Comprehensive demo showcasing both embeddings and LLM capabilities working together.

### Code

```python
# scratch/cloudera_demo.py
import os
from pathlib import Path
from dotenv import load_dotenv
from windsurf_agent.config import Config
from windsurf_agent.llm_client import WindsurfLLMClient
from windsurf_agent.embedding_client import WindsurfEmbeddingClient
from windsurf_agent.vector_store import SimpleVectorStore, Document

def test_llm_completion():
    """Test the Cloudera LLM model for text completion."""
    print("🤖 Testing Cloudera LLM Model...")
    print("=" * 50)
    
    config = Config.from_env()
    llm_client = WindsurfLLMClient(config.llm)
    
    prompt = "What is machine learning? Explain in simple terms."
    print(f"Prompt: {prompt}")
    print("-" * 30)
    
    try:
        response = llm_client.complete(prompt)
        print(f"Response: {response}")
        print("✅ LLM test successful!")
    except Exception as e:
        print(f"❌ LLM test failed: {e}")

def test_llm_chat():
    """Test the Cloudera LLM model for chat completion."""
    print("💬 Testing Cloudera LLM Chat...")
    print("=" * 50)
    
    config = Config.from_env()
    llm_client = WindsurfLLMClient(config.llm)
    
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

def test_embeddings():
    """Test the Cloudera embedding model."""
    print("🔍 Testing Cloudera Embedding Model...")
    print("=" * 50)
    
    config = Config.from_env()
    embedding_client = WindsurfEmbeddingClient(config.embedding)
    
    text = "Cloudera provides enterprise data platform solutions."
    print(f"Text: {text}")
    print("-" * 30)
    
    try:
        embedding = embedding_client.get_embedding(text)
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        print("✅ Single embedding test successful!")
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")

def test_rag_pipeline():
    """Test a simple RAG pipeline."""
    print("🔗 Testing RAG Pipeline...")
    print("=" * 50)
    
    config = Config.from_env()
    llm_client = WindsurfLLMClient(config.llm)
    embedding_client = WindsurfEmbeddingClient(config.embedding)
    vector_store = SimpleVectorStore(config.vector_store)
    
    # Sample documents
    documents = [
        "Cloudera Data Platform combines analytics and machine learning.",
        "CDP supports both on-premises and cloud deployments.",
        "Machine learning workloads can be deployed using CML."
    ]
    
    print("Creating document embeddings...")
    try:
        for doc in documents:
            embedding = embedding_client.get_embedding(doc)
            vector_store.add_document(text=doc, embedding=embedding)
        
        print("✅ Documents added to vector store")
        
        # Test search and generation
        query = "What deployment options are available?"
        query_embedding = embedding_client.get_embedding(query)
        results = vector_store.similarity_search(query_embedding, k=2)
        
        context = "\n".join([doc.text for doc, _ in results])
        
        rag_prompt = f"""Based on this context, answer the question:

Context:
{context}

Question: {query}

Answer:"""
        
        response = llm_client.complete(rag_prompt)
        print(f"RAG Response: {response}")
        print("✅ RAG pipeline test successful!")
        
    except Exception as e:
        print(f"❌ RAG pipeline test failed: {e}")

def main():
    """Run all demo tests."""
    print("🚀 Cloudera AI Models Demo with Cascade")
    print("=" * 60)
    
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
        return
    
    print("✅ Environment variables configured")
    print()
    
    # Run tests
    test_llm_completion()
    test_llm_chat()
    test_embeddings()
    test_rag_pipeline()
    
    print("\n🎉 Demo completed!")

if __name__ == "__main__":
    main()
```

### Running the Demo

```bash
python scratch/cloudera_demo.py
```

---

## 🔧 Demo 4: AI Code Assistant

### Overview
An AI-powered code assistant that helps developers write, debug, and optimize code using Cloudera models.

### Code

```python
# scratch/code_assistant_demo.py
import os
from pathlib import Path
from dotenv import load_dotenv
from windsurf_agent.config import Config
from windsurf_agent.llm_client import WindsurfLLMClient
from windsurf_agent.embedding_client import WindsurfEmbeddingClient
from windsurf_agent.vector_store import SimpleVectorStore, Document
import ast
import re

class AICodeAssistant:
    """AI-powered code assistant using Cloudera models."""
    
    def __init__(self):
        config = Config.from_env()
        self.llm_client = WindsurfLLMClient(config.llm)
        self.embedding_client = WindsurfEmbeddingClient(config.embedding)
        self.vector_store = SimpleVectorStore(config.vector_store)
        self.code_examples = []
        self.load_code_examples()
    
    def load_code_examples(self):
        """Load code examples into the knowledge base."""
        examples = [
            {
                "title": "Python Function Definition",
                "code": """def calculate_sum(numbers):
    \"\"\"Calculate the sum of a list of numbers.
    
    Args:
        numbers (list): List of numbers to sum
        
    Returns:
        float: Sum of all numbers
    \"\"\"
    total = 0
    for num in numbers:
        total += num
    return total""",
                "description": "Basic function with docstring and type hints"
            },
            {
                "title": "Error Handling Pattern",
                "code": """try:
    result = risky_operation()
    if result is None:
        raise ValueError("Operation returned None")
    return result
except (ValueError, TypeError) as e:
    logger.error(f"Operation failed: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise""",
                "description": "Comprehensive error handling with logging"
            },
            {
                "title": "Class Definition",
                "code": """class DataProcessor:
    \"\"\"Process data with validation and error handling.
    
    Attributes:
        data (list): Input data to process
        processed_count (int): Number of processed items
    \"\"\"
    
    def __init__(self, data):
        self.data = data
        self.processed_count = 0
        self.validate_data()
    
    def validate_data(self):
        \"\"\"Validate input data.\"\"\"
        if not isinstance(self.data, list):
            raise TypeError("Data must be a list")
        if len(self.data) == 0:
            raise ValueError("Data cannot be empty")
    
    def process(self):
        \"\"\"Process all data items.\"\"\"
        results = []
        for item in self.data:
            try:
                processed = self._process_item(item)
                results.append(processed)
                self.processed_count += 1
            except Exception as e:
                logger.warning(f"Failed to process item {item}: {e}")
        return results""",
                "description": "Class with initialization, validation, and processing"
            }
        ]
        
        print("📚 Loading code examples into knowledge base...")
        for example in examples:
            # Create searchable text from code and description
            searchable_text = f"{example['title']}: {example['description']}\n\n{example['code']}"
            embedding = self.embedding_client.get_embedding(searchable_text)
            
            self.vector_store.add_document(
                text=searchable_text,
                embedding=embedding,
                metadata={
                    "title": example["title"],
                    "code": example["code"],
                    "description": example["description"]
                }
            )
            self.code_examples.append(example)
        
        print(f"✅ Loaded {len(examples)} code examples")
    
    def suggest_code_improvement(self, code_snippet):
        """Suggest improvements for a given code snippet."""
        print(f"🔍 Analyzing code for improvements...")
        
        # Create embedding for the input code
        code_embedding = self.embedding_client.get_embedding(code_snippet)
        
        # Find similar code examples
        results = self.vector_store.similarity_search(code_embedding, k=3)
        
        # Build context from similar examples
        context_parts = []
        for doc, score in results:
            context_parts.append(f"Similar Pattern ({score:.3f}):\n{doc.text}")
        
        context = "\n\n".join(context_parts)
        
        # Generate improvement suggestions
        prompt = f"""As an expert Python developer, analyze this code and suggest improvements:

Code to analyze:
```python
{code_snippet}
```

Reference patterns:
{context}

Provide specific, actionable suggestions for:
1. Code structure and organization
2. Error handling
3. Performance optimization
4. Best practices
5. Documentation

Format your response with clear sections and code examples where helpful."""
        
        try:
            response = self.llm_client.chat([
                {"role": "system", "content": "You are an expert Python code reviewer and mentor."},
                {"role": "user", "content": prompt}
            ])
            
            return response
        except Exception as e:
            return f"❌ Failed to generate suggestions: {e}"
    
    def generate_code_from_description(self, description):
        """Generate code from natural language description."""
        print(f"🤖 Generating code from description...")
        
        # Find relevant examples
        desc_embedding = self.embedding_client.get_embedding(description)
        results = self.vector_store.similarity_search(desc_embedding, k=2)
        
        context_parts = []
        for doc, score in results:
            context_parts.append(f"Reference Example ({score:.3f}):\n{doc.metadata.get('code', '')}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Generate Python code based on this description:

Description: {description}

Reference examples:
{context}

Requirements:
1. Write clean, production-ready code
2. Include proper error handling
3. Add docstrings and type hints
4. Follow PEP 8 style guidelines
5. Include example usage

Provide only the code without explanation:"""
        
        try:
            response = self.llm_client.chat([
                {"role": "system", "content": "You are an expert Python developer who writes clean, production-ready code."},
                {"role": "user", "content": prompt}
            ])
            
            return response
        except Exception as e:
            return f"❌ Failed to generate code: {e}"
    
    def debug_code(self, code_snippet, error_message):
        """Help debug code with error analysis."""
        print(f"🐛 Analyzing code for debugging...")
        
        prompt = f"""Help debug this Python code:

Code:
```python
{code_snippet}
```

Error: {error_message}

Analyze the code and:
1. Identify the root cause of the error
2. Explain what went wrong
3. Provide the corrected code
4. Suggest preventive measures

Be thorough and educational in your explanation."""
        
        try:
            response = self.llm_client.chat([
                {"role": "system", "content": "You are an expert Python debugger who helps developers understand and fix their code."},
                {"role": "user", "content": prompt}
            ])
            
            return response
        except Exception as e:
            return f"❌ Failed to debug code: {e}"

def main():
    """Run the AI code assistant demo."""
    print("🚀 AI Code Assistant Demo with Cloudera Models")
    print("=" * 55)
    
    assistant = AICodeAssistant()
    
    print("\n" + "=" * 55)
    print("🎯 Available Features:")
    print("  1. Code improvement suggestions")
    print("  2. Code generation from description")
    print("  3. Debug assistance")
    print("  4. Interactive mode")
    print()
    
    # Demo 1: Code improvement
    print("📝 Demo 1: Code Improvement")
    print("-" * 30)
    sample_code = """def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result"""
    
    print("Original code:")
    print(sample_code)
    print("\nAI Suggestions:")
    suggestions = assistant.suggest_code_improvement(sample_code)
    print(suggestions)
    
    # Demo 2: Code generation
    print("\n📝 Demo 2: Code Generation")
    print("-" * 30)
    description = "Create a function that reads a CSV file, validates the data, and returns a pandas DataFrame"
    print(f"Description: {description}")
    print("\nGenerated Code:")
    generated_code = assistant.generate_code_from_description(description)
    print(generated_code)
    
    # Demo 3: Debug assistance
    print("\n📝 Demo 3: Debug Assistance")
    print("-" * 30)
    buggy_code = """def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)"""
    
    error = "ZeroDivisionError: division by zero"
    print("Buggy code:")
    print(buggy_code)
    print(f"Error: {error}")
    print("\nDebug Analysis:")
    debug_help = assistant.debug_code(buggy_code, error)
    print(debug_help)
    
    print("\n🎉 Code Assistant Demo completed!")

if __name__ == "__main__":
    main()
```

### Running the Demo

```bash
python scratch/code_assistant_demo.py
```

---

## 🧪 Demo 5: Testing and Validation

### Overview
Comprehensive testing examples for Cloudera AI integration.

### Code

```python
# tests/test_cloudera_integration.py
import pytest
import numpy as np
from windsurf_agent.config import Config
from windsurf_agent.embedding_client import WindsurfEmbeddingClient
from windsurf_agent.llm_client import WindsurfLLMClient
from windsurf_agent.vector_store import SimpleVectorStore

class TestClouderaIntegration:
    """Integration tests for Cloudera AI models."""
    
    @pytest.fixture
    def config(self):
        return Config.from_env()
    
    @pytest.fixture
    def embedding_client(self, config):
        return WindsurfEmbeddingClient(config.embedding)
    
    @pytest.fixture
    def llm_client(self, config):
        return WindsurfLLMClient(config.llm)
    
    @pytest.fixture
    def vector_store(self, config):
        return SimpleVectorStore(config.vector_store)
    
    def test_embedding_generation(self, embedding_client):
        """Test embedding generation with real Cloudera model."""
        text = "This is a test for Cloudera embedding model."
        
        embedding = embedding_client.get_embedding(text)
        
        # Verify embedding properties
        assert isinstance(embedding, list)
        assert len(embedding) == 1024
        assert all(isinstance(x, float) for x in embedding)
        assert all(-1 <= x <= 1 for x in embedding)  # Normalized embeddings
    
    def test_batch_embeddings(self, embedding_client):
        """Test batch embedding generation."""
        texts = [
            "First test document",
            "Second test document", 
            "Third test document"
        ]
        
        embeddings = embedding_client.get_embeddings(texts)
        
        assert len(embeddings) == len(texts)
        for embedding in embeddings:
            assert len(embedding) == 1024
            assert all(isinstance(x, float) for x in embedding)
    
    def test_llm_completion(self, llm_client):
        """Test LLM text completion."""
        prompt = "What is machine learning? Explain in one sentence."
        
        response = llm_client.complete(prompt)
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert len(response) < 1000  # Reasonable length
    
    def test_llm_chat(self, llm_client):
        """Test LLM chat completion."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ]
        
        response = llm_client.chat(messages)
        
        assert isinstance(response, str)
        assert len(response) > 0
        # Should contain the answer "4" or similar
        assert any(num in response.lower() for num in ["4", "four"])
    
    def test_vector_store_operations(self, vector_store, embedding_client):
        """Test vector store operations."""
        # Add documents
        texts = [
            "Document about machine learning",
            "Document about data science",
            "Document about artificial intelligence"
        ]
        
        doc_ids = []
        for text in texts:
            embedding = embedding_client.get_embedding(text)
            doc_id = vector_store.add_document(text, embedding)
            doc_ids.append(doc_id)
        
        assert len(doc_ids) == len(texts)
        assert len(vector_store) == len(texts)
        
        # Test similarity search
        query = "machine learning algorithms"
        query_embedding = embedding_client.get_embedding(query)
        results = vector_store.similarity_search(query_embedding, k=2)
        
        assert len(results) == 2
        # First result should be most relevant
        assert "machine learning" in results[0][0].text.lower()
    
    def test_rag_pipeline(self, embedding_client, llm_client, vector_store):
        """Test complete RAG pipeline."""
        # Knowledge base
        documents = [
            "Cloudera Data Platform supports hybrid cloud deployments.",
            "Machine learning can be deployed using Cloudera ML.",
            "Data governance is built into the Cloudera platform."
        ]
        
        # Add to vector store
        for doc in documents:
            embedding = embedding_client.get_embedding(doc)
            vector_store.add_document(doc, embedding)
        
        # Query and retrieve
        query = "What deployment options are available?"
        query_embedding = embedding_client.get_embedding(query)
        results = vector_store.similarity_search(query_embedding, k=2)
        
        # Generate response
        context = "\n".join([doc.text for doc, _ in results])
        prompt = f"Based on this context: {context}\n\nAnswer: {query}"
        
        response = llm_client.complete(prompt)
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert "cloud" in response.lower() or "deployment" in response.lower()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Running Tests

```bash
python -m pytest tests/test_cloudera_integration.py -v
```

---

## 📊 Performance Benchmarks

### Code

```python
# scratch/performance_benchmarks.py
import time
import statistics
from windsurf_agent.config import Config
from windsurf_agent.embedding_client import WindsurfEmbeddingClient
from windsurf_agent.llm_client import WindsurfLLMClient

class PerformanceBenchmarks:
    """Benchmark Cloudera AI model performance."""
    
    def __init__(self):
        config = Config.from_env()
        self.embedding_client = WindsurfEmbeddingClient(config.embedding)
        self.llm_client = WindsurfLLMClient(config.llm)
    
    def benchmark_embeddings(self, num_requests=10):
        """Benchmark embedding generation performance."""
        print(f"🔍 Benchmarking Embedding Generation ({num_requests} requests)")
        print("-" * 50)
        
        texts = [f"Test document {i}" for i in range(num_requests)]
        times = []
        
        for text in texts:
            start_time = time.time()
            embedding = self.embedding_client.get_embedding(text)
            end_time = time.time()
            
            times.append(end_time - start_time)
            print(f"Request {len(times)}: {times[-1]:.3f}s")
        
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n📊 Embedding Performance:")
        print(f"  Average: {avg_time:.3f}s")
        print(f"  Min: {min_time:.3f}s")
        print(f"  Max: {max_time:.3f}s")
        print(f"  Throughput: {num_requests/sum(times):.2f} requests/second")
        
        return times
    
    def benchmark_batch_embeddings(self, batch_sizes=[1, 5, 10, 20]):
        """Benchmark batch embedding performance."""
        print(f"🔍 Benchmarking Batch Embeddings")
        print("-" * 50)
        
        for batch_size in batch_sizes:
            texts = [f"Batch test {i}" for i in range(batch_size)]
            
            start_time = time.time()
            embeddings = self.embedding_client.get_embeddings(texts)
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_per_item = total_time / batch_size
            
            print(f"Batch Size {batch_size}: {total_time:.3f}s total, {avg_per_item:.3f}s per item")
    
    def benchmark_llm_completion(self, num_requests=5):
        """Benchmark LLM completion performance."""
        print(f"🤖 Benchmarking LLM Completion ({num_requests} requests)")
        print("-" * 50)
        
        prompts = [f"Explain concept {i} in one sentence." for i in range(num_requests)]
        times = []
        
        for prompt in prompts:
            start_time = time.time()
            response = self.llm_client.complete(prompt)
            end_time = time.time()
            
            times.append(end_time - start_time)
            print(f"Request {len(times)}: {times[-1]:.3f}s, Response length: {len(response)} chars")
        
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n📊 LLM Performance:")
        print(f"  Average: {avg_time:.3f}s")
        print(f"  Min: {min_time:.3f}s")
        print(f"  Max: {max_time:.3f}s")
        print(f"  Throughput: {num_requests/sum(times):.2f} requests/second")
        
        return times
    
    def benchmark_llm_chat(self, num_requests=5):
        """Benchmark LLM chat performance."""
        print(f"💬 Benchmarking LLM Chat ({num_requests} requests)")
        print("-" * 50)
        
        messages_list = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"What is the capital of country {i}?"}
            ]
            for i in range(num_requests)
        ]
        
        times = []
        for messages in messages_list:
            start_time = time.time()
            response = self.llm_client.chat(messages)
            end_time = time.time()
            
            times.append(end_time - start_time)
            print(f"Request {len(times)}: {times[-1]:.3f}s, Response length: {len(response)} chars")
        
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n📊 Chat Performance:")
        print(f"  Average: {avg_time:.3f}s")
        print(f"  Min: {min_time:.3f}s")
        print(f"  Max: {max_time:.3f}s")
        print(f"  Throughput: {num_requests/sum(times):.2f} requests/second")
        
        return times
    
    def run_all_benchmarks(self):
        """Run all performance benchmarks."""
        print("🚀 Cloudera AI Performance Benchmarks")
        print("=" * 60)
        
        self.benchmark_embeddings()
        print()
        self.benchmark_batch_embeddings()
        print()
        self.benchmark_llm_completion()
        print()
        self.benchmark_llm_chat()
        
        print("\n🎉 Benchmarks completed!")

if __name__ == "__main__":
    benchmarks = PerformanceBenchmarks()
    benchmarks.run_all_benchmarks()
```

### Running Benchmarks

```bash
python scratch/performance_benchmarks.py
```

---

## 📚 Additional Resources

### Environment Setup Script

```bash
#!/bin/bash
# setup_cloudera_env.sh

echo "🚀 Setting up Cloudera AI Environment"

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env template
cat > .env << EOF
# Cloudera AI Configuration
WINDSURF_EMBEDDING_API_KEY="your-jwt-token"
WINDSURF_EMBEDDING_BASE_URL="https://your-cloudera-embedding-endpoint/v1"
WINDSURF_EMBEDDING_MODEL="nvidia/nv-embedqa-e5-v5"

WINDSURF_LLM_API_KEY="your-jwt-token"
WINDSURF_LLM_BASE_URL="https://your-cloudera-llm-endpoint/v1"
WINDSURF_LLM_MODEL="nvidia/llama-3.3-nemotron-super-49b-v1.5"

WINDSURF_VECTOR_STORE_DIMENSION=1024
EOF

echo "✅ Environment setup complete!"
echo "📝 Please edit .env with your Cloudera credentials"
```

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port for web applications
EXPOSE 8000

# Default command
CMD ["python", "scratch/cloudera_demo.py"]
```

---

*These examples provide a complete foundation for building AI applications with Cloudera models and Cascade. For concepts and architecture, refer to the main technical blog.*
