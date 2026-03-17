# Cloudera AI Models Demo with Cascade

This repository demonstrates how to use Cloudera's AI models (embeddings and LLM) with the Cascade framework for building intelligent applications.

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Ensure your `.env` file is configured with your Cloudera API credentials:

```bash
# Embedding Configuration
WINDSURF_EMBEDDING_API_KEY="your-cloudera-api-key"
WINDSURF_EMBEDDING_BASE_URL="https://your-cloudera-endpoint/namespaces/serving-default/endpoints/your-embedding-model"
WINDSURF_EMBEDDING_MODEL="nvidia/nv-embedqa-e5-v5"

# LLM Configuration  
WINDSURF_LLM_API_KEY="your-cloudera-api-key"
WINDSURF_LLM_BASE_URL="https://your-cloudera-endpoint/namespaces/serving-default/endpoints/your-llm-model"
WINDSURF_LLM_MODEL="nvidia/llama-3.3-nemotron-super-49b-v1"

# Vector Store Configuration
WINDSURF_VECTOR_STORE_DIMENSION=1024
```

### 3. Run Tests

```bash
pytest tests/ -v
```

## 📚 Demo Scripts

### 1. Embeddings Demo (`scratch/embeddings_demo.py`)

Demonstrates semantic search using Cloudera's embedding model:

```bash
python scratch/embeddings_demo.py
```

**Features:**
- Document embedding and storage
- Semantic search capabilities
- Text similarity analysis
- Cosine similarity calculations

### 2. RAG System Demo (`scratch/simple_rag_demo.py`)

Interactive Q&A system using Retrieval-Augmented Generation:

```bash
python scratch/simple_rag_demo.py
```

**Features:**
- Knowledge base creation
- Interactive Q&A interface
- Context retrieval for questions
- Semantic document search

### 3. Full Demo (`scratch/cloudera_demo.py`)

Comprehensive demo of all Cloudera AI capabilities:

```bash
python scratch/cloudera_demo.py
```

**Features:**
- LLM text completion
- LLM chat completion
- Embedding generation
- RAG pipeline integration

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  RAG System      │───▶│  Cloudera       │
│                 │    │                  │    │  Embeddings     │
└─────────────────┘    │                  │    └─────────────────┘
                       │                                         
                       │                                         
                       ▼                                         
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Context        │◀───│  Vector Store    │◀───│  Document       │
│  Retrieval      │    │                  │    │  Embeddings     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🔧 Key Components

### Embedding Client (`windsurf_agent/embedding_client.py`)

- Handles communication with Cloudera embedding models
- Supports single and batch embedding generation
- Automatic retry logic and error handling
- Compatible with Cloudera ML endpoints

### Vector Store (`windsurf_agent/vector_store.py`)

- Efficient similarity search using FAISS
- Document storage and retrieval
- Configurable similarity metrics
- Persistent storage options

### RAG System (`scratch/simple_rag_demo.py`)

- Combines embeddings and vector search
- Interactive Q&A capabilities
- Context-aware responses
- Extensible knowledge base

## 📊 Performance

### Embedding Model
- **Model**: NVIDIA NV-EmbedQA-E5-V5
- **Dimensions**: 1024
- **Use Case**: Semantic search, document similarity

### Vector Store
- **Index Type**: FAISS IndexFlatIP
- **Similarity Metric**: Cosine Similarity
- **Search Speed**: O(log n) for n documents

## 🎯 Use Cases

### 1. Document Search
```python
# Search documents using semantic similarity
results = rag.search("What machine learning capabilities exist?")
```

### 2. Q&A Systems
```python
# Ask questions and get relevant context
context = rag.ask_question("How does Cloudera handle security?")
```

### 3. Content Recommendation
```python
# Find similar content based on embeddings
similar_docs = vector_store.similarity_search(query_embedding, k=5)
```

## 🔍 Example Output

### Semantic Search Results
```
Query: What machine learning capabilities does Cloudera offer?
Top 3 most relevant documents:
  1. [Score: 0.838] Machine learning workloads can be deployed on Cloudera's platform...
  2. [Score: 0.821] Data scientists use Cloudera Machine Learning for model development...
  3. [Score: 0.779] Real-time analytics is supported through Cloudera's streaming capabilities...
```

### RAG Q&A Results
```
Question: What cloud platforms does Cloudera support?
Relevant documents:
  1. [Score: 0.870] The platform integrates with major cloud providers like AWS, Azure, and Google Cloud...
  2. [Score: 0.865] Cloudera offers both self-service and managed deployment options for flexibility...
  3. [Score: 0.852] The platform supports real-time data processing through Apache Spark and Flink integrations...
```

## 🧪 Testing

Run the complete test suite:

```bash
pytest tests/ -v --cov=windsurf_agent
```

Key test files:
- `tests/test_embedding_client.py` - Embedding client functionality
- `tests/test_vector_store.py` - Vector store operations
- `tests/test_llm_client.py` - LLM client functionality

## 🚧 Troubleshooting

### Common Issues

1. **404 Errors on LLM Endpoints**
   - Check if the base URL is correct (remove `/v1` suffix)
   - Verify the endpoint is accessible

2. **Dimension Mismatch**
   - Ensure `WINDSURF_VECTOR_STORE_DIMENSION` matches embedding output
   - Current embeddings output 1024 dimensions

3. **API Key Issues**
   - Verify JWT tokens are not expired
   - Check environment variable loading

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔮 Future Enhancements

- [ ] LLM integration for complete RAG responses
- [ ] Streaming chat capabilities
- [ ] Document chunking for large texts
- [ ] Multi-modal embeddings
- [ ] Advanced filtering options

## 📝 License

This project is part of the Cloudera Inference with Windsurf demonstration.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!
