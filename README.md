# Cloudera AI Inference Service Client

![Verify](images/Verify.png)
![Cloudera AI Inference Models](images/CAI_Inference_Models.png)

A Python client for interacting with AI models hosted on Cloudera's AI Inference Service. This library provides a clean, Pythonic interface for deploying, managing, and querying models with enterprise-grade security and scalability.

## Features

- **Model Management**: Deploy, update, and manage AI models on Cloudera AI Inference Service
- **Enterprise Security**: Built-in authentication and authorization for secure model access
- **Scalable Inference**: Efficiently handle high-volume inference requests
- **Model Monitoring**: Track performance metrics and usage statistics
- **Multiple Framework Support**: Compatible with popular ML frameworks (PyTorch, TensorFlow, etc.)
- **Example Use Cases**:
  - **RAG Pipelines**: Build Retrieval Augmented Generation systems
  - **Text Generation**: Deploy and query large language models
  - **Embedding Models**: Generate and work with text embeddings
  - **Custom Models**: Deploy your own trained models

![RAG Example](images/rag_example.png)

## Installation

```bash
pip install cloudera-ai-inference
```

## Configuration

1. **Set up credentials**
   Create a `.env` file in your project root:

   ```env
   # Required for Cloudera AI Inference Service
   CLOUDERA_AI_ENDPOINT=https://your-cloudera-ai-instance.cloudera.site
   CLOUDERA_AI_API_KEY=your_api_key_here
   
   # Optional: Default model configurations
   DEFAULT_LLM_MODEL=llama-2-13b-chat
   DEFAULT_EMBEDDING_MODEL=e5-large-v2
   ```

2. **Or configure programmatically**
   ```python
   from cloudera_ai import configure
   
   configure(
       endpoint="https://your-cloudera-ai-instance.cloudera.site",
       api_key="your_api_key_here"
   )
   ```

## Cloudera Integration

For organizations requiring Cloudera-hosted models, the framework provides built-in support for enforcing Cloudera endpoints. This ensures all LLM interactions go through approved Cloudera infrastructure.

### Key Benefits

- Enforce usage of Cloudera-hosted models only
- Simple one-line enforcement
- Automatic validation of endpoints
- Detailed error messages for misconfigurations

![Cloudera AI Inference Models](images/CAI_Inference_Models.png)

See the [Cloudera Integration Guide](docs/cloudera_endpoints.md) for detailed configuration options and examples.

## Quick Start

### Basic Model Inference

```python
from cloudera_ai import InferenceClient

# Initialize the client
client = InferenceClient()

# List available models
models = client.list_models()
print("Available models:", models)

# Get a model instance
llm = client.get_model("llama-2-13b-chat")

# Generate text
response = llm.generate("Explain quantum computing in simple terms")
print(response)
```

### RAG Example

```python
from cloudera_ai import RAGPipeline

# Initialize RAG pipeline with your models
rag = RAGPipeline(
    embedding_model="e5-large-v2",
    llm_model="llama-2-13b-chat"
)

# Add documents to the knowledge base
documents = [
    "Cloudera AI Inference Service provides scalable model serving.",
    "The service supports multiple ML frameworks and custom models.",
    "Enterprise security features include RBAC and data encryption."
]
rag.add_documents(documents)

# Query the knowledge base
response = rag.query("What security features does Cloudera AI offer?")
print("Answer:", response["answer"])
```

![LLM Example](images/llm_example.png)
```

## Advanced Usage

### Using Custom Configuration

You can customize the agent's behavior by passing a configuration dictionary:

```python
from windsurf_agent.agent import WindsurfAgent

config = {
    "embedding_model": "custom-embedding-model",
    "llm_model": "custom-llm-model",
    "similarity_top_k": 5,
    "chunk_size": 1000,
    "chunk_overlap": 200
}

agent = WindsurfAgent(config=config)
```

### Working with Vector Store

The vector store handles document storage and retrieval:

```python
from windsurf_agent.vector_store import VectorStore
from windsurf_agent.embedding_client import EmbeddingClient

# Initialize components
embedding_client = EmbeddingClient()
vector_store = VectorStore(embedding_client=embedding_client)

# Add documents
document_ids = vector_store.add_documents(["Document 1 text", "Document 2 text"])

# Search for similar documents
results = vector_store.similarity_search("search query", k=3)
```

## Configuration

Configuration can be provided through:

1. Environment variables
2. Configuration dictionary
3. `.env` file

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WINDSURF_API_KEY` | API key for Windsurf services | Required |
| `WINDSURF_API_BASE_URL` | Base URL for Windsurf API | Required |
| `EMBEDDING_MODEL` | Name of the embedding model to use | `windsurf-embedding-v1` |
| `LLM_MODEL` | Name of the language model to use | `windsurf-llm-v1` |
| `CHUNK_SIZE` | Size of text chunks for processing | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `SIMILARITY_TOP_K` | Number of similar documents to retrieve | `3` |

## Documentation

For detailed usage examples and guides, see:

- [Examples Guide](docs/examples.md) - Complete documentation of example scripts and usage patterns
- [Testing Guide](docs/testing.md) - Information on running and writing tests

## Testing

For comprehensive testing information, see the [Testing Guide](docs/testing.md).

Run the test suite with:

```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
