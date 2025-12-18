# Examples Guide

This guide provides detailed documentation for the example scripts included in the `windsurf_agent/examples/` directory. These examples demonstrate how to use the Windsurf Agent for various tasks.

## Table of Contents

- [Basic Usage](#basic-usage)
- [Advanced RAG Example](#advanced-rag-example)
- [Chat Bot](#chat-bot)
- [LLM Example](#llm-example)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Basic Usage

File: `basic_usage.py`

This example shows the most straightforward way to use the Windsurf Agent for RAG (Retrieval-Augmented Generation) tasks.

### Features Demonstrated:
- Initializing the agent
- Adding documents to the knowledge base
- Querying the knowledge base
- Viewing results with sources

### Code Walkthrough:

```python
from windsurf_agent.agent import WindsurfAgent

# Initialize the agent
agent = WindsurfAgent()

# Add documents to the knowledge base
documents = [
    "The capital of France is Paris.",
    "The Eiffel Tower is located in Paris.",
    "France is known for its wine and cheese.",
    "The official language of France is French."
]
agent.add_to_knowledge_base(documents)

# Query the knowledge base
question = "What is the capital of France?"
result = agent.rag_query(question)

# Display results
print(f"Question: {question}")
print(f"Answer: {result['answer']}")
print("\nSources:")
for i, source in enumerate(result["sources"], 1):
    print(f"{i}. {source['text']} (Score: {source['score']:.3f})")
```

### Running the Example:

```bash
python windsurf_agent/examples/basic_usage.py
```

## Advanced RAG Example

File: `advanced_rag_example.py`

This example demonstrates more advanced usage of the RAG pipeline with custom configuration and document processing.

### Features Demonstrated:
- Custom agent configuration
- Document chunking with overlap
- Advanced querying with metadata filtering
- Custom similarity search parameters

### Key Configuration Options:

```python
config = {
    "chunk_size": 500,          # Characters per chunk
    "chunk_overlap": 50,        # Overlap between chunks
    "similarity_top_k": 3,      # Number of similar documents to retrieve
    "llm": {
        "model": "windsurf-llm-v1",
        "temperature": 0.7,
        "max_tokens": 200
    },
    "embedding": {
        "model": "windsurf-embedding-v1"
    }
}
```

### Running the Example:

```bash
python windsurf_agent/examples/advanced_rag_example.py
```

## Chat Bot

File: `chat_bot.py`

This example demonstrates how to build an interactive chat interface using the Windsurf Agent.

### Features:
- Interactive chat interface
- Conversation history
- Context-aware responses
- Configurable system prompts

### Example Usage:

```bash
python windsurf_agent/examples/chat_bot.py
```

Example interaction:
```
> What can you tell me about France?
[Agent] France is a country in Western Europe known for its rich history, culture, and cuisine...

> What's the capital?
[Agent] The capital of France is Paris, which is also known as the "City of Light"...
```

## LLM Example

File: `llm_example.py`

This example demonstrates direct usage of the LLM client for text generation tasks.

### Features Demonstrated:
- Text completion
- Chat completion
- Streaming responses
- Custom model parameters

### Example Code Snippets:

```python
from windsurf_agent.llm_client import LLMClient

# Initialize the client
client = LLMClient()

# Text completion
response = client.complete("Once upon a time")
print(response)

# Chat completion
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me a joke about programming."}
]
response = client.chat(messages)
print(response)
```

### Running the Example:

```bash
python windsurf_agent/examples/llm_example.py
```

## Configuration

All examples can be configured using environment variables or a `.env` file. Here are the common configuration options:

```env
# Required API configuration
WINDSURF_API_KEY=your_api_key
WINDSURF_API_BASE_URL=https://api.windsurf.ai/v1

# Optional overrides
WINDSURF_LLM_MODEL=windsurf-llm-v1
WINDSURF_EMBEDDING_MODEL=windsurf-embedding-v1
WINDSURF_CHUNK_SIZE=500
WINDSURF_CHUNK_OVERLAP=50
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Ensure `WINDSURF_API_KEY` is set correctly
   - Verify the API key has the necessary permissions

2. **Connection Issues**
   - Check your internet connection
   - Verify the API base URL is correct
   - Check for any firewall or proxy settings

3. **Model Not Found**
   - Verify the model name is correct
   - Check if the model is available in your subscription

4. **Memory Issues**
   - For large documents, reduce `chunk_size`
   - Limit the number of documents in the knowledge base

### Getting Help

For additional support, please refer to:
- [API Documentation](https://docs.windsurf.ai)
- [GitHub Issues](https://github.com/yourorg/windsurf-agent/issues)
- Support: support@windsurf.ai
