# Testing Guide

This document provides a comprehensive guide to testing the Cloudera Inference With Windsurf project. It covers unit testing, integration testing, and best practices for writing and running tests.

## Table of Contents

- [Testing Guide](#testing-guide)
  - [Table of Contents](#table-of-contents)
  - [Test Structure](#test-structure)
  - [Running Tests](#running-tests)
  - [Test Examples](#test-examples)
    - [Unit Testing](#unit-testing)
    - [Integration Testing](#integration-testing)
  - [Test Fixtures](#test-fixtures)
  - [Mocking External Services](#mocking-external-services)
  - [Best Practices](#best-practices)
  - [Troubleshooting](#troubleshooting)

## Test Structure

The test suite is organized in the `tests/` directory, mirroring the structure of the main package. The main test files are:

- `test_agent.py`: Tests for the main agent functionality
- `test_embedding_client.py`: Tests for the embedding client
- `test_llm_client.py`: Tests for the LLM client
- `test_vector_store.py`: Tests for the vector store implementation

## Running Tests

To run the entire test suite:

```bash
pytest tests/
```

To run a specific test file:

```bash
pytest tests/test_agent.py
```

To run a specific test function:

```bash
pytest tests/test_agent.py::test_rag_query -v
```

For more verbose output:

```bash
pytest -v
```

To run tests with coverage report:

```bash
pytest --cov=windsurf_agent tests/
```

## Test Examples

### Unit Testing

Unit tests focus on testing individual components in isolation. Here are some examples:

#### Testing the Embedding Client

```python
def test_get_embedding_success(embedding_config, mock_embedding_response):
    with responses.RequestsMock() as rsps:
        rsps.add(
            responses.POST,
            "https://test-embedding.windsurf.ai/v1/embeddings",
            json=mock_embedding_response,
            status=200
        )
        
        client = WindsurfEmbeddingClient(embedding_config)
        embedding = client.get_embedding("test text")
        
        assert len(embedding) == 5
        assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
```

#### Testing the LLM Client

```python
def test_chat_success(llm_config):
    mock_response = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "This is a test chat response."
            }
        }]
    }
    
    with responses.RequestsMock() as rsps:
        rsps.add(
            responses.POST,
            "https://test-llm.windsurf.ai/v1/chat/completions",
            json=mock_response,
            status=200
        )
        
        client = WindsurfLLMClient(llm_config)
        response = client.chat([{"role": "user", "content": "Hello!"}])
        
        assert response == "This is a test chat response."
```

### Integration Testing

Integration tests verify that different components work together correctly:

```python
def test_rag_query(mock_config):
    with patch('windsurf_agent.agent.WindsurfEmbeddingClient') as mock_embedding, \
         patch('windsurf_agent.agent.WindsurfLLMClient') as mock_llm, \
         patch('windsurf_agent.agent.SimpleVectorStore') as mock_store:
        
        # Setup mocks
        mock_llm.return_value.chat.return_value = "Test response"
        mock_store.return_value.similarity_search.return_value = [
            {"text": "Test document", "score": 0.9}
        ]
        
        agent = WindsurfAgent(config=mock_config)
        result = agent.rag_query("Test query")
        
        assert result["answer"] == "Test response"
        assert len(result["sources"]) == 1
        assert result["sources"][0]["text"] == "Test document"
```

## Test Fixtures

Pytest fixtures are used to set up test data and dependencies. Key fixtures include:

### Embedding Client Fixture

```python
@pytest.fixture(scope="module")
def embedding_config():
    return EmbeddingConfig(
        base_url=os.getenv("WINDSURF_EMBEDDING_BASE_URL", "https://test-embedding.windsurf.ai/v1"),
        api_key=os.getenv("WINDSURF_EMBEDDING_API_KEY", "test_api_key"),
        model=os.getenv("WINDSURF_EMBEDDING_MODEL", "test-embedding-model"),
        timeout=int(os.getenv("WINDSURF_EMBEDDING_TIMEOUT", "5")),
        max_retries=int(os.getenv("WINDSURF_EMBEDDING_MAX_RETRIES", "3"))
    )
```

### LLM Client Fixture

```python
@pytest.fixture
def llm_config():
    return LLMConfig(
        base_url="https://test-llm.windsurf.ai/v1",
        api_key="test_api_key",
        model="test-llm-model",
        temperature=0.7,
        max_tokens=100,
        timeout=30,
        max_retries=3
    )
```

## Mocking External Services

Use the `responses` library to mock HTTP requests:

```python
@responses.activate
def test_get_embeddings_success(embedding_config):
    mock_response = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3], "index": 0},
            {"embedding": [0.4, 0.5, 0.6], "index": 1}
        ]
    }
    
    responses.add(
        responses.POST,
        "https://test-embedding.windsurf.ai/v1/embeddings",
        json=mock_response,
        status=200
    )
    
    client = WindsurfEmbeddingClient(embedding_config)
    embeddings = client.get_embeddings(["text1", "text2"])
    
    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert embeddings[1] == [0.4, 0.5, 0.6]
```

## Best Practices

1. **Isolate Tests**: Each test should be independent and not rely on the state from other tests.
2. **Use Fixtures**: Share common test setup code using fixtures.
3. **Mock External Services**: Always mock external API calls to make tests reliable and fast.
4. **Test Edge Cases**: Include tests for error conditions and edge cases.
5. **Keep Tests Focused**: Each test should verify a single piece of functionality.
6. **Use Descriptive Names**: Test names should clearly describe what they're testing.
7. **Test Coverage**: Aim for high test coverage, especially for critical paths.

## Troubleshooting

### Common Issues

1. **Authentication Errors**: Ensure your test environment has the required API keys set.
2. **Network Issues**: Make sure to mock all external API calls in tests.
3. **Test Isolation**: If tests are interfering with each other, check for shared state between tests.

### Debugging Tests

To debug a failing test, you can use `pdb`:

```python
import pdb; pdb.set_trace()
```

Or run pytest with the `--pdb` flag to drop into the debugger on failure:

```bash
pytest --pdb tests/test_agent.py::test_failing_test
```

For more information, refer to the [pytest documentation](https://docs.pytest.org/).
