# Enforcing Cloudera-Hosted LLMs

This guide explains how to configure your environment to ensure that all LLM interactions go through Cloudera-hosted endpoints.

## Overview

The Windsurf agent framework includes a mechanism to enforce that only Cloudera-hosted LLM endpoints can be used. This is particularly important for:

- Air-gapped environments
- Compliance requirements
- Security policies
- Cost control

## Allowed Cloudera Domains

The following domains are whitelisted as valid Cloudera endpoints:

- `cloudera.com`
- `cloudera.site` (for Cloudera Machine Learning deployments)
- `cdp.cloudera.com`
- `cloudera-ml.ai`
- `cloudera-ml.cloud`

## Configuration Steps

### 1. Environment Variables

Create a `.env` file in your project root with these required variables:

```bash
# Required LLM Configuration
WINDSURF_LLM_BASE_URL=https://your-cloudera-ml-endpoint  # Must be a Cloudera-hosted endpoint
WINDSURF_LLM_API_KEY=your-api-key
WINDSURF_LLM_MODEL=nvidia/llama-3.3-nemotron-super-49b-v1  # Default model

# Optional: Additional Configuration
WINDSURF_LLM_TEMPERATURE=0.7
WINDSURF_LLM_MAX_TOKENS=2048
WINDSURF_LLM_TIMEOUT=60
WINDSURF_LLM_MAX_RETRIES=3
```

## Enforcing Cloudera Models

To enforce that only Cloudera-hosted models are used in your application, you need to call `enforce_cloudera_models()` at the start of your application, before any LLM clients are initialized.

### Example Usage

```python
from windsurf_agent.agent import WindsurfAgent
from scratch.cloudera_config import enforce_cloudera_models

# Enforce Cloudera models before initializing any agents
enforce_cloudera_models()

# Now all LLM calls will be forced to use Cloudera endpoints
try:
    agent = WindsurfAgent()
    response = agent.generate("Hello, world!")
    print(response)
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Error Handling

If you attempt to use a non-Cloudera endpoint, the system will raise a `ValueError` with a message indicating that the endpoint must be a Cloudera-hosted service.

## Default Configuration

If no model is specified, the system defaults to using `nvidia/llama-3.3-nemotron-super-49b-v1` as the default model. This can be overridden by setting the `WINDSURF_LLM_MODEL` environment variable.

## Best Practices

1. Always call `enforce_cloudera_models()` at the start of your application
2. Store your API keys securely using environment variables or a secrets manager
3. Handle configuration errors gracefully in your application
4. Monitor your Cloudera endpoint usage and set up alerts for any configuration issues
