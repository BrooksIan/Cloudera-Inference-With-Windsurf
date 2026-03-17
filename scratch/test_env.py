#!/usr/bin/env python3
"""Test script to verify environment variables are loaded correctly."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from project root .env file
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"✅ Loaded environment from {env_path}")

# Check the vector store dimension
dimension = os.getenv("WINDSURF_VECTOR_STORE_DIMENSION", "not_set")
print(f"WINDSURF_VECTOR_STORE_DIMENSION: {dimension}")

# Test creating a vector store config
from windsurf_agent.config import Config, VectorStoreConfig

try:
    config = Config.from_env()
    print(f"✅ Config loaded successfully")
    print(f"Vector store dimension: {config.vector_store.dimension}")
except Exception as e:
    print(f"❌ Config failed: {e}")
