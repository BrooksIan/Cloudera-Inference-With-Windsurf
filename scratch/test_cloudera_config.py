"""Tests for the Cloudera configuration enforcement."""
import os
import pytest
from unittest.mock import patch, MagicMock

# Import the module to test
#from scratch.cloudera_config import ClouderaLLMConfig, ClouderaConfig, enforce_cloudera_models
from windsurf_agent.config import LLMConfig, Config
# Update the import line in test_cloudera_config.py
# In test_cloudera_config.py
from .cloudera_config import ClouderaLLMConfig, ClouderaConfig, enforce_cloudera_models


# Test data
VALID_CLOUDERA_URL = "https://ml-64288d82-5dd.go01-dem.ylcu-atmi.cloudera.site/api/v1"
INVALID_URL = "https://api.openai.com/v1"
VALID_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1"

class TestClouderaLLMConfig:
    """Test the ClouderaLLMConfig class."""
    
    def test_valid_cloudera_endpoint(self):
        """Test that valid Cloudera endpoints are accepted."""
        config = ClouderaLLMConfig(
            base_url=VALID_CLOUDERA_URL,
            api_key="test-key",
            model=VALID_MODEL
        )
        assert config.base_url == VALID_CLOUDERA_URL
        assert config.model == VALID_MODEL
    
    def test_invalid_endpoint_raises_error(self):
        """Test that non-Cloudera endpoints raise a ValueError."""
        with pytest.raises(ValueError, match="must be a Cloudera-hosted service"):
            ClouderaLLMConfig(
                base_url=INVALID_URL,
                api_key="test-key",
                model=VALID_MODEL
            )
    
    @pytest.mark.parametrize("domain", [
        "test.cloudera.com",
        "api.cdp.cloudera.com/v1",
        "ml-service.cloudera.site",
        "inference.cloudera-ml.ai"
    ])
    def test_allowed_domains(self, domain):
        """Test that various Cloudera domain formats are accepted."""
        url = f"https://{domain}/path"
        config = ClouderaLLMConfig(
            base_url=url,
            api_key="test-key",
            model=VALID_MODEL
        )
        assert config.base_url == url

class TestClouderaConfig:
    """Test the ClouderaConfig class."""
    
    @patch.dict(os.environ, {
        "WINDSURF_LLM_BASE_URL": VALID_CLOUDERA_URL,
        "WINDSURF_LLM_API_KEY": "test-key",
        "WINDSURF_LLM_MODEL": VALID_MODEL
    })
    def test_from_env_valid(self):
        """Test loading config from environment variables with valid Cloudera URL."""
        config = ClouderaConfig.from_env()
        assert config.llm.base_url == VALID_CLOUDERA_URL
        assert config.llm.model == VALID_MODEL
    
    @patch.dict(os.environ, {
        "WINDSURF_LLM_BASE_URL": INVALID_URL,
        "WINDSURF_LLM_API_KEY": "test-key"
    })
    def test_from_env_invalid_url(self):
        """Test that invalid URLs in environment variables raise an error."""
        with pytest.raises(ValueError, match="must be a Cloudera-hosted service"):
            ClouderaConfig.from_env()
    
    @patch.dict(os.environ, {
        "WINDSURF_LLM_BASE_URL": VALID_CLOUDERA_URL,
        "WINDSURF_LLM_API_KEY": "test-key"
    })
    def test_default_model(self):
        """Test that the default model is set correctly."""
        config = ClouderaConfig.from_env()
        assert "llama-3.3-nemotron" in config.llm.model.lower()

def test_enforce_cloudera_models():
    """Test that the enforce_cloudera_models function monkey patches the config."""
    # Import here to avoid test interference
    import windsurf_agent.config as config_module
    
    # Save original classes
    original_config = config_module.Config
    original_llm_config = config_module.LLMConfig
    
    try:
        # Apply the monkey patch
        enforce_cloudera_models()
        
        # Verify the classes were replaced
        assert config_module.Config is ClouderaConfig
        assert config_module.LLMConfig is ClouderaLLMConfig
        
    finally:
        # Restore original classes
        config_module.Config = original_config
        config_module.LLMConfig = original_llm_config

@pytest.fixture
def mock_config():
    """Fixture to provide a mock config for testing."""
    with patch('windsurf_agent.config.Config') as mock_config_class:
        mock_config = MagicMock()
        mock_config.llm = MagicMock()
        mock_config.llm.base_url = ""
        mock_config.llm.model = ""
        mock_config.embedding = MagicMock()
        mock_config.vector_store = MagicMock()
        mock_config.log_level = "INFO"
        mock_config_class.from_env.return_value = mock_config
        yield mock_config

class TestIntegration:
    """Integration tests for the configuration system."""
    
    @patch.dict(os.environ, {
        "WINDSURF_LLM_BASE_URL": VALID_CLOUDERA_URL,
        "WINDSURF_LLM_API_KEY": "test-key"
    })
    def test_end_to_end_config_loading(self, mock_config):
        """Test that the configuration loads correctly from environment variables."""
        # Apply the monkey patch
        enforce_cloudera_models()
        
        # Import here to get the patched version
        from windsurf_agent.config import Config, LLMConfig
        
        # This should use our ClouderaConfig
        config = Config.from_env()
        
        # Verify the config was created with our values
        assert config.llm.base_url == VALID_CLOUDERA_URL
        assert "llama-3.3-nemotron" in config.llm.model.lower()
        
        # Verify we can't create a non-Cloudera config
        with pytest.raises(ValueError):
            LLMConfig(base_url=INVALID_URL, api_key="test-key")
