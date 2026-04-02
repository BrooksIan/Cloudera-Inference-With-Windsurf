# windsurf_agent/__init__.py
from .base_agent import BaseAgent
from .agent_factory import AgentFactory
from .ClouderaLLMClient import ClouderaLLMClient

# Automatically enforce Cloudera-only models when the package is imported
try:
    import sys
    from pathlib import Path
    
    # Add project root to path for importing scratch modules
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from scratch.cloudera_config import ClouderaConfig, ClouderaLLMConfig
    from . import config as config_module
    
    # Monkey patch to enforce Cloudera-only models
    config_module.Config = ClouderaConfig
    config_module.LLMConfig = ClouderaLLMConfig
    
except ImportError:
    # Silently fail if Cloudera config is not available
    pass
except Exception:
    # Silently fail on other errors during initialization
    pass

__all__ = ['BaseAgent', 'AgentFactory', 'ClouderaLLMClient']
