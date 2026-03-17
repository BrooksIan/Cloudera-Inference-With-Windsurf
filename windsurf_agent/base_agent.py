# windsurf_agent/base_agent.py
from typing import Dict, Any, Optional
from .ClouderaLLMClient import ClouderaLLMClient

class BaseAgent:
    """Base agent class that enforces Cloudera LLM usage for all agents."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.llm = ClouderaLLMClient()  # Enforces Cloudera LLM usage
        self.initialize()
    
    def initialize(self):
        """Override this method in child classes for custom initialization."""
        pass
    
    def process(self, *args, **kwargs):
        """Main processing method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement process()")
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Standard method for generating responses using Cloudera LLM."""
        messages = [{"role": "user", "content": prompt}]
        return "".join(self.llm.chat_completion(messages, **kwargs))