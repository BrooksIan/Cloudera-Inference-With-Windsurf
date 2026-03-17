# Example: windsurf_agent/agents/code_agent.py
from typing import Dict, Any
from ..base_agent import BaseAgent

class CodeAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        # Additional initialization specific to CodeAgent
    
    def generate_code(self, requirements: str) -> str:
        prompt = f"Generate Python code that meets these requirements:\n{requirements}"
        return self.generate_response(prompt, temperature=0.2)