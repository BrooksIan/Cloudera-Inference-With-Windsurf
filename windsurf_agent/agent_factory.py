# windsurf_agent/agent_factory.py
from typing import Dict, Type, Any
from .base_agent import BaseAgent

class AgentFactory:
    _agents: Dict[str, Type[BaseAgent]] = {}
    
    @classmethod
    def register_agent(cls, name: str, agent_class: Type[BaseAgent]):
        """Register a new agent type."""
        if not issubclass(agent_class, BaseAgent):
            raise ValueError(f"{agent_class.__name__} must be a subclass of BaseAgent")
        cls._agents[name] = agent_class
    
    @classmethod
    def create_agent(cls, agent_type: str, config: Dict[str, Any] = None) -> BaseAgent:
        """Create an agent of the specified type."""
        agent_class = cls._agents.get(agent_type)
        if not agent_class:
            raise ValueError(f"Unknown agent type: {agent_type}")
        return agent_class(config or {})
    
    @classmethod
    def list_agents(cls) -> list:
        """List all registered agent types."""
        return list(cls._agents.keys())

# Register default agents
from .agents.code_agent import CodeAgent
AgentFactory.register_agent('code', CodeAgent)
# Register other agents...