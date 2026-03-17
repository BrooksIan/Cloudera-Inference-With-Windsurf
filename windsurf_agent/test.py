from windsurf_agent import AgentFactory

# Create a code agent
code_agent = AgentFactory.create_agent('code')

# Generate code
code = code_agent.generate_code("Function to calculate factorial")
print(code)