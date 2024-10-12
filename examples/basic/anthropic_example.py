import os
from microagent.core import Microagent
from microagent.types import Agent

# Check if ANTHROPIC_API_KEY is set in the environment
if "ANTHROPIC_API_KEY" not in os.environ:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set. Please set it before running this script.")

# Initialize Microagent with Anthropic as the LLM provider
client = Microagent(llm_type='anthropic')

agent = Agent(
    name="ClaudeHaikuAgent",
    instructions="You are Claude 3 Haiku, a helpful and concise AI assistant.",
    model="claude-3-haiku-20240307"
)

# The system message will be handled separately by the AnthropicClient
messages = [
    {"role": "system", "content": agent.instructions},
    {"role": "user", "content": "Hi Claude! Can you briefly explain what makes you unique?"}
]

response = client.run(agent=agent, messages=messages)

print(f"Claude Haiku: {response.messages[-1]['content']}")