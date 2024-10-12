import os
from microagent.core import Microagent
from microagent.types import Agent

# Check if GOOGLE_API_KEY is set in the environment
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it before running this script.")

# Initialize Microagent with Gemini as the LLM provider
client = Microagent(llm_type='gemini')

agent = Agent(
    name="GeminiAgent",
    instructions="You are an AI assistant powered by Google's Gemini model. You're helpful, creative, and knowledgeable.",
    model="gemini-pro"  # This is the text-only model
)

messages = [
    {"role": "system", "content": agent.instructions},
    {"role": "user", "content": "Hi Gemini! Can you explain what makes you unique compared to other AI models?"}
]

response = client.run(agent=agent, messages=messages)

print(f"Gemini: {response.messages[-1]['content']}")