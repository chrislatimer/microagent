import os
from microagent.core import Microagent
from microagent.types import Agent

# Check if GROQ_API_KEY is set in the environment
if "GROQ_API_KEY" not in os.environ:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please set it before running this script.")

# Initialize Microagent with Groq as the LLM provider
client = Microagent(llm_type='groq')

agent = Agent(
    name="GroqAgent",
    instructions="You are a helpful agent powered by Groq.",
    model="llama-3.1-8b-instant",
    tool_choice="auto"
)

messages = [{"role": "user", "content": "Hi! Tell me about Groq."}]
response = client.run(agent=agent, messages=messages)

print(response.messages[-1]["content"])