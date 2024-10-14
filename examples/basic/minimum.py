from microagent import Agent
from microagent.core import Microagent

client = Microagent(llm_type='openai')

agent = Agent(
    name="Agent",
    instructions="You are a helpful agent.",
    model="gpt-3.5-turbo",
)

messages = [{"role": "user", "content": "Hi!"}]
response = client.run(agent=agent, messages=messages)

print(response.messages[-1]["content"])
