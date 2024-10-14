from microagent import Microagent, Agent

client = Microagent(llm_type='openai')

agent_a = Agent(
    name="Agent A",
    instructions="You are a helpful agent.",
    model="gpt-3.5-turbo",
)

agent_b = Agent(
    name="Agent B",
    instructions="You specialize in concise responses.",
    model="gpt-3.5-turbo",
)

def transfer_to_concise_agent():
    """Transfer spanish speaking users immediately."""
    return agent_b

agent_a.functions.append(transfer_to_concise_agent)

response = client.run(
    agent=agent_a,
    messages=[{"role": "user", "content": "I need a brief answer."}],
)

print(response.messages[-1]["content"])