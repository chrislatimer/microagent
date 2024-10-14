![Microagent Logo](assets/microagent.png)

# Microagent Framework

Microagent is a lightweight framework for orchestrating multi-agent systems, inspired by and forked from OpenAI's Swarm project.

It adds support for Groq and Anthropic LLMs while retaining the same agent semantics.

## Overview

Microagent focuses on providing a simple yet powerful interface for creating and managing networks of AI agents. It leverages the core concepts introduced in Swarm, such as agent coordination and handoffs, while introducing its own enhancements and modifications.

> **Note**: Microagent is a separate project from OpenAI's Swarm. Because Swarm is positioned as an experimental framework with no intention to maintain it, Microagent looks to pick up the torch and build on it. While it shares some foundational concepts, it has its own development trajectory and feature set and has already deviated quite a bit.

## Key Features

- **Lightweight Agent Orchestration**: Create and manage networks of AI agents with ease.
- **Flexible Handoffs**: Seamlessly transfer control between agents during execution.
- **Function Integration**: Easily integrate Python functions as tools for your agents.
- **Context Management**: Maintain and update context variables across agent interactions.
- **Streaming Support**: Real-time streaming of agent responses for interactive applications.

## Installation

```shell
pip install git+https://github.com/chrislatimer/microagent.git
```

## Quick Start

```python
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
```

## Acknowledgments

Microagent builds upon the innovative work done by OpenAI in their Swarm project. We are grateful for their contributions to the field of multi-agent systems and open-source AI development.

## License

Microagent is released under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Microagent: Empowering developers to build sophisticated multi-agent systems with ease.

# Thanks to the creators of Swarm:

- Ilan Bigio - [ibigio](https://github.com/ibigio)
- James Hills - [jhills20](https://github.com/jhills20)
- Shyamal Anadkat - [shyamal-anadkat](https://github.com/shyamal-anadkat)
- Charu Jaiswal - [charuj](https://github.com/charuj)
- Colin Jarvis - [colin-openai](https://github.com/colin-openai)
- Katia Gil Guzman - [katia-openai](https://github.com/katia-openai)
