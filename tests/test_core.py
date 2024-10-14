import pytest
import vcr
from microagent.core import Microagent
from microagent.types import Agent, Result

# Configure VCR
my_vcr = vcr.VCR(
    cassette_library_dir='tests/fixtures/vcr_cassettes',
    record_mode='new_episodes',
    match_on=['method', 'scheme', 'host', 'port', 'path', 'query'],
    filter_headers=['authorization'],
)

LLM_TYPES = ['openai', 'anthropic', 'groq']
# LLM_TYPES = ['openai']

@pytest.fixture(params=LLM_TYPES)
def microagent(request):
    llm_type = request.param
    model_map = {
        'openai': "gpt-3.5-turbo",
        'anthropic': "claude-3-sonnet-20240229",
        'groq': "llama3-groq-70b-8192-tool-use-preview",
        'gemini': "gemini-pro"
    }
    return Microagent(llm_type=llm_type), model_map[llm_type], llm_type

def use_vcr_for_llm(test_function):
    """Decorator to use VCR with the specific LLM type."""
    def wrapper(microagent, *args, **kwargs):
        client, model, llm_type = microagent
        cassette_name = f'{test_function.__name__}_{llm_type}.yaml'
        with my_vcr.use_cassette(cassette_name):
            return test_function(microagent, *args, **kwargs)
    return wrapper

@use_vcr_for_llm
def test_tool_call(microagent):
    client, model, _ = microagent
    def test_function(arg1, arg2):
        """Test function with two args"""
        return f"Function called with {arg1} and {arg2}"
    agent = Agent(name="Test Agent", instructions="Test instructions", model=model, functions=[test_function])
    messages = [{"role": "user", "content": "Call the test function with arg1=value1 and arg2=value2"}]
    response = client.run(agent=agent, messages=messages, max_turns=3)
    assert len(response.messages) >= 2
    assert any("Function called with value1 and value2" in (msg.get('content') or '') for msg in response.messages)

@use_vcr_for_llm
def test_multiple_tool_calls(microagent):
    client, model, _ = microagent
    def function1(arg):
        """Function1 with an arg"""
        return f"Function 1 called with {arg}"
    def function2(arg):
        """Function2 with an arg"""
        return f"Function 2 called with {arg}"
    agent = Agent(name="Test Agent", instructions="Test instructions", model=model, functions=[function1, function2])
    messages = [{"role": "user", "content": "Call both functions with different arguments"}]
    response = client.run(agent=agent, messages=messages, max_turns=5)
    assert len(response.messages) >= 3
    content = ' '.join(msg.get('content') or '' for msg in response.messages)
    assert "Function 1 called with" in content
    assert "Function 2 called with" in content

@use_vcr_for_llm
def test_agent_handoff(microagent):
    client, model, _ = microagent
    agent2 = Agent(name="Agent 2", instructions="Agent 2 instructions", model=model)
    def handoff_function():
        """handoff Function with no params"""
        return agent2
    agent1 = Agent(name="Agent 1", instructions="Agent 1 instructions", model=model, functions=[handoff_function])
    messages = [{"role": "user", "content": "call function to handoff to Agent 2"}]
    response = client.run(agent=agent1, messages=messages, max_turns=3)
    assert response.agent.name == "Agent 2"

@use_vcr_for_llm
def test_context_variables(microagent):
    client, model, _ = microagent
    def update_context(key, value):
        """Update context function with key and value"""
        return Result(value=f"Updated {key} to {value}", context_variables={key: value})
    agent = Agent(name="Test Agent", instructions="Test instructions", model=model, functions=[update_context])
    messages = [{"role": "user", "content": "Update context with key 'test_key' and value 'test_value'"}]
    response = client.run(agent=agent, messages=messages)
    assert "test_key" in response.context_variables
    assert response.context_variables["test_key"] == "test_value"

@use_vcr_for_llm
def test_max_turns(microagent):
    client, model, _ = microagent
    def loop_function():
        """Function to start the loop"""
        return "Looping"
    agent = Agent(name="Test Agent", instructions="Test instructions", model=model, functions=[loop_function])
    messages = [{"role": "user", "content": "Start a loop that calls the loop_function repeatedly"}]
    response = client.run(agent=agent, messages=messages, max_turns=3)
    assert len(response.messages) <= 7

@use_vcr_for_llm
def test_execute_tools_false(microagent):
    client, model, _ = microagent
    def test_function():
        """Function that shouldn't be called"""
        return "This should not be called"
    agent = Agent(name="Test Agent", instructions="Test instructions", model=model, functions=[test_function])
    messages = [{"role": "user", "content": "Call the test_function"}]
    response = client.run(agent=agent, messages=messages, execute_tools=False, max_turns=3)
    assert len(response.messages) == 1
    assert "tool_calls" in response.messages[0]