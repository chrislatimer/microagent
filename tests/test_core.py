import pytest
from unittest.mock import Mock
import json
from microagent.core import Microagent, Agent, Response, Result

DEFAULT_RESPONSE_CONTENT = "This is a default response."

class MockLLMClient:
    def __init__(self, llm_type):
        self.llm_type = llm_type
        self.responses = []

    def set_response(self, response):
        self.responses = [response]

    def set_sequential_responses(self, responses):
        self.responses = responses

    def chat_completion(self, **kwargs):
        if not self.responses:
            raise ValueError("No mock responses set")
        return self.responses.pop(0)

    def stream_chat_completion(self, **kwargs):
        return self.chat_completion(**kwargs)

def create_mock_response(llm_type, message, function_calls=None):
    if llm_type in ['openai', 'groq', 'gemini']:
        response = {
            "choices": [{
                "message": message
            }]
        }
        if function_calls:
            response["choices"][0]["message"]["tool_calls"] = function_calls
        return response
    elif llm_type == 'anthropic':
        response = Mock()
        response.content = message['content']
        if function_calls:
            response.tool_calls = function_calls
        else:
            response.tool_calls = []  # Always set tool_calls, even if empty
        return response
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")

@pytest.fixture
def mock_llm_client(request):
    return MockLLMClient(request.param)

@pytest.mark.parametrize("mock_llm_client", ["openai", "anthropic", "groq", "gemini"], indirect=True)
def test_run_with_simple_message(mock_llm_client, monkeypatch):
    agent = Agent(name="Test Agent", instructions="Test instructions", model="gpt-3.5-turbo")
    mock_llm_client.set_response(create_mock_response(mock_llm_client.llm_type, {"role": "assistant", "content": DEFAULT_RESPONSE_CONTENT}, function_calls=[]))
    
    monkeypatch.setattr(Microagent, '_get_client', lambda self, llm_type: mock_llm_client)
    client = Microagent(llm_type=mock_llm_client.llm_type)
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    response = client.run(agent=agent, messages=messages)
    
    assert len(response.messages) == 1
    assert response.messages[0]["content"] == DEFAULT_RESPONSE_CONTENT

@pytest.mark.parametrize("mock_llm_client", ["openai", "anthropic", "groq", "gemini"], indirect=True)
def test_tool_call(mock_llm_client, monkeypatch):
    expected_location = "San Francisco"

    get_weather_mock = Mock()
    def get_weather(location):
        get_weather_mock(location=location)
        return "It's sunny today."

    agent = Agent(name="Test Agent", instructions="Test instructions", model="gpt-3.5-turbo", functions=[get_weather])
    messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]

    if mock_llm_client.llm_type in ['openai', 'groq', 'gemini']:
        function_calls = [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": json.dumps({"location": expected_location})}}]
    else:  # anthropic
        function_call = Mock()
        function_call.id = "call_1"
        function_call.type = "function"
        function_call.function = Mock()
        function_call.function.name = "get_weather"
        function_call.function.arguments = json.dumps({"location": expected_location})
        function_calls = [function_call]

    mock_llm_client.set_sequential_responses([
        create_mock_response(
            mock_llm_client.llm_type,
            message={"role": "assistant", "content": ""},
            function_calls=function_calls
        ),
        create_mock_response(
            mock_llm_client.llm_type,
            {"role": "assistant", "content": DEFAULT_RESPONSE_CONTENT}
        ),
    ])

    monkeypatch.setattr(Microagent, '_get_client', lambda self, llm_type: mock_llm_client)
    client = Microagent(llm_type=mock_llm_client.llm_type)
    response = client.run(agent=agent, messages=messages)

    get_weather_mock.assert_called_once_with(location=expected_location)
    assert len(response.messages) == 3
    assert response.messages[-1]["content"] == DEFAULT_RESPONSE_CONTENT

@pytest.mark.parametrize("mock_llm_client", ["openai", "anthropic", "groq", "gemini"], indirect=True)
def test_execute_tools_false(mock_llm_client, monkeypatch):
    expected_location = "San Francisco"

    get_weather_mock = Mock()
    def get_weather(location):
        get_weather_mock(location=location)
        return "It's sunny today."

    agent = Agent(name="Test Agent", instructions="Test instructions", model="gpt-3.5-turbo", functions=[get_weather])
    messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]

    if mock_llm_client.llm_type in ['openai', 'groq', 'gemini']:
        function_calls = [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": json.dumps({"location": expected_location})}}]
    else:  # anthropic
        function_call = Mock()
        function_call.id = "call_1"
        function_call.type = "function"
        function_call.function = Mock()
        function_call.function.name = "get_weather"
        function_call.function.arguments = json.dumps({"location": expected_location})
        function_calls = [function_call]

    mock_response = create_mock_response(
        mock_llm_client.llm_type,
        message={"role": "assistant", "content": ""},
        function_calls=function_calls
    )
    mock_llm_client.set_response(mock_response)

    monkeypatch.setattr(Microagent, '_get_client', lambda self, llm_type: mock_llm_client)
    client = Microagent(llm_type=mock_llm_client.llm_type)
    response = client.run(agent=agent, messages=messages, execute_tools=False)

    get_weather_mock.assert_not_called()

    assert len(response.messages) == 1
    tool_calls = response.messages[0].get("tool_calls", [])
    assert tool_calls and len(tool_calls) == 1
    
    if mock_llm_client.llm_type in ['openai', 'groq', 'gemini']:
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert json.loads(tool_calls[0]["function"]["arguments"])["location"] == expected_location
    else:  # anthropic
        assert tool_calls[0].function.name == "get_weather"
        assert json.loads(tool_calls[0].function.arguments)["location"] == expected_location

@pytest.mark.parametrize("mock_llm_client", ["openai", "anthropic", "groq", "gemini"], indirect=True)
def test_handoff(mock_llm_client, monkeypatch):
    agent2 = Agent(name="Test Agent 2", instructions="Test instructions 2", model="gpt-3.5-turbo")

    def transfer_to_agent2():
        return agent2

    agent1 = Agent(name="Test Agent 1", instructions="Test instructions 1", model="gpt-3.5-turbo", functions=[transfer_to_agent2])

    if mock_llm_client.llm_type in ['openai', 'groq', 'gemini']:
        function_calls = [{"id": "call_1", "type": "function", "function": {"name": "transfer_to_agent2", "arguments": "{}"}}]
    else:  # anthropic
        function_call = Mock()
        function_call.id = "call_1"
        function_call.type = "function"
        function_call.function = Mock()
        function_call.function.name = "transfer_to_agent2"
        function_call.function.arguments = "{}"
        function_calls = [function_call]

    mock_llm_client.set_sequential_responses([
        create_mock_response(
            mock_llm_client.llm_type,
            message={"role": "assistant", "content": ""},
            function_calls=function_calls
        ),
        create_mock_response(
            mock_llm_client.llm_type,
            {"role": "assistant", "content": DEFAULT_RESPONSE_CONTENT}
        ),
    ])

    monkeypatch.setattr(Microagent, '_get_client', lambda self, llm_type: mock_llm_client)
    client = Microagent(llm_type=mock_llm_client.llm_type)
    messages = [{"role": "user", "content": "I want to talk to agent 2"}]
    response = client.run(agent=agent1, messages=messages)

    assert len(response.messages) == 3
    assert response.messages[-1]["content"] == DEFAULT_RESPONSE_CONTENT
    assert response.agent.name == "Test Agent 2"