import pytest
import vcr
from microagent.llm.groq_client import GroqClient

my_vcr = vcr.VCR(
    cassette_library_dir='tests/fixtures/vcr_cassettes',
    record_mode='once',
    match_on=['method', 'scheme', 'host', 'port', 'path', 'query'],
    filter_headers=['authorization'],
)

@pytest.fixture
def groq_client():
    return GroqClient()

@my_vcr.use_cassette('test_groq_chat_completion.yaml')
def test_chat_completion(groq_client):
    messages = [{"role": "user", "content": "Say this is a test"}]
    response = groq_client.chat_completion(messages, model="mixtral-8x7b-32768")

    assert isinstance(response, dict)
    assert "content" in response
    assert isinstance(response["content"], str)
    assert len(response["content"]) > 0

@my_vcr.use_cassette('test_groq_stream_chat_completion.yaml')
def test_stream_chat_completion(groq_client):
    messages = [{"role": "user", "content": "Say this is a test"}]
    stream = groq_client.stream_chat_completion(messages, model="mixtral-8x7b-32768")

    assert hasattr(stream, '__iter__')
    
    response_content = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            response_content += chunk.choices[0].delta.content
    
    assert len(response_content) > 0

def test_prepare_messages(groq_client):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello", "sender": "John", "tool_name": "chat"},
        {"role": "assistant", "content": "Hi there!", "tool_name": "chat"}
    ]
    prepared_messages = groq_client.prepare_messages(messages)
    assert len(prepared_messages) == 3
    assert all('sender' not in msg for msg in prepared_messages)
    assert all('tool_name' not in msg for msg in prepared_messages)
    assert prepared_messages[1]['role'] == 'user'
    assert prepared_messages[1]['content'] == 'Hello'

def test_prepare_tools(groq_client):
    tools = [{
        "type": "function",
        "function": {
            "name": "test_func",
            "description": "A test function",
            "parameters": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string"}
                },
                "required": ["param1"]
            }
        }
    }]
    prepared_tools = groq_client.prepare_tools(tools)
    assert prepared_tools == tools

def test_parse_response(groq_client):
    class MockChoice:
        class MockMessage:
            role = "assistant"
            content = "This is a test response"
            tool_calls = None
        message = MockMessage()

    class MockResponse:
        choices = [MockChoice()]

    response = MockResponse()
    parsed_response = groq_client.parse_response(response)
    
    assert isinstance(parsed_response, dict)
    assert parsed_response['role'] == 'assistant'
    assert parsed_response['content'] == "This is a test response"
    assert parsed_response['tool_calls'] is None

def test_parse_response_with_tool_calls(groq_client):
    class MockToolCall:
        id = "call_1"
        type = "function"
        function = {"name": "test_func", "arguments": '{"param1": "value1"}'}

    class MockChoice:
        class MockMessage:
            role = "assistant"
            content = None
            tool_calls = [MockToolCall()]
        message = MockMessage()

    class MockResponse:
        choices = [MockChoice()]

    response = MockResponse()
    parsed_response = groq_client.parse_response(response)
    
    assert isinstance(parsed_response, dict)
    assert parsed_response['role'] == 'assistant'
    assert parsed_response['content'] is None
    assert parsed_response['tool_calls'] is not None
    assert parsed_response['tool_calls'][0].id == "call_1"
    assert parsed_response['tool_calls'][0].function['name'] == "test_func"
    assert parsed_response['tool_calls'][0].function['arguments'] == '{"param1": "value1"}'