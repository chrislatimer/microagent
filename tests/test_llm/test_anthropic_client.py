import pytest
import vcr
from anthropic import Anthropic
from microagent.llm.anthropic_client import AnthropicClient

# VCR configuration
my_vcr = vcr.VCR(
    cassette_library_dir='tests/fixtures/vcr_cassettes',
    record_mode='once',
    match_on=['method', 'scheme', 'host', 'port', 'path', 'query'],
    filter_headers=['authorization'],
)

@pytest.fixture
def anthropic_client():
    return AnthropicClient()

@my_vcr.use_cassette('test_anthropic_chat_completion.yaml')
def test_chat_completion(anthropic_client):
    messages = [{"role": "user", "content": "Say this is a test"}]
    response = anthropic_client.chat_completion(messages)

    assert isinstance(response, dict)
    assert "content" in response
    assert isinstance(response["content"], list)
    assert len(response["content"]) > 0
    assert isinstance(response["content"][0].text, str)

@my_vcr.use_cassette('test_anthropic_stream_chat_completion.yaml')
def test_stream_chat_completion(anthropic_client):
    messages = [{"role": "user", "content": "Say this is a test"}]
    stream = anthropic_client.stream_chat_completion(messages)

    assert hasattr(stream, '__iter__')
    
    response_content = ""
    for chunk in stream:
        if hasattr(chunk, 'delta'):
            if hasattr(chunk.delta, 'text'):
                response_content += chunk.delta.text
        elif hasattr(chunk, 'content'):
            for content_block in chunk.content:
                if hasattr(content_block, 'text'):
                    response_content += content_block.text
    
    assert len(response_content) > 0


def test_prepare_messages(anthropic_client):
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    prepared_messages = anthropic_client.prepare_messages(messages)
    assert len(prepared_messages) == 2
    assert prepared_messages[0]['role'] == 'user'
    assert prepared_messages[1]['role'] == 'assistant'

def test_prepare_tools(anthropic_client):
    tools = [{
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
    prepared_tools = anthropic_client.prepare_tools(tools)
    assert len(prepared_tools) == 1
    assert prepared_tools[0]['type'] == 'function'
    assert prepared_tools[0]['function']['name'] == 'test_func'
    assert 'parameters' in prepared_tools[0]['function']

def test_parse_response(anthropic_client):
    class MockResponse:
        content = "This is a test response"
        tool_calls = None

    response = MockResponse()
    parsed_response = anthropic_client.parse_response(response)
    
    assert isinstance(parsed_response, dict)
    assert parsed_response['role'] == 'assistant'
    assert parsed_response['content'] == "This is a test response"
    assert parsed_response['tool_calls'] is None