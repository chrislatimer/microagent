import pytest
import vcr
from microagent.llm.openai_client import OpenAIClient

my_vcr = vcr.VCR(
    cassette_library_dir='tests/fixtures/vcr_cassettes',
    record_mode='once',
    match_on=['method', 'scheme', 'host', 'port', 'path', 'query'],
    filter_headers=['authorization'],
)

@pytest.fixture
def openai_client():
    return OpenAIClient()

@my_vcr.use_cassette('test_chat_completion.yaml')
def test_chat_completion(openai_client):
    messages = [{"role": "user", "content": "Say this is a test"}]
    response = openai_client.chat_completion(messages, model="gpt-3.5-turbo")

    assert isinstance(response, dict)
    assert "content" in response
    assert isinstance(response["content"], str)
    assert len(response["content"]) > 0

@my_vcr.use_cassette('test_stream_chat_completion.yaml')
def test_stream_chat_completion(openai_client):
    messages = [{"role": "user", "content": "Say this is a test"}]
    stream = openai_client.stream_chat_completion(messages, model="gpt-3.5-turbo")

    assert hasattr(stream, '__iter__')
    
    response_content = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            response_content += chunk.choices[0].delta.content
    
    assert len(response_content) > 0

def test_prepare_messages(openai_client):
    messages = [{"role": "user", "content": "Hello"}]
    assert openai_client.prepare_messages(messages) == messages

def test_prepare_tools(openai_client):
    tools = [{"function": {"name": "test_func"}}]
    assert openai_client.prepare_tools(tools) == tools

@my_vcr.use_cassette('test_parse_response.yaml')
def test_parse_response(openai_client):
    messages = [{"role": "user", "content": "Say this is a test"}]
    response = openai_client.chat_completion(messages, model="gpt-3.5-turbo")
    
    assert isinstance(response, dict)
    assert "content" in response
    assert isinstance(response["content"], str)
    assert len(response["content"]) > 0

    class MockResponse:
        class MockChoice:
            class MockMessage:
                role = "assistant"
                content = "Mocked response"
                tool_calls = None
            message = MockMessage()
        choices = [MockChoice()]

    mock_response = MockResponse()
    parsed_response = openai_client.parse_response(mock_response)
    assert parsed_response == {"role": "assistant", "content": "Mocked response", "tool_calls": []}