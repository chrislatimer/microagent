import pytest
import vcr
from microagent.llm.gemini_client import GeminiClient
import google.generativeai as genai

my_vcr = vcr.VCR(
    cassette_library_dir='tests/fixtures/vcr_cassettes',
    record_mode='once',
    match_on=['method', 'scheme', 'host', 'port', 'path', 'query'],
    filter_headers=['authorization'],
)

@pytest.fixture
def gemini_client():
    return GeminiClient()

@my_vcr.use_cassette('test_gemini_chat_completion.yaml')
def test_chat_completion(gemini_client):
    messages = [{"role": "user", "content": "Say this is a test"}]
    response = gemini_client.chat_completion(messages)

    assert isinstance(response, dict)
    assert "content" in response
    assert isinstance(response["content"], str)
    assert len(response["content"]) > 0

@my_vcr.use_cassette('test_gemini_stream_chat_completion.yaml')
def test_stream_chat_completion(gemini_client):
    messages = [{"role": "user", "content": "Say this is a test"}]
    response = gemini_client.stream_chat_completion(messages)

    assert isinstance(response, dict)
    assert "content" in response
    assert isinstance(response["content"], str)
    assert len(response["content"]) > 0

def test_prepare_messages(gemini_client):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"}
    ]
    prepared_messages = gemini_client.prepare_messages(messages)
    assert len(prepared_messages) == 1
    assert prepared_messages[0]['role'] == 'user'
    assert prepared_messages[0]['parts'][0] == 'Hello'

def test_prepare_tools(gemini_client):
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
    prepared_tools = gemini_client.prepare_tools(tools)
    assert len(prepared_tools) == 1
    assert prepared_tools[0]['name'] == 'test_func'
    assert 'parameters' in prepared_tools[0]

def test_parse_response_with_content(gemini_client):
    class MockResponse:
        class MockCandidate:
            class MockContent:
                class MockPart:
                    function_call = None
                parts = [MockPart()]
            content = MockContent()
        candidates = [MockCandidate()]
        text = "This is a test response"

    response = MockResponse()
    parsed_response = gemini_client.parse_response(response)
    
    assert isinstance(parsed_response, dict)
    assert parsed_response['role'] == 'assistant'
    assert parsed_response['content'] == "This is a test response"
    assert parsed_response['tool_calls'] is None

def test_parse_response_with_function_call(gemini_client):
    class MockResponse:
        class MockCandidate:
            class MockContent:
                class MockPart:
                    class MockFunctionCall:
                        name = "test_func"
                        args = {"param1": "value1"}
                    function_call = MockFunctionCall()
                parts = [MockPart()]
            content = MockContent()
        candidates = [MockCandidate()]

    response = MockResponse()
    parsed_response = gemini_client.parse_response(response)
    
    assert isinstance(parsed_response, dict)
    assert parsed_response['role'] == 'assistant'
    assert parsed_response['content'] is None
    assert parsed_response['tool_calls'] is not None
    assert parsed_response['tool_calls'][0]['function']['name'] == "test_func"
    assert parsed_response['tool_calls'][0]['function']['arguments'] == '{"param1": "value1"}'