from unittest.mock import Mock
import json

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
        response = self.responses.pop(0)
        if self.llm_type == 'anthropic':
            return response
        return {'choices': [{'message': response}]}

    def stream_chat_completion(self, **kwargs):
        return self.chat_completion(**kwargs)


def create_mock_response(llm_type, message, function_calls=None):
    if llm_type == 'anthropic':
        response = Mock()
        response.content = message['content']
        response.tool_calls = function_calls or []
        return response
    else:
        response = message
        if function_calls:
            response['tool_calls'] = function_calls
        return response