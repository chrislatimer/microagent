from typing import Dict, Any, List
import groq
from .base import LLMClient

class GroqClient(LLMClient):
    def __init__(self):
        self.client = groq.Groq()

    def chat_completion(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        prepared_messages = self.prepare_messages(messages)
        response = self.client.chat.completions.create(messages=prepared_messages, **kwargs)
        return self.parse_response(response)

    def stream_chat_completion(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        prepared_messages = self.prepare_messages(messages)
        return self.client.chat.completions.create(messages=prepared_messages, stream=True, **kwargs)

    def prepare_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {k: v for k, v in msg.items() if k not in ['sender', 'tool_name']}
            for msg in messages
        ]

    def prepare_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return tools

    def parse_response(self, response: Any) -> Dict[str, Any]:
        return {
            "role": response.choices[0].message.role,
            "content": response.choices[0].message.content,
            "tool_calls": getattr(response.choices[0].message, 'tool_calls', None)
        }