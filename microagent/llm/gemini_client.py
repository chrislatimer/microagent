import os
import json
from typing import Dict, Any, List
import google.generativeai as genai
from .base import LLMClient

class GeminiClient(LLMClient):
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    def chat_completion(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        prepared_messages = self.prepare_messages(messages)
        response = self.model.generate_content(
            prepared_messages,
            generation_config=genai.types.GenerationConfig(
                temperature=kwargs.get('temperature', 0.7),
                max_output_tokens=kwargs.get('max_tokens', 1024),
            ),
            tools=self.prepare_tools(kwargs.get('tools', []))
        )
        return self.parse_response(response)

    def stream_chat_completion(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        return self.chat_completion(messages, **kwargs)

    def prepare_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            genai.types.ContentDict(role=m['role'], parts=[m['content']])
            for m in messages if m['role'] != 'system'
        ]

    def prepare_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "name": tool["function"]["name"],
                "description": tool["function"].get("description", ""),
                "parameters": tool["function"]["parameters"]
            }
            for tool in tools
        ]

    def parse_response(self, response: Any) -> Dict[str, Any]:
        if response.candidates[0].content.parts[0].function_call:
            function_call = response.candidates[0].content.parts[0].function_call
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": function_call.name,
                        "arguments": json.dumps(function_call.args)
                    }
                }]
            }
        else:
            return {
                "role": "assistant",
                "content": response.text,
                "tool_calls": None
            }