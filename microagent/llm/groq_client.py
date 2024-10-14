from typing import Dict, Any, List
import groq
from .base import LLMClient
import json

class GroqClient(LLMClient):
    def __init__(self):
        self.client = groq.Groq()

    def chat_completion(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        prepared_messages = self.prepare_messages(messages)
        chat_params = self.prepare_chat_params(messages=prepared_messages, **kwargs)
        
        response = self.client.chat.completions.create(**chat_params)
        return self.parse_response(response)

    def stream_chat_completion(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        prepared_messages = self.prepare_messages(messages)
        chat_params = self.prepare_chat_params(messages=prepared_messages, **kwargs)
        return self.client.chat.completions.create(stream=True, **chat_params)

    def prepare_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {k: v for k, v in msg.items() if k not in ['sender', 'tool_name']}
            for msg in messages
        ]

    def prepare_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return tools

    def parse_response(self, response: Any) -> Dict[str, Any]:
        if isinstance(response, dict):
            return response
        
        else:    
            # Extract the first choice from the response
            choice = response.choices[0] if response.choices else None
            
            if choice and choice.message:
                parsed_response = {
                    "role": choice.message.role,
                    "content": choice.message.content,
                }
                
                # Handle tool calls
                if choice.message.tool_calls:
                    parsed_response["tool_calls"] = [
                        {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        }
                        for tool_call in choice.message.tool_calls
                    ]
                
                return parsed_response
            else:
                return {
                    "role": None,
                    "content": None,
                    "tool_calls": None
                }

    def prepare_chat_params(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        params = {
            "model": kwargs.get('model', 'llama3-groq-70b-8192-tool-use-preview'),  # Default model for Groq
            "messages": messages
        }
        if 'tools' in kwargs and kwargs['tools']:
            params["tools"] = kwargs['tools']
        if 'tool_choice' in kwargs:
            params["tool_choice"] = kwargs['tool_choice']
        return params

    def prepare_system_message(self, instructions: str) -> Dict[str, Any]:
        return {"role": "system", "content": instructions}

    def prepare_tool_response(self, tool_call_id: str, tool_name: str, content: str) -> Dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": content,
        }