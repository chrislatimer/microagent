from typing import Dict, Any, List
from openai import OpenAI

class OpenAIClient:
    def __init__(self):
        self.client = OpenAI()

    def chat_completion(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        if 'tools' in kwargs and not kwargs['tools']:
            del kwargs['tools']
        if 'model' not in kwargs:
            kwargs['model'] = 'gpt-3.5-turbo'  # Default model
        response = self.client.chat.completions.create(messages=messages, **kwargs)
        return self.parse_response(response)

    def stream_chat_completion(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        if 'tools' in kwargs and not kwargs['tools']:
            del kwargs['tools']
        if 'model' not in kwargs:
            kwargs['model'] = 'gpt-3.5-turbo'  # Default model
        return self.client.chat.completions.create(messages=messages, stream=True, **kwargs)

    def prepare_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return messages

    def prepare_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return tools

    def parse_response(self, response: Any) -> Dict[str, Any]:
        if isinstance(response, dict):
            return response
        if isinstance(response, list):
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call['id'],
                        "type": tool_call['type'],
                        "function": {
                            "name": tool_call['function']['name'],
                            "arguments": tool_call['function']['arguments']
                        }
                    } for tool_call in response
                ]
            }
        message = response.choices[0].message
        return {
            "role": message.role,
            "content": message.content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                } for tool_call in (message.tool_calls or [])
            ]
        }
    
    def prepare_chat_params(self, **kwargs) -> Dict[str, Any]:
        params = {
            "model": kwargs.get('model', 'gpt-3.5-turbo'),
            "messages": [
                {k: v for k, v in message.items() if k != 'tool_calls' or v}
                for message in kwargs['messages']
            ],
        }
        if 'tools' in kwargs:
            params["tools"] = kwargs['tools']
        if 'tool_choice' in kwargs:
            params["tool_choice"] = kwargs['tool_choice']
        return params

    # def prepare_chat_params(self, **kwargs) -> Dict[str, Any]:
    #     params = {
    #         "model": kwargs.get('model', 'gpt-3.5-turbo'),
    #         "messages": kwargs['messages'],
    #     }
    #     if 'tools' in kwargs:
    #         params["tools"] = [{"type": "function", "function": tool} for tool in self.prepare_tools(kwargs['tools'])]
    #     if 'tool_choice' in kwargs:
    #         params["tool_choice"] = kwargs['tool_choice']
    #     return params
    
    def prepare_system_message(self, instructions: str) -> Dict[str, Any]:
        return {"role": "system", "content": instructions}

    def prepare_tool_response(self, tool_call_id: str, tool_name: str, content: str) -> Dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "content": content,
        }
