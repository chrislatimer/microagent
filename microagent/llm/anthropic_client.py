from typing import Dict, Any, List
from anthropic import Anthropic
from .base import LLMClient
import json

class AnthropicClient(LLMClient):
    def __init__(self):
        self.client = Anthropic()
        self.default_model = "claude-3-opus-20240229"
        self.default_max_tokens = 1000

    def chat_completion(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        # Prepare parameters for the API call
        params = self.prepare_chat_params(messages=messages, **kwargs)

        # Call the LLM API
        response = self.client.messages.create(**params)

        return response


    def stream_chat_completion(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        # Prepare parameters for the API call
        params = self.prepare_chat_params(messages=messages, **kwargs)
        params['stream'] = True  # Enable streaming

        # Call the LLM API
        return self.client.messages.create(**params)

    def prepare_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # return [
        #     {'role': m['role'], 'content': m['content']}
        #     for m in messages
        # ]
        return messages
    
    def prepare_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prepared_tools = []
        for tool in tools:
            if 'function' in tool:
                # This is the initial format
                function = tool['function']
                prepared_tool = {
                    "name": function['name'],
                    "description": function.get('description', ''),
                    "input_schema": {
                        "type": "object",
                        "properties": function['parameters']['properties'],
                        "required": function['parameters'].get('required', [])
                    }
                }
            else:
                # This is the already processed format
                prepared_tool = {
                    "name": tool['name'],
                    "description": tool.get('description', ''),
                    "input_schema": tool['input_schema']
                }
            prepared_tools.append(prepared_tool)
        return prepared_tools

    def parse_response(self, response: Any) -> Dict[str, Any]:
        content = []
        tool_calls = []

        # Loop through response content and differentiate based on the type
        if hasattr(response, 'content') and isinstance(response.content, list):
            for block in response.content:
                if hasattr(block, 'type'):
                    if block.type == 'text':
                        # Collect text content from TextBlock
                        content.append(getattr(block, 'text', ''))
                    elif block.type == 'tool_use':
                        # Collect tool usage details from ToolUseBlock
                        tool_calls.append({
                            "id": getattr(block, 'id', 'unknown'),
                            "function": {
                                "name": getattr(block, 'name', ''),
                                "arguments": json.dumps(getattr(block, 'input', {}))
                            }
                        })

        # Return the structured response
        return {
            "role": "assistant",
            "content": "\n".join(content) if content else "None",
            "tool_calls": tool_calls if tool_calls else None
        }
    
    def prepare_chat_params(self, **kwargs) -> Dict[str, Any]:
        
        params = {
            "model": kwargs.get('model', self.default_model),
            "max_tokens": kwargs.get('max_tokens', self.default_max_tokens),
            "messages": [
                {k: v for k, v in message.items() if k != 'tool_calls' and k!= 'sender'}
                for message in kwargs['messages']
            ],
        }
        if 'tools' in kwargs and kwargs['tools']:
            params["tools"] = self.prepare_tools(kwargs['tools'])
        
        # Handle system message
        if params['messages'] and params['messages'][0]['role'] == 'system':
            system_message = params['messages'].pop(0)
            params['system'] = [{"type": "text", "text": system_message['content']}]
        
        
        return params
    
    def prepare_tool_response(self, tool_call_id: str, tool_name: str, content: str) -> Dict[str, Any]:
        return {
            "role": "user",
            "content": f"Tool '{tool_name}' response: {content}"
        }
    
    def prepare_system_message(self, instructions: str) -> Dict[str, Any]:
        return {"role": "system", "content": instructions}