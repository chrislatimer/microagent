# Standard library imports
import copy
import json
import os
from collections import defaultdict
from typing import List, Callable, Union, Dict, Any
from abc import ABC, abstractmethod

# Package/library imports
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
import groq

# Local imports
from .util import function_to_json, debug_print, merge_chunk
from .types import (
    Agent,
    AgentFunction,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    Function,
    Response,
    Result,
)

__CTX_VARS_NAME__ = "context_variables"

class LLMClient(ABC):
    @abstractmethod
    def chat_completion(self, **kwargs):
        pass

    @abstractmethod
    def stream_chat_completion(self, **kwargs):
        pass

class OpenAIClient(LLMClient):
    def __init__(self):
        self.client = OpenAI()

    def chat_completion(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)

    def stream_chat_completion(self, **kwargs):
        return self.chat_completion(**kwargs, stream=True)

class AnthropicClient(LLMClient):
    def __init__(self):
        self.client = Anthropic()  # Use Anthropic directly here

    def chat_completion(self, **kwargs):
        messages = kwargs.pop('messages')
        
        # Extract the system message if it exists
        system_message = next((m['content'] for m in messages if m['role'] == 'system'), None)
        
        # Filter out the system message from the regular messages
        anthropic_messages = [
            {'role': m['role'], 'content': m['content']}
            for m in messages if m['role'] != 'system'
        ]
        
        create_params = {
            "model": kwargs.get('model', 'claude-3-haiku-20240307'),
            "max_tokens": kwargs.get('max_tokens', 1024),
            "temperature": kwargs.get('temperature', 0.7),
            "messages": anthropic_messages
        }
        
        # Add system message if it exists
        if system_message:
            create_params["system"] = system_message

        return self.client.messages.create(**create_params)

    def stream_chat_completion(self, **kwargs):
        kwargs['stream'] = True
        return self.chat_completion(**kwargs)

class GroqClient(LLMClient):
    def __init__(self):
        self.client = groq.Groq()

    def chat_completion(self, **kwargs):
        response = self.client.chat.completions.create(**kwargs)
        return {
            "choices": [{
                "message": {
                    "role": response.choices[0].message.role,
                    "content": response.choices[0].message.content,
                    "tool_calls": getattr(response.choices[0].message, 'tool_calls', [])
                }
            }]
        }

    def stream_chat_completion(self, **kwargs):
        return self.chat_completion(**kwargs, stream=True)

class GeminiClient(LLMClient):
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    def chat_completion(self, **kwargs):
        messages = kwargs.pop('messages')
        tools = kwargs.pop('tools', None)
        gemini_messages = []
        for m in messages:
            if m['role'] == 'system':
                continue  # Gemini doesn't use system messages in the same way
            gemini_messages.append(genai.types.ContentDict(role=m['role'], parts=[m['content']]))
        
        if tools:
            # Convert tools to Gemini's function declarations format
            function_declarations = []
            for tool in tools:
                function_declarations.append({
                    "name": tool["function"]["name"],
                    "description": tool["function"].get("description", ""),
                    "parameters": tool["function"]["parameters"]
                })
        
        try:
            response = self.model.generate_content(
                gemini_messages,
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get('temperature', 0.7),
                    max_output_tokens=kwargs.get('max_tokens', 1024),
                ),
                tools=function_declarations if tools else None
            )
            
            # Convert Gemini response to a format similar to OpenAI's
            if response.candidates[0].content.parts[0].function_call:
                function_call = response.candidates[0].content.parts[0].function_call
                return {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": "call_1",  # Gemini doesn't provide an ID, so we use a placeholder
                                "type": "function",
                                "function": {
                                    "name": function_call.name,
                                    "arguments": json.dumps(function_call.args)
                                }
                            }]
                        }
                    }]
                }
            else:
                return {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": response.text
                        }
                    }]
                }
        except Exception as e:
            print(f"Error in Gemini API call: {str(e)}")
            raise

    def stream_chat_completion(self, **kwargs):
        # Gemini doesn't support streaming in the same way as OpenAI
        # For simplicity, we'll just return the non-streaming version
        return self.chat_completion(**kwargs)

class Microagent:
    def __init__(self, llm_type='openai'):
        self.llm_type = llm_type
        self.client = self._get_client(llm_type)

    def _get_client(self, llm_type):
        clients = {
            'openai': OpenAIClient(),
            'anthropic': AnthropicClient(),
            'groq': GroqClient(),
            'gemini': GeminiClient()
        }
        return clients.get(llm_type, OpenAIClient())

    def get_chat_completion(
        self,
        agent: Agent,
        history: List,
        context_variables: dict,
        model_override: str,
        stream: bool,
        debug: bool,
    ) -> ChatCompletionMessage:
        context_variables = defaultdict(str, context_variables)
        instructions = (
            agent.instructions(context_variables)
            if callable(agent.instructions)
            else agent.instructions
        )
        messages = [{"role": "system", "content": instructions}] + history
        debug_print(debug, "Getting chat completion for...:", messages)

        tools = [function_to_json(f) for f in agent.functions]
        for tool in tools:
            params = tool["function"]["parameters"]
            params["properties"].pop(__CTX_VARS_NAME__, None)
            if __CTX_VARS_NAME__ in params["required"]:
                params["required"].remove(__CTX_VARS_NAME__)

        create_params = {
            "model": model_override or agent.model,
            "messages": messages,
            "tools": tools or None,
            "tool_choice": agent.tool_choice,
        }

        if tools and self.llm_type == 'openai':
            create_params["parallel_tool_calls"] = agent.parallel_tool_calls

        if stream:
            return self.client.stream_chat_completion(**create_params)
        else:
            return self.client.chat_completion(**create_params)
        
    def handle_function_result(self, result, debug) -> Result:
        match result:
            case Result() as result:
                return result

            case Agent() as agent:
                return Result(
                    value=json.dumps({"assistant": agent.name}),
                    agent=agent,
                )
            case _:
                try:
                    return Result(value=str(result))
                except Exception as e:
                    error_message = f"Failed to cast response to string: {result}. Make sure agent functions return a string or Result object. Error: {str(e)}"
                    debug_print(debug, error_message)
                    raise TypeError(error_message)
    
    def handle_tool_calls(
        self,
        tool_calls: Union[List[Dict], List[Any]],
        functions: List[AgentFunction],
        context_variables: dict,
        debug: bool,
    ) -> Response:
        function_map = {f.__name__: f for f in functions}
        partial_response = Response(
            messages=[], agent=None, context_variables={})

        if not isinstance(tool_calls, list):
            tool_calls = [tool_calls]

        for tool_call in tool_calls:
            try:
                if isinstance(tool_call, dict):
                    name = tool_call['function']['name']
                    arguments = tool_call['function']['arguments']
                    tool_call_id = tool_call.get('id', 'unknown')
                elif hasattr(tool_call, 'function'):
                    name = tool_call.function.name
                    arguments = tool_call.function.arguments
                    tool_call_id = getattr(tool_call, 'id', 'unknown')
                else:
                    raise ValueError(f"Unsupported tool call structure: {tool_call}")

                if name not in function_map:
                    raise ValueError(f"Tool {name} not found in function map.")

                debug_print(debug, f"Processing tool call: {name} with arguments {arguments}")

                func = function_map[name]
                args = json.loads(arguments)
                if __CTX_VARS_NAME__ in func.__code__.co_varnames:
                    args[__CTX_VARS_NAME__] = context_variables
                raw_result = func(**args)

                result: Result = self.handle_function_result(raw_result, debug)
                partial_response.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "tool_name": name,
                    "content": result.value,
                })
                partial_response.context_variables.update(result.context_variables)
                if result.agent:
                    partial_response.agent = result.agent
                    return partial_response

            except Exception as e:
                error_message = f"Error processing tool call: {str(e)}"
                debug_print(debug, error_message)
                partial_response.messages.append({
                    "role": "tool",
                    "tool_call_id": getattr(tool_call, 'id', 'unknown'),
                    "tool_name": getattr(tool_call, 'name', 'unknown'),
                    "content": error_message,
                })

        return partial_response

    def run_and_stream(
        self,
        agent: Agent,
        messages: List,
        context_variables: dict = {},
        model_override: str = None,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ):
        active_agent = agent
        context_variables = copy.deepcopy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        while len(history) - init_len < max_turns:

            message = {
                "content": "",
                "sender": agent.name,
                "role": "assistant",
                "function_call": None,
                "tool_calls": defaultdict(
                    lambda: {
                        "function": {"arguments": "", "name": ""},
                        "id": "",
                        "type": "",
                    }
                ),
            }

            # get completion with current history, agent
            completion = self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=True,
                debug=debug,
            )

            yield {"delim": "start"}
            for chunk in completion:
                delta = json.loads(chunk.choices[0].delta.json())
                if delta["role"] == "assistant":
                    delta["sender"] = active_agent.name
                yield delta
                delta.pop("role", None)
                delta.pop("sender", None)
                merge_chunk(message, delta)
            yield {"delim": "end"}

            message["tool_calls"] = list(
                message.get("tool_calls", {}).values())
            if not message["tool_calls"]:
                message["tool_calls"] = None
            debug_print(debug, "Received completion:", message)
            history.append(message)

            if not message["tool_calls"] or not execute_tools:
                debug_print(debug, "Ending turn.")
                break

            # convert tool_calls to objects
            tool_calls = []
            for tool_call in message["tool_calls"]:
                function = Function(
                    arguments=tool_call["function"]["arguments"],
                    name=tool_call["function"]["name"],
                )
                tool_call_object = ChatCompletionMessageToolCall(
                    id=tool_call["id"], function=function, type=tool_call["type"]
                )
                tool_calls.append(tool_call_object)

            # handle function calls, updating context_variables, and switching agents
            partial_response = self.handle_tool_calls(
                tool_calls, active_agent.functions, context_variables, debug
            )
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        yield {
            "response": Response(
                messages=history[init_len:],
                agent=active_agent,
                context_variables=context_variables,
            )
        }

    def run(
        self,
        agent: Agent,
        messages: List,
        context_variables: dict = {},
        model_override: str = None,
        stream: bool = False,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ) -> Response:
        if stream:
            return self.run_and_stream(
                agent=agent,
                messages=messages,
                context_variables=context_variables,
                model_override=model_override,
                debug=debug,
                max_turns=max_turns,
                execute_tools=execute_tools,
            )
        active_agent = agent
        context_variables = copy.deepcopy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)
        while len(history) - init_len < max_turns and active_agent:
            completion = self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=stream,
                debug=debug
            )
            
            if self.llm_type in ['openai', 'groq', 'gemini']:
                message = completion['choices'][0]['message']
                tool_calls = message.get('tool_calls', [])
            elif self.llm_type == 'anthropic':
                message = {"role": "assistant", "content": completion.content}
                tool_calls = getattr(completion, 'tool_calls', [])
                if tool_calls:
                    message['tool_calls'] = tool_calls  # Add tool_calls to the message for Anthropic
            else:
                raise ValueError(f"Unsupported LLM type: {self.llm_type}")

            message['sender'] = active_agent.name
            debug_print(debug, "Received completion:", message)
            history.append(message)

            if not tool_calls or not execute_tools:
                debug_print(debug, "Ending turn.")
                break

            partial_response = self.handle_tool_calls(
                tool_calls, active_agent.functions, context_variables, debug
            )
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        return Response(
            messages=history[init_len:],
            agent=active_agent,
            context_variables=context_variables,
        )