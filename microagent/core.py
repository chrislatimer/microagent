from typing import List, Dict, Any
from microagent.llm.factory import LLMFactory
from .types import Agent, Response, Result
from .util import function_to_json, debug_print
import json


class Microagent:
    def __init__(self, llm_type='openai'):
        self.client = LLMFactory.create(llm_type)

    def get_chat_completion(
        self,
        agent: Agent,
        history: List[Dict[str, Any]],
        context_variables: Dict[str, Any],
        model_override: str,
        stream: bool,
        debug: bool,
    ) -> Dict[str, Any]:
        messages = self._prepare_messages(agent, history, context_variables, debug)
        tools = self._prepare_tools(agent, debug)
        
        params = {
            "model": model_override or agent.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": agent.tool_choice if agent.tool_choice is not None else "auto",
        }

        if stream:
            return self.client.stream_chat_completion(**params)
        else:
            return self.client.chat_completion(**params)

    def _prepare_messages(self, agent: Agent, history: List[Dict[str, Any]], context_variables: Dict[str, Any], debug: bool) -> List[Dict[str, Any]]:
        instructions = agent.instructions(context_variables) if callable(agent.instructions) else agent.instructions
        system_message = self.client.prepare_system_message(instructions)
        messages = [system_message] + history
        debug_print(debug, "Using instructions:", instructions)
        debug_print(debug, "Getting chat completion for:", messages)

        return messages

    def _prepare_tools(self, agent: Agent, debug: bool) -> List[Dict[str, Any]]:
        tools = [function_to_json(f) for f in agent.functions]
        debug_print(debug, "Tools is set to:", tools)
        return tools

    def handle_tool_calls(
        self,
        tool_calls: Any,
        functions: List[Any],
        context_variables: Dict[str, Any],
        debug: bool,
    ) -> Response:
        function_map = {f.__name__: f for f in functions}
        partial_response = Response(messages=[], agent=None, context_variables={})

        for tool_call in tool_calls:
            try:
                name = tool_call['function']['name']
                arguments = tool_call['function']['arguments']
                tool_call_id = tool_call['id']

                if name not in function_map:
                    raise ValueError(f"Tool {name} not found in function map.")

                debug_print(debug, f"Processing tool call: {name} with arguments {arguments}")

                func = function_map[name]
                args = json.loads(arguments)
                if "context_variables" in func.__code__.co_varnames:
                    args["context_variables"] = context_variables
                raw_result = func(**args)
                result: Result = self._handle_function_result(raw_result, debug)
                tool_response = self.client.prepare_tool_response(
                    tool_call_id=tool_call_id,
                    tool_name=name,
                    content=result.value
                )
                partial_response.messages.append(tool_response)
                partial_response.context_variables.update(result.context_variables)
                if result.agent:
                    partial_response.agent = result.agent
                    return partial_response

            except Exception as e:
                error_message = f"Error processing tool call: {str(e)}"
                debug_print(debug, error_message)
                partial_response.messages.append({
                    "role": "tool",  #TODO: OAI lets you use tool, Anthropic needs user
                    "tool_call_id": tool_call.get('id', 'unknown'),
                    "tool_name": tool_call['function']['name'],
                    "content": error_message,
                })

        return partial_response

    def _handle_function_result(self, result: Any, debug: bool) -> Result:
        if isinstance(result, Result):
            return result
        elif isinstance(result, Agent):
            return Result(value=json.dumps({"assistant": result.name}), agent=result)
        else:
            try:
                return Result(value=str(result))
            except Exception as e:
                error_message = f"Failed to cast response to string: {result}. Make sure agent functions return a string or Result object. Error: {str(e)}"
                debug_print(debug, error_message)
                raise TypeError(error_message)

    def run(
        self,
        agent: Agent,
        messages: List[Dict[str, Any]],
        context_variables: Dict[str, Any] = {},
        model_override: str = None,
        stream: bool = False,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ) -> Response:
        active_agent = agent
        context_variables = context_variables.copy()
        history = messages.copy()
        init_len = len(messages)
        turn_count = 0

        while turn_count < max_turns and active_agent:
            print(f"Turn {turn_count} - Active agent: {active_agent.name}")
            
            # Get LLM completion
            completion = self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=stream,
                debug=debug
            )

            print(completion)
            
            # Parse response
            message = self.client.parse_response(completion)
            message['sender'] = active_agent.name

            # Update history
            history.append(message)

            # Handle tool calls if applicable
            tool_calls = message.get('tool_calls', [])

            if not tool_calls or not execute_tools:
                print("Ending turn. No tool calls or tool execution disabled.")
                break

            partial_response = self.handle_tool_calls(
                tool_calls, active_agent.functions, context_variables, debug
            )

            # Update history and context variables
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)

            # Update agent if applicable
            if partial_response.agent:
                active_agent = partial_response.agent
                print("Agent updated to:", active_agent)

            turn_count += 1

        print("Run method complete. Returning response.")
        return Response(
            messages=history[init_len:],
            agent=active_agent,
            context_variables=context_variables,
        )