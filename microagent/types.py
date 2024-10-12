from typing import List, Callable, Union, Optional, Dict, Any
from pydantic import BaseModel

# Remove OpenAI-specific imports
# from openai.types.chat import ChatCompletionMessage
# from openai.types.chat.chat_completion_message_tool_call import (
#     ChatCompletionMessageToolCall,
#     Function,
# )

AgentFunction = Callable[[], Union[str, "Agent", dict]]

class Function(BaseModel):
    arguments: str
    name: str

class ChatCompletionMessageToolCall(BaseModel):
    id: str
    function: Function
    type: str

class ChatCompletionMessage(BaseModel):
    content: Optional[str]
    role: str
    tool_calls: Optional[List[ChatCompletionMessageToolCall]]

class Agent(BaseModel):
    name: str
    instructions: Union[str, Callable[..., str]]
    model: str
    functions: List[Callable] = []
    tool_choice: Optional[str] = None
    parallel_tool_calls: bool = True

class Response(BaseModel):
    messages: List[Dict[str, Any]]
    agent: Optional[Agent]
    context_variables: Dict[str, Any]
class Result(BaseModel):
    value: str = ""
    agent: Optional[Agent] = None
    context_variables: Dict[str, Any] = {}