from abc import ABC, abstractmethod
from typing import Dict, Any, List

class LLMClient(ABC):
    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    def stream_chat_completion(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        pass

    @abstractmethod
    def prepare_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def prepare_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def parse_response(self, response: Any) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def prepare_chat_params(self, **kwargs) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def prepare_system_message(self, instructions: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def prepare_tool_response(self, tool_call_id: str, tool_name: str, content: str) -> Dict[str, Any]:
        pass