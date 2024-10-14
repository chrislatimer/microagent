from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from .groq_client import GroqClient

class LLMFactory:
    @staticmethod
    def create(llm_type):
        if llm_type == 'openai':
            return OpenAIClient()
        elif llm_type == 'anthropic':
            return AnthropicClient()
        elif llm_type == 'groq':
            return GroqClient()
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")