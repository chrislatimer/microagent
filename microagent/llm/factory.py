from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from .groq_client import GroqClient
from .gemini_client import GeminiClient

class LLMFactory:
    @staticmethod
    def create(llm_type):
        if llm_type == 'openai':
            return OpenAIClient()
        elif llm_type == 'anthropic':
            return AnthropicClient()
        elif llm_type == 'groq':
            return GroqClient()
        elif llm_type == 'gemini':
            return GeminiClient()
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")