"""LLM Factory for creating LLM instances."""
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.schema import BaseLanguageModel
from loguru import logger

from app.config import settings


class LLMFactory:
    """Factory for creating LLM instances."""

    @staticmethod
    def create_llm(
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> BaseLanguageModel:
        """
        Create LLM instance based on provider.

        Args:
            provider: LLM provider ('openai' or 'ollama')
            model: Model name
            temperature: Temperature for generation
            **kwargs: Additional parameters

        Returns:
            LLM instance
        """
        provider = provider or settings.llm_provider
        temperature = temperature or settings.llm_temperature

        logger.info(f"Creating LLM with provider: {provider}")

        if provider == "openai":
            return LLMFactory._create_openai_llm(model, temperature, **kwargs)
        elif provider == "ollama":
            return LLMFactory._create_ollama_llm(model, temperature, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    @staticmethod
    def _create_openai_llm(
        model: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> ChatOpenAI:
        """Create OpenAI LLM instance."""
        model = model or settings.llm_model

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment variables")

        logger.info(f"Creating OpenAI LLM with model: {model}")

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=settings.openai_api_key,
            max_tokens=settings.llm_max_tokens,
            **kwargs
        )

    @staticmethod
    def _create_ollama_llm(
        model: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Ollama:
        """Create Ollama LLM instance."""
        model = model or settings.ollama_model

        logger.info(f"Creating Ollama LLM with model: {model}")

        return Ollama(
            model=model,
            temperature=temperature,
            base_url=settings.ollama_base_url,
            **kwargs
        )


# Convenience function
def get_llm(**kwargs) -> BaseLanguageModel:
    """Get default LLM instance."""
    return LLMFactory.create_llm(**kwargs)
