from __future__ import annotations

import asyncio
import json
import os
from abc import ABC, abstractmethod
from typing import Any

import httpx
import structlog

logger = structlog.get_logger()


class BaseLLMProvider(ABC):
    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 60,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    @abstractmethod
    async def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        ...


class OpenAIProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 60,
    ) -> None:
        super().__init__(model, temperature, max_tokens, timeout)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        if not self.api_key:
            logger.warning("openai_api_key_missing", fallback="mock")
            self._client = None
        else:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self.api_key, base_url=self.base_url, timeout=timeout
            )

    async def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        if not self._client:
            return f"[Mock OpenAI] {messages[-1]['content'][:100]}..."
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error("openai_completion_error", error=str(e))
            raise


class AnthropicProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-3-5-sonnet-20240620",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 60,
    ) -> None:
        super().__init__(model, temperature, max_tokens, timeout)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("anthropic_api_key_missing", fallback="mock")
            self._client = None
        else:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self.api_key, timeout=timeout)

    async def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        if not self._client:
            return f"[Mock Anthropic] {messages[-1]['content'][:100]}..."
        try:
            system_msg = next(
                (m["content"] for m in messages if m.get("role") == "system"), None
            )
            user_messages = [m for m in messages if m.get("role") != "system"]
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                system=system_msg,
                messages=user_messages,  # type: ignore
            )
            return response.content[0].text if response.content else ""
        except Exception as e:
            logger.error("anthropic_completion_error", error=str(e))
            raise


class OllamaProvider(BaseLLMProvider):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 60,
    ) -> None:
        super().__init__(model, temperature, max_tokens, timeout)
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", self.temperature),
                    "num_predict": kwargs.get("max_tokens", self.max_tokens),
                },
            }
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"] if "message" in data else data.get("response", "")
        except Exception as e:
            logger.error("ollama_completion_error", error=str(e))
            raise

    async def list_models(self) -> list[str]:
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error("ollama_list_models_error", error=str(e))
            return []


class LLMProviderFactory:
    _providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider,
    }

    @classmethod
    def create(cls, provider: str, **kwargs: Any) -> BaseLLMProvider:
        provider_class = cls._providers.get(provider)
        if not provider_class:
            raise ValueError(
                f"Unknown LLM provider: {provider}. "
                f"Available: {list(cls._providers.keys())}"
            )
        return provider_class(**kwargs)
