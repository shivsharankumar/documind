# src/documind/llm.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import anthropic
import openai
from anthropic import Anthropic
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import settings

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    role: Role
    content: str


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    finish_reason: str
    raw: Any  # original provider response, for debugging


class LLM:
    """Provider-agnostic LLM wrapper.

    Why this exists:
    - Single retry policy
    - Single place to log/trace (we'll add this in Chapter 6)
    - Easy to swap providers
    - Easy to mock in tests
    """

    def __init__(
        self,
        provider: str = settings.default_provider,
        model: str = settings.default_model,
    ):
        self.provider = provider
        self.model = model
        if provider == "openai":
            self._client = OpenAI(api_key=settings.openai_api_key)
        elif provider == "anthropic":
            self._client = Anthropic(api_key=settings.anthropic_api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type(
            (
                openai.RateLimitError,
                openai.APIConnectionError,
                anthropic.RateLimitError,
                anthropic.APIConnectionError,
            )
        ),
        reraise=True,
    )
    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        response_format: dict | None = None,
    ) -> LLMResponse:
        if self.provider == "openai":
            return self._complete_openai(messages, temperature, max_tokens, response_format)
        return self._complete_anthropic(messages, temperature, max_tokens)

    def _complete_openai(self, messages, temperature, max_tokens, response_format):
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        choice = resp.choices[0]
        return LLMResponse(
            text=choice.message.content or "",
            model=resp.model,
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            finish_reason=choice.finish_reason,
            raw=resp,
        )

    def _complete_anthropic(self, messages, temperature, max_tokens):
        # Anthropic separates system from messages
        sys = next((m.content for m in messages if m.role == "system"), None)
        msgs = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        resp = self._client.messages.create(
            model=self.model,
            system=sys,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return LLMResponse(
            text="".join(b.text for b in resp.content if b.type == "text"),
            model=resp.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            finish_reason=resp.stop_reason,
            raw=resp,
        )
