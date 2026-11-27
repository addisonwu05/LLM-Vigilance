"""chat(): one call for every provider the experiments used; ask(): one turn of a dialogue.

Arguments that are ``None`` are *omitted* from the request, which matters: several original runs
sent no ``max_tokens`` or ``temperature`` at all, and sending a default is not always equivalent
to omitting it.
"""
from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any, Callable

from dotenv import load_dotenv

from .config import load_models

load_dotenv()

Message = dict[str, Any]
ChatFn = Callable[..., str]   # chat(), or a test double with the same signature


@lru_cache(maxsize=None)
def _client(provider: str):
    spec = load_models()["providers"][provider]
    key = os.environ.get(spec["api_key_env"])
    if not key:
        raise RuntimeError(f"Set {spec['api_key_env']} in .env to call the {provider} API")
    if provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=key)
    if provider == "azure_openai":
        from openai import AzureOpenAI
        endpoint = os.environ.get(spec["endpoint_env"])
        if not endpoint:
            raise RuntimeError(f"Set {spec['endpoint_env']} in .env to call Azure OpenAI")
        return AzureOpenAI(api_key=key, api_version=spec["api_version"], azure_endpoint=endpoint)
    if provider == "gemini":
        from openai import OpenAI
        return OpenAI(api_key=key, base_url=spec["base_url"])
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=key)
    if provider == "together":
        from together import Together
        return Together(api_key=key)
    raise ValueError(f"unknown provider {provider!r}")


def chat(provider: str, model_id: str, messages: list[Message], *, max_tokens: int | None = None,
         temperature: float | None = None, retries: int = 3) -> str:
    """Send one chat request and return the reply text."""
    client = _client(provider)
    kwargs: dict[str, Any] = {"model": model_id, "messages": messages}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    for attempt in range(retries + 1):
        try:
            if provider == "anthropic":
                if max_tokens is None:
                    raise ValueError("Anthropic requests need max_tokens")
                return client.messages.create(**kwargs).content[0].text
            return client.chat.completions.create(**kwargs).choices[0].message.content
        except Exception:  # noqa: BLE001 - retry any transient provider error
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def ask(chat_fn: ChatFn, spec: dict, history: list[Message], content: Any, **options) -> list[Message]:
    """Send ``history`` plus one user message to the model in ``spec`` (its provider and id).

    Returns a new list: the dialogue including the model's reply. ``options`` (max_tokens,
    temperature) are passed on to ``chat_fn``.
    """
    messages = [*history, {"role": "user", "content": content}]
    reply = chat_fn(spec["provider"], spec["id"], messages, **options)
    return [*messages, {"role": "assistant", "content": reply}]
