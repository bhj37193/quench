from __future__ import annotations

import os

from src.types import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    UsageInfo,
)

_UPSTREAM_BASE_URL = os.environ.get("UPSTREAM_BASE_URL", "https://api.openai.com/v1")
_UPSTREAM_API_KEY = os.environ.get("UPSTREAM_API_KEY", "")

# Auto-detect Anthropic: explicit UPSTREAM_PROVIDER=anthropic or URL contains "anthropic"
_IS_ANTHROPIC = (
    os.environ.get("UPSTREAM_PROVIDER", "").lower() == "anthropic"
    or "anthropic" in _UPSTREAM_BASE_URL
)

_openai_client = None
_anthropic_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import AsyncOpenAI
        _openai_client = AsyncOpenAI(
            base_url=_UPSTREAM_BASE_URL,
            api_key=_UPSTREAM_API_KEY,
        )
    return _openai_client


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.AsyncAnthropic(api_key=_UPSTREAM_API_KEY)
    return _anthropic_client


async def _forward_openai(req: ChatCompletionRequest) -> ChatCompletionResponse:
    client = _get_openai_client()
    messages = [{"role": m.role, "content": m.content or ""} for m in req.messages]

    kwargs: dict = {"model": req.model, "messages": messages, "stream": False}
    if req.temperature is not None:
        kwargs["temperature"] = req.temperature
    if req.max_tokens is not None:
        kwargs["max_tokens"] = req.max_tokens
    if req.top_p is not None:
        kwargs["top_p"] = req.top_p
    if req.frequency_penalty is not None:
        kwargs["frequency_penalty"] = req.frequency_penalty
    if req.presence_penalty is not None:
        kwargs["presence_penalty"] = req.presence_penalty
    if req.stop is not None:
        kwargs["stop"] = req.stop
    if req.n is not None:
        kwargs["n"] = req.n

    upstream = await client.chat.completions.create(**kwargs)

    choices = [
        ChatCompletionChoice(
            index=c.index,
            message=ChatCompletionMessage(role=c.message.role, content=c.message.content),
            finish_reason=c.finish_reason,
        )
        for c in upstream.choices
    ]
    usage = UsageInfo(
        prompt_tokens=upstream.usage.prompt_tokens if upstream.usage else 0,
        completion_tokens=upstream.usage.completion_tokens if upstream.usage else 0,
        total_tokens=upstream.usage.total_tokens if upstream.usage else 0,
    )
    return ChatCompletionResponse(
        id=upstream.id,
        object=upstream.object,
        created=upstream.created,
        model=upstream.model,
        choices=choices,
        usage=usage,
    )


async def _forward_anthropic(req: ChatCompletionRequest) -> ChatCompletionResponse:
    client = _get_anthropic_client()

    # Anthropic separates system prompt from conversation messages
    system_parts = []
    messages = []
    for msg in req.messages:
        if msg.role == "system":
            system_parts.append(msg.content or "")
        else:
            messages.append({"role": msg.role, "content": msg.content or ""})

    kwargs: dict = {
        "model": req.model,
        "messages": messages,
        "max_tokens": req.max_tokens or 1024,
    }
    if system_parts:
        kwargs["system"] = "\n".join(system_parts)
    if req.temperature is not None:
        kwargs["temperature"] = req.temperature
    if req.top_p is not None:
        kwargs["top_p"] = req.top_p
    if req.stop is not None:
        kwargs["stop_sequences"] = req.stop if isinstance(req.stop, list) else [req.stop]

    upstream = await client.messages.create(**kwargs)

    content = upstream.content[0].text if upstream.content else ""
    usage = UsageInfo(
        prompt_tokens=upstream.usage.input_tokens,
        completion_tokens=upstream.usage.output_tokens,
        total_tokens=upstream.usage.input_tokens + upstream.usage.output_tokens,
    )
    return ChatCompletionResponse(
        id=upstream.id,
        model=upstream.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=content),
                finish_reason=upstream.stop_reason or "stop",
            )
        ],
        usage=usage,
    )


async def forward(req: ChatCompletionRequest) -> ChatCompletionResponse:
    if _IS_ANTHROPIC:
        return await _forward_anthropic(req)
    return await _forward_openai(req)
