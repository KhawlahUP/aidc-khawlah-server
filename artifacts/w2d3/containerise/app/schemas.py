"""OpenAI-compatible request and response shapes.

This file is GIVEN, complete. The contract's teeth are not the exercise: your
job is to fill in the routes in main.py so they read these requests and return
these responses. Do not weaken these models. The Agentic AI cohort's client
(and the openai Python client) expects exactly these field names.

The `tools`/`tool_choice` fields are accepted from day 1 (the contract says a
consumer's payload always validates) and go unused until the tool-calling
engine at tier 1.
"""
from __future__ import annotations

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    """One turn in the conversation."""
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """The body of POST /v1/chat/completions."""
    model: str
    # min_length=1 rejects an empty messages list before any handler code runs
    # (Extra Lab W2D2, step 1).
    messages: List[ChatMessage] = Field(..., min_length=1)
    # optional generation controls, with OpenAI-compatible names and defaults.
    # max_tokens has no upper bound here: the reference CLAMPS oversized asks
    # to its MAX_TOKENS setting rather than rejecting them (day 5 sets it).
    # ge=1 rejects zero/negative max_tokens instead of letting it reach
    # model.generate().
    max_tokens: int = Field(default=256, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False
    # part of the course contract from day 1 so a consumer's payload always
    # validates; accepted and unused until the tool-calling engine (tier 1).
    tools: Optional[List[dict]] = None
    tool_choice: Optional[Union[str, dict]] = None

    @field_validator("messages")
    @classmethod
    def last_message_must_be_user_or_system(cls, v):
        """Reject a request that asks the model to continue its own turn."""
        if v and v[-1].role == "assistant":
            raise ValueError(
                "the last message must be from 'user' or 'system', not 'assistant'"
            )
        return v


class ResponseMessage(BaseModel):
    """The assistant message inside a completion choice."""
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    """One completion choice. Week 2 always returns exactly one (index 0)."""
    index: int = 0
    message: ResponseMessage
    finish_reason: Literal["stop", "length"] = "stop"


class Usage(BaseModel):
    """Token accounting. All three must be present and non-negative."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """The body returned by POST /v1/chat/completions (non-streaming)."""
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


class ModelCard(BaseModel):
    """One entry in GET /v1/models."""
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "aidc"


class ModelList(BaseModel):
    """The body returned by GET /v1/models."""
    object: Literal["list"] = "list"
    data: List[ModelCard]


class HealthResponse(BaseModel):
    """The body returned by GET /health."""
    status: Literal["ok"] = "ok"
    model: str
