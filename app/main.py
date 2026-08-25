"""serving-stack: the FastAPI service (week 2, CPU, tiny model).

This is the starter. GET /health is done for you and works as soon as the model
loads: treat it as the worked example. Your job is the two routes marked TODO.
Correctness before speed. The model runs on CPU this week; do not add a GPU.

Run it:
    uvicorn main:app --host 0.0.0.0 --port 8000

Model: Qwen/Qwen2.5-0.5B-Instruct (about 0.5B params; loads on CPU in seconds
once cached). The first ever load downloads weights; the prep-week verify-env
pass pre-seeded the Hugging Face cache, so a cached load is fast.
"""
from __future__ import annotations

import json
import os
import time
import uuid

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer

from schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ModelCard,
    ModelList,
    HealthResponse,
    ResponseMessage,
    Usage,
)

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
device = "cpu"  # CPU only this week; do not add a GPU.

app = FastAPI(title="serving-stack", version="wk2")

# Load once at import time. CPU only this week.
print(f"loading {MODEL_ID} on cpu ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
model.to("cpu")
model.eval()
print("model ready")


# ---------------------------------------------------------------------------
# GET /health  -- DONE. This is the worked example. Copy its shape.
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness and readiness.

    Contract: returns 200 with {"status": "ok", "model": "<id>"} once the model
    is loaded. Kubernetes probes (week 4) and the agentic client's retry logic
    (weeks 4 to 6) call this. It must be cheap and must not run the model.
    """
    return HealthResponse(status="ok", model=MODEL_ID)


# ---------------------------------------------------------------------------
# GET /v1/models  -- TODO
# ---------------------------------------------------------------------------
@app.get("/v1/models", response_model=ModelList)
def list_models() -> ModelList:
    """List the served model id(s).

    Contract (OpenAI-compatible):
      response body: {"object": "list", "data": [ {ModelCard}, ... ]}
      each ModelCard has: id (== MODEL_ID), object == "model", created (unix
      seconds), owned_by.
    Week 2 serves exactly one model, so data has one entry: MODEL_ID.

    Build a ModelList from schemas.py and return it. Use int(time.time()) for
    created.
    """
    return ModelList(
        data=[ModelCard(id=MODEL_ID, created=int(time.time()))]
    )


# ---------------------------------------------------------------------------
# POST /v1/chat/completions  -- TODO (non-streaming first)
# ---------------------------------------------------------------------------
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
    """Run the model over the messages and return an OpenAI-compatible completion.

    Contract (non-streaming, the week-2 target):
      request:  ChatCompletionRequest (model, messages[], max_tokens, temperature)
      response: ChatCompletionResponse with
        id            a unique string, e.g. "chatcmpl-" + uuid4().hex
        object        "chat.completion"
        created       int(time.time())
        model         req.model (echo it back today; the reference rejects
                        unknown ids with a 400 model_not_found - match that
                        behaviour once your served id is stable, because the
                        consumer's client checks the id character for character)
        choices[0]    Choice(message=ResponseMessage(role="assistant",
                        content=<generated text>), finish_reason="stop" or "length")
        usage         Usage(prompt_tokens, completion_tokens, total_tokens),
                        all non-negative and total == prompt + completion

    Suggested steps:
      1. Build the prompt with the chat template:
           input_ids = tokenizer.apply_chat_template(
               [m.model_dump() for m in req.messages],
               add_generation_prompt=True, return_tensors="pt")
      2. prompt_tokens = input_ids.shape[1]
      3. Generate (no_grad, do_sample based on temperature > 0):
           out = model.generate(input_ids, max_new_tokens=req.max_tokens)
      4. new_tokens = out[0][prompt_tokens:]; completion_tokens = len(new_tokens)
      5. text = tokenizer.decode(new_tokens, skip_special_tokens=True)
      6. finish_reason = "length" if completion_tokens >= req.max_tokens else "stop"
      7. Assemble and return the ChatCompletionResponse.

    Generation blocks the event loop this week. That is acceptable: week 3's
    engine owns concurrency. Name it, do not solve it here.
    """
    # Match the reference: reject an unknown model id with 400, character for
    # character against the id we actually serve.
    if req.model != MODEL_ID:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"model '{req.model}' not found", "type": "model_not_found"}},
        )

    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    # NOTE: return_dict=True is requested explicitly rather than relying on
    # the default return type of apply_chat_template. Newer transformers
    # majors changed that default; asking for the dict keeps this call site
    # correct across versions instead of assuming .shape exists on whatever
    # comes back. (Bug Lab W2D2: the upgrade that broke the contract.)
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    prompt_tokens = input_ids.shape[1]

    do_sample = req.temperature > 0.0

    if req.stream:
        return StreamingResponse(
            _stream_chat_completion(input_ids, attention_mask, req, prompt_tokens),
            media_type="text/event-stream",
        )

    with torch.no_grad():
        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=req.max_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        if do_sample:
            gen_kwargs["temperature"] = req.temperature
        out = model.generate(**gen_kwargs)

    new_tokens = out[0][prompt_tokens:]
    completion_tokens = len(new_tokens)
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    finish_reason = "length" if completion_tokens >= req.max_tokens else "stop"

    return ChatCompletionResponse(
        id="chatcmpl-" + uuid.uuid4().hex,
        created=int(time.time()),
        model=req.model,
        choices=[
            Choice(
                index=0,
                message=ResponseMessage(role="assistant", content=text),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


# ---------------------------------------------------------------------------
# Streaming (Delta Step 5). Not required for the green check; verify.py
# probes it and reports whether it is implemented, but does not fail without
# it. Generation still runs synchronously to completion under the hood (the
# model has no incremental/streaming generate path added this week) — tokens
# are decoded one at a time from the full output and yielded as SSE chunks so
# the wire format matches the OpenAI streaming contract.
# ---------------------------------------------------------------------------
def _stream_chat_completion(input_ids, attention_mask, req, prompt_tokens):
    completion_id = "chatcmpl-" + uuid.uuid4().hex
    created = int(time.time())
    do_sample = req.temperature > 0.0

    with torch.no_grad():
        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=req.max_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        if do_sample:
            gen_kwargs["temperature"] = req.temperature
        out = model.generate(**gen_kwargs)

    new_tokens = out[0][prompt_tokens:]
    completion_tokens = len(new_tokens)
    finish_reason = "length" if completion_tokens >= req.max_tokens else "stop"

    def _chunk(delta: dict, finish: str | None = None) -> str:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    # first chunk announces the role, matching the OpenAI SSE shape
    yield _chunk({"role": "assistant"})

    for token_id in new_tokens:
        piece = tokenizer.decode([token_id], skip_special_tokens=True)
        if piece:
            yield _chunk({"content": piece})

    yield _chunk({}, finish=finish_reason)
    yield "data: [DONE]\n\n"