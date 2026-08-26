import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="device-aware serving")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CompletionRequest(BaseModel):
    prompt: str
    require_gpu: bool = False


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/v1/chat/completions")
def chat_completions(req: CompletionRequest):
    if req.require_gpu and DEVICE != "cuda":
        raise HTTPException(
            status_code=400,
            detail=(
                "This request set require_gpu=true, but this instance is "
                "running in CPU-fallback mode (no CUDA device available). "
                "Retry against a GPU-backed instance, or drop require_gpu."
            ),
        )
    return {"reply": f"(on {DEVICE}) you said: {req.prompt}", "device": DEVICE}