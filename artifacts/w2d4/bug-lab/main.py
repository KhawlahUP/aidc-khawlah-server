import torch
from fastapi import FastAPI, HTTPException

app = FastAPI()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@app.post("/v1/embeddings")
def embeddings(payload: dict):
    if DEVICE != "cuda":
        raise HTTPException(400, "Embeddings require a GPU-backed instance; this instance is running in CPU-fallback mode.")
    return {"vector": [0.1] * 8, "device_used": DEVICE}

@app.post("/v1/chat/completions")
def chat_completions(payload: dict):
    if payload.get("require_gpu") and DEVICE != "cuda":
        raise HTTPException(400, "GPU required, running on CPU")
    return {"ok": True}