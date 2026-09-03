# AIDC Bootcamp - Khawlah's Serving Stack

**Locked model:** `Qwen/Qwen2.5-1.5B-Instruct-AWQ`
`--dtype half --max-model-len 4096 --gpu-memory-utilization 0.85 --enable-auto-tool-choice --tool-call-parser hermes`

## Week 2

| Day | Topic | Result |
|---|---|---|
| [w2d1](artifacts/w2d1/) | Model memory and precision | Extra Lab: budget solver |
| [w2d2](artifacts/w2d2/) | FastAPI wrapper | Bug Lab: fixed |
| [w2d3](artifacts/w2d3/) | Containerise | Bug Lab: fixed |
| [w2d4](artifacts/w2d4/) | GPU image + CPU fallback | Bug Lab: fixed |
| [w2d5](artifacts/w2d5/) | Docker Compose + auth | Bug Lab: fixed |

## Week 3

| Day | Topic | Result |
|---|---|---|
| [w3d1](artifacts/w3d1/) | GPU profiling | Bug Lab + Extra Lab: PASS |
| [w3d2](artifacts/w3d2/) | Inference anatomy | Bug Lab + Extra Lab: PASS |
| [w3d3](artifacts/w3d3/) | Engine swap (vLLM) | Bug Lab + Extra Lab: PASS |
| [w3d4](artifacts/w3d4/) | Quantise and lock | AWQ locked, 10/10 smoke test |
| [w3d5](artifacts/w3d5/) | Benchmark harness | Knee: concurrency 2, 176 tok/s |

Full write-up (predictions, results, screenshots) in each day's own README.

Colab notebooks archived in [`artifacts/colab-notebooks/`](artifacts/colab-notebooks/).
