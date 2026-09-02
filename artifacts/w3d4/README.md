# Week 3, Day 4: quantise and lock the model

Environment: Google Colab, Tesla T4 GPU (15360 MiB)
Isolated venv (Python 3.10, `/content/venv`) - same setup pattern as day 3.

The model locked today is the model served for the rest of the course. After
this week, what gets served does not change; only how it is operated does.
The function-calling smoke test is the real gate: an endpoint the agentic
cohort cannot get reliable tool calls from is useless to them regardless of
how fast or small it is.

## Predictions (filled before running anything)

| Question | Prediction |
|---|---|
| AWQ memory.used vs yesterday's fp16, same --gpu-memory-utilization 0.85 | About the same, not much lower |
| Tokens/s with vLLM's fused AWQ kernels vs fp16 | Same range or faster (different story than day 1's bitsandbytes) |
| Smoke test: valid tool calls out of the 8 attempts that want one | Most or all of them |

## Actual results

### AWQ memory: the counter-intuitive result, confirmed exactly

```
memory.used: 11723 MiB
```

This matches the lab's own reference T4 run (11.7 GB) almost exactly.
Despite AWQ storing weights at 4-bit (roughly a quarter of fp16's bytes),
`nvidia-smi` barely moved from fp16's resident figure. The reason: with
`--gpu-memory-utilization 0.85` set, vLLM spends whatever memory the smaller
weights free up on more KV-cache blocks, up to that same fraction. The
savings are real, but they show up as capacity, not as a smaller number.

Confirmed directly from the server log:

```
# GPU blocks: 22955, # CPU blocks: 9362
```

This is where the freed memory actually went - more KV-cache headroom for
concurrent requests, not a lower `nvidia-smi` reading.

![AWQ memory.used (11723 MiB) and GPU blocks confirmed from server log](images/W3D4-1-vram-and-gpu-blocks-awq.png)

### AWQ throughput: faster than fp16, unlike day 1's bitsandbytes path

```
AWQ: 128 tokens in 2.30s = 55.7 tokens/s
```

Compared against day 3's fp16-via-vLLM concurrency-1 number (47.9
tokens/s), AWQ came out **faster**, not slower. This is the opposite of day
1's bitsandbytes result, where int8 quantisation cost real speed - the
difference is vLLM's fused AWQ kernels versus bitsandbytes' unfused int8
path.

![AWQ single-request throughput: 55.7 tokens/s, faster than fp16's 47.9](images/W3D4-2-awq-tokens-per-s.png)

### Five-prompt quality spot check: fp16 vs AWQ

Both models handled four of the five prompts comparably well. The one
prompt worth flagging: the tool-call scenario ("weather in Riyadh and time
in Tokyo"), asked as a plain chat completion **without** a `tools`
parameter, so neither model had a real function-calling contract to work
against.

- **AWQ** invented plausible-looking but fictitious API URLs
  (`api.weather.com/w/chat`, `api.timeZone.com`) - a fabrication, not a
  parsed tool call.
- **fp16** produced a cleaner, more structured JSON sketch with sensible
  tool names (`WeatherForecast`, `Time`) - still not a real tool call, but
  a more coherent guess at the shape of one.

Neither output is the real signal here, since the real test - with an
actual `tools` schema and a parser that can produce genuine `tool_calls` -
is the smoke test below. This spot check is judgment recorded, not scored,
per the lab's own framing.

The quantisation explanation prompt was worth keeping on record for
contrast: AWQ answered with an unusually vivid analogy ("stuffing your
clothing with fewer hamsters than you realistically need"), while fp16
gave a more literally technical explanation (rounding continuous values
into discrete units). Both were correct; AWQ's was simply more
idiosyncratic.

### The function-calling smoke test: both models score a perfect 10/10

**fp16** (relaunched with the same `--enable-auto-tool-choice
--tool-call-parser hermes` flags):

```
{'model': 'Qwen/Qwen2.5-1.5B-Instruct', 'total_attempts': 10, 'score': 10,
 'distractor_attempts': 2, 'distractor_call_free': 2,
 'distractor_majority_clean': True, 'passed': True}
```

**AWQ**:

```json
{
  "model": "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
  "total_attempts": 10,
  "score": 10,
  "distractor_attempts": 2,
  "distractor_call_free": 2,
  "distractor_majority_clean": true,
  "per_prompt": {
    "two_tool": {"k": 4, "wants_call": true, "valid": 4, "call_free": 0},
    "single": {"k": 4, "wants_call": true, "valid": 4, "call_free": 0},
    "distractor": {"k": 2, "wants_call": false, "valid": 0, "call_free": 2}
  },
  "passed": true
}
```

Both models hit the maximum possible score with the real `tools` schema in
place: every two-tool and single-tool prompt produced valid parseable
`tool_calls`, and both distractor attempts correctly stayed call-free -
the restraint half of the gate, not just the obedience half.

## The lock decision: AWQ

```
Model: Qwen/Qwen2.5-1.5B-Instruct-AWQ
Quantization: awq
Flags: --dtype half --max-model-len 4096 --gpu-memory-utilization 0.85
       --enable-auto-tool-choice --tool-call-parser hermes
Smoke Test Score: 10/10 (distractor_majority_clean: True)
```

Reasoning: AWQ matched fp16's perfect smoke score exactly while running
faster (55.7 vs 47.9 tokens/s single-request) and providing meaningfully
more KV-cache headroom (22955 GPU blocks) for concurrent load. With both
candidates passing the real gate identically, the quantised build's speed
and capacity advantage broke the tie - the memory-savings-as-capacity
argument this morning's Cell 2 made in the abstract turned out to matter
concretely once both models cleared the bar this afternoon set.

## Notes

- The isolated Python 3.10 virtual environment (`/content/venv`) had to be
  rebuilt from scratch mid-session after a two-hour idle gap silently
  disconnected the Colab runtime. `nvidia-smi` still reported a healthy T4
  afterward - the GPU attachment survived - but everything under
  `/content/` (the venv, all installed packages) was gone, since it lives
  on the runtime's ephemeral storage, not anywhere persistent. A
  `FileNotFoundError` on `/content/venv/bin/python` was the tell; the fix
  was rerunning the full install cell from `apt-get` onward, not just the
  vLLM launch.
- AWQ has its own separate weights repository from fp16 (a different
  Hugging Face repo, not a runtime conversion), so the first AWQ launch
  after any environment rebuild is a genuine cold download, not a
  cache hit.
- The tool-call parser flag is per model family, not universal:
  Qwen2.5 and Hermes-3 use `hermes`; Llama-3.1 needs `llama3_json`.
  Mismatching it produces tool requests as prose in `message.content`
  instead of parsed `tool_calls` - a named failure mode this session
  avoided by keeping the parser fixed across both fp16 and AWQ launches.
- Only one vLLM server ran at a time throughout the day - fp16 and AWQ
  were never loaded simultaneously - to avoid the OOM risk of two full
  model loads competing for the same T4.

Files: `smoke_result.json`, `model-lock.md`
