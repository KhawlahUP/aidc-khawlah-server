Week 2, Day 2: wrap the model

Environment: Windows laptop (local, CPU) Model: Qwen/Qwen2.5-0.5B-Instruct

Predictions (filled before writing any route)
Question	Prediction
usage.prompt_tokens for a 10-word user message, 32 tokens requested	~20-25
usage.completion_tokens for the same request	~30-32 (close to the max_tokens requested, since the model usually keeps generating until it hits the limit unless it naturally stops)
Which route passes its test first, with least code?	/v1/models — it is a static response with no model inference involved, just building and returning a ModelList object.
Does an unmodified openai client work with only a base_url change?	Yes — because the server implements the same OpenAI-compatible /v1 contract (same field names, same response shape), so the openai Python client doesn't need any code changes, only pointing base_url at the local server.
Actual results (after running verify.py / client_test.py)
Field	Measured
usage.prompt_tokens	24 (client_test.py) / 35 (verify.py, different prompt)
usage.completion_tokens	12 (client_test.py) / 3 (verify.py)
finish_reason	stop
client_test.py	✅ pass — reply: "Three primary colors are red, yellow, and blue."
verify.py	✅ GREEN CHECK: PASS
Screenshots

Show Image Show Image Show Image Show Image

Notes
The route implementations, non-streaming and streaming, live in app/main.py.
app/schemas.py is hardened per the Extra Lab (contract fuzzing): empty messages, non-positive max_tokens, out-of-range temperature, and a trailing assistant message are all rejected with 422 before the handler runs.
The chat route calls apply_chat_template(..., return_dict=True) and reads encoded["input_ids"] explicitly rather than assuming .shape on the raw return value. This is the fix from the Bug Lab (the upgrade that broke the contract): it is stable across the transformers major version that changed what apply_chat_template returns by default, so the workaround does not need to be reapplied if the pin changes later.
Generation blocks the event loop this week; that is the named, accepted limitation. Concurrency is Week 3's engine's job.
Streaming (Delta Step 5, optional) is implemented: verify.py confirmed "streaming: implemented, saw SSE chunks and [DONE] terminator".
Concurrency probe result: wall=4.030s, sum-of-individual=7.521s (looks concurrent). This is worth flagging rather than treating as routine: the lab's own failure-mode notes say a single-worker sync FastAPI service serving a real model should show serial execution, since model.generate() is synchronous CPU-bound work. A "looks concurrent" verdict here is more likely explained by PyTorch's internal CPU threading (BLAS/MKL splitting a single generate() call across multiple cores) than by the two requests genuinely running in parallel — but this is an observation to note, not a bug that was fixed, since the route itself has no concurrency logic added.
Extra lab: contract fuzzing

12/12 cases passed: yes Concurrency probe verdict: "looks concurrent" (wall=4.030s, sum-of-individual=7.521s) — see note above; flagged as unexpected rather than treated as the routine "looks serial" result.

Bug lab: the upgrade that broke the contract

pip show transformers version installed: 4.46.2 type(apply_chat_template(...)) before the fix: with return_dict=True the call returns a dict-like BatchEncoding (accessed via encoded["input_ids"]) rather than a raw Tensor with .shape directly on it — this is exactly the shape change the Bug Lab describes, and return_dict=True sidesteps it by being explicit about what's returned instead of relying on the default.

Separately: AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32) raised TypeError: Qwen2ForCausalLM.__init__() got an unexpected keyword argument 'dtype' on transformers==4.46.2 — this version only accepts the (deprecated-but-working) torch_dtype= keyword, not the newer dtype= name. Fixed by using torch_dtype=torch.float32.

Files: app/main.py, app/schemas.py, app/requirements.txt, app/client_test.py, app/verify.py, app/fuzz_client.py, app/requests.md