"""Extra Lab W2D2: contract fuzzing.

Throws malformed, missing, out-of-range, and unusual input at your own
/v1/chat/completions and checks that every case fails (422) or succeeds (200)
on purpose -- no 500s, no hangs.

Start the server first:
    uvicorn main:app --host 0.0.0.0 --port 8000
then:
    python fuzz_client.py
"""
import os
import sys
import time
import concurrent.futures
import httpx

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")
MODEL = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
results = []


def case(name, expected_status, payload=None, raw_body=None, headers=None):
    expected = {expected_status} if isinstance(expected_status, int) else set(expected_status)
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    try:
        if raw_body is not None:
            r = httpx.post(f"{BASE_URL}/v1/chat/completions", content=raw_body, headers=hdrs, timeout=30)
        else:
            r = httpx.post(f"{BASE_URL}/v1/chat/completions", json=payload, headers=hdrs, timeout=30)
        ok = r.status_code in expected
        results.append((name, ok, r.status_code, expected))
        return ok, r
    except Exception as e:
        results.append((name, False, f"EXCEPTION: {e}", expected))
        return False, None


def run_cases():
    # --- should be rejected (422) ---
    case("missing 'messages' field", 422, payload={"model": MODEL, "max_tokens": 16})
    case("missing 'model' field", 422, payload={"messages": [{"role": "user", "content": "hi"}]})
    case("'messages' is a string, not a list", 422, payload={"model": MODEL, "messages": "hi"})
    case("empty 'messages' list", 422, payload={"model": MODEL, "messages": []})
    case("invalid role in message", 422, payload={"model": MODEL, "messages": [{"role": "wizard", "content": "hi"}]})
    case("negative max_tokens", 422, payload={"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": -5})
    case("max_tokens is zero", 422, payload={"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 0})
    case("temperature out of range", 422, payload={"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "temperature": 5.0})
    case("last message has role 'assistant'", 422, payload={"model": MODEL, "messages": [{"role": "assistant", "content": "hi"}]})
    case("malformed JSON body", 422, raw_body=b'{"model": "x", "messages": [', headers={"Content-Type": "application/json"})

    # --- should succeed (200) despite being unusual ---
    case("unicode/emoji content", 200, payload={"model": MODEL, "messages": [{"role": "user", "content": "hello \U0001F44B \u4e16\u754c"}], "max_tokens": 16})
    case("minimal valid single-word message", 200, payload={"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16})


def run_concurrency_probe(n=2):
    """Informational only -- does not affect pass/fail."""
    payload = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}

    def one():
        t0 = time.time()
        httpx.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=30)
        return time.time() - t0

    t_wall0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        durations = list(ex.map(lambda _: one(), range(n)))
    wall = time.time() - t_wall0
    serial_estimate = sum(durations)
    verdict = "looks serial" if wall > 0.8 * serial_estimate else "looks concurrent"
    print(f"\nconcurrency probe (n={n}): wall={wall:.3f}s, sum-of-individual={serial_estimate:.3f}s ({verdict})")


def main():
    run_cases()
    n_pass = sum(1 for _, ok, _, _ in results if ok)
    n_total = len(results)
    for name, ok, status, expected in results:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}]  {name:40s} got={status} expected={expected}")
    run_concurrency_probe()
    print(f"\n{n_pass}/{n_total} cases passed")
    print("GREEN CHECK: PASS" if n_pass == n_total else "GREEN CHECK: FAIL (see cases marked FAIL above)")
    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
