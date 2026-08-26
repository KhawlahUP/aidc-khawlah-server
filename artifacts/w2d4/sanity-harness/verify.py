#!/usr/bin/env python3
# Green check for the extra W2D4 lab (device-agnostic sanity harness).
# Run with the server up:  python verify.py
# Prints exactly one line last: GREEN CHECK: PASS  or  GREEN CHECK: FAIL (<reason>)
# stdlib only, no code shared with the student's harness.
#
# Device-aware on purpose: it reads which device the server CLAIMS from
# /health, then holds it to the matching contract. On cpu, require_gpu=true
# must refuse with a clean 400 and a sentence a person could act on. On cuda,
# the same request must simply work. Both are passes for their environment;
# tier 0 is the cpu run.
import json, os, urllib.error, urllib.request
from typing import NoReturn

BASE = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 30


class _Stop(Exception):
    """Ends the check without killing a notebook kernel."""


def _fail(reason) -> NoReturn:
    print("GREEN CHECK: FAIL (%s)" % reason)
    raise _Stop()


def get_json(path):
    try:
        with urllib.request.urlopen(BASE + path, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except json.JSONDecodeError:
            return e.code, None
    except Exception as e:
        _fail("GET %s failed: %s (is the server running on %s?)" % (path, e, BASE))


def post(payload):
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
        method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, {"_raw": body.decode(errors="replace")[:300]}
    except Exception as e:
        _fail("POST /v1/chat/completions did not complete: %s" % e)


def main():
    status, health = get_json("/health")
    if status != 200 or not isinstance(health, dict):
        _fail("/health did not answer 200 with a JSON body")
    device = health.get("device")
    if device not in ("cpu", "cuda"):
        _fail("/health device is %r; the contract is 'cpu' or 'cuda'" % (device,))
    print("  server claims device=%s" % device)

    status, body = post({"prompt": "hello"})
    if status != 200:
        _fail("plain request failed with %s on %s; degradation must never break "
              "the normal path" % (status, device))
    if isinstance(body, dict) and body.get("device") not in (None, device):
        _fail("response says device=%r but /health says %r; the service is "
              "lying about itself somewhere" % (body.get("device"), device))

    status, body = post({"prompt": "hello", "require_gpu": True})
    if device == "cuda":
        if status != 200:
            _fail("require_gpu=true failed with %s on a cuda instance" % status)
        print("  ok  require_gpu succeeds on cuda")
    else:
        if status == 500:
            _fail("require_gpu=true on cpu returned 500; the refusal check runs "
                  "after device-specific code, move it first in the handler")
        if status != 400:
            _fail("require_gpu=true on cpu returned %s, expected a clean 400" % status)
        detail = (body or {}).get("detail", "")
        if not isinstance(detail, str) or len(detail) < 20 or "gpu" not in detail.lower():
            _fail("the 400 carries no usable message; detail=%r" % (detail,))
        if "Traceback" in json.dumps(body or {}):
            _fail("the 400 body contains a traceback; that is a crash in costume")
        print("  ok  require_gpu refused cleanly on cpu")

    status, _ = get_json("/health")
    if status != 200:
        _fail("/health broke after the run; something wedged the server")

    print("GREEN CHECK: PASS")


if __name__ == "__main__":
    try:
        main()
    except _Stop:
        raise SystemExit(1)
