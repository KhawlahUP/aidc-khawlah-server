import os, sys
import httpx

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")

def check_health_reports_device():
    r = httpx.get(f"{BASE_URL}/health", timeout=10)
    if r.status_code != 200:
        return False, f"/health returned {r.status_code}"
    device = r.json().get("device")
    if device not in ("cpu", "cuda"):
        return False, f"/health device field is {device!r}, expected 'cpu' or 'cuda'"
    return True, device

def check_normal_request_works(device):
    r = httpx.post(f"{BASE_URL}/v1/chat/completions", json={"prompt": "hello"}, timeout=30)
    if r.status_code != 200:
        return False, f"normal request failed: {r.status_code} {r.text[:200]}"
    return True, None

def check_gpu_only_request(device):
    r = httpx.post(f"{BASE_URL}/v1/chat/completions",
                    json={"prompt": "hello", "require_gpu": True}, timeout=30)
    if device == "cpu":
        if r.status_code != 400:
            return False, f"expected clean 400 on CPU with require_gpu=true, got {r.status_code}: {r.text[:200]}"
        detail = r.json().get("detail", "")
        if not detail or "GPU" not in detail:
            return False, f"400 response has no useful message: {r.json()}"
        return True, None
    else:
        if r.status_code != 200:
            return False, f"expected 200 on GPU with require_gpu=true, got {r.status_code}: {r.text[:200]}"
        return True, None

def main():
    checks = []
    ok, device_or_reason = check_health_reports_device()
    checks.append(("health reports a valid device", ok, device_or_reason))
    if not ok:
        print_and_exit(checks)
    device = device_or_reason

    ok, reason = check_normal_request_works(device)
    checks.append(("normal request succeeds regardless of device", ok, reason))

    ok, reason = check_gpu_only_request(device)
    label = ("GPU-only request fails cleanly on CPU (400, clear message)" if device == "cpu"
             else "GPU-only request succeeds on real GPU")
    checks.append((label, ok, reason))
    print_and_exit(checks)

def print_and_exit(checks):
    all_ok = True
    for name, ok, reason in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}" + (f" ({reason})" if not ok and reason else ""))
        all_ok = all_ok and ok
    print("\nGREEN CHECK: PASS" if all_ok else "\nGREEN CHECK: FAIL (see FAIL lines above)")
    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()