#!/usr/bin/env bash
# verify.sh for Extra Lab W2D3: multi-stage build golf
# Checks both images exist, reads their true sizes from docker image inspect,
# holds the multi-stage image to the 300 MB target and at least 20% savings,
# boots the multi-stage container, and exercises /health, /registry, one
# model lookup and the 404 path.
set -u

TARGET_MB=300
MIN_SAVINGS_PCT=20
PORT="${PORT:-8000}"
NAME="registry-verify"

fail() { echo "GREEN CHECK: FAIL ($1)"; cleanup; exit 1; }
cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
cleanup

# 1. both images must exist
naive_bytes=$(docker image inspect registry:naive --format='{{.Size}}' 2>/dev/null) \
  || fail "registry:naive image not found; build it first"
multi_bytes=$(docker image inspect registry:multistage --format='{{.Size}}' 2>/dev/null) \
  || fail "registry:multistage image not found; build it first"

naive_mb=$(python3 -c "print($naive_bytes/1024/1024)")
multi_mb=$(python3 -c "print($multi_bytes/1024/1024)")
savings_pct=$(python3 -c "print(($naive_bytes-$multi_bytes)/$naive_bytes*100)")

echo "naive:       ${naive_mb} MB"
echo "multistage:  ${multi_mb} MB"
echo "savings:     ${savings_pct}%"

python3 -c "exit(0 if $multi_mb <= $TARGET_MB else 1)" \
  || fail "multi-stage image ${multi_mb}MB exceeds ${TARGET_MB}MB target"
python3 -c "exit(0 if $savings_pct >= $MIN_SAVINGS_PCT else 1)" \
  || fail "savings ${savings_pct}% below ${MIN_SAVINGS_PCT}% minimum"

# 2. boot the multi-stage image and exercise the routes
docker run -d --name "$NAME" -p "${PORT}:8000" registry:multistage >/dev/null 2>&1 \
  || fail "docker run failed (port ${PORT} in use, or image will not start)"

deadline=$(( $(date +%s) + 30 ))
healthy=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/health" 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then healthy=1; break; fi
  sleep 1
done
[ "$healthy" -eq 1 ] || fail "/health did not return 200"

reg=$(curl -s "http://localhost:${PORT}/registry")
echo "$reg" | grep -q "Qwen2.5-0.5B-Instruct" || fail "/registry missing expected model"

lookup=$(curl -s "http://localhost:${PORT}/registry/Qwen2.5-1.5B-Instruct")
echo "$lookup" | grep -q "repo_id" || fail "/registry/{name} lookup failed"

code404=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/registry/nonexistent")
[ "$code404" = "404" ] || fail "/registry/nonexistent did not return 404"

cleanup
echo "GREEN CHECK: PASS"
exit 0
