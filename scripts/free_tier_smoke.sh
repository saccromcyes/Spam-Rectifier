#!/usr/bin/env bash
# Run the production image inside the limits of Render's free instance (512 MB RAM, 0.1 CPU),
# wait until it is healthy, send one real request, and report start time, request time and peak memory.
# Fails if the container is OOM-killed, never becomes healthy, or the request doesn't succeed.
#
#   HEALTH=/api/health METHOD=POST REQ_PATH=/api/ask BODY='{"question":"..."}' EXPECT='"done"' bash scripts/free_tier_smoke.sh
set -uo pipefail

IMAGE=${IMAGE:-app}
HEALTH=${HEALTH:-/health}
METHOD=${METHOD:-GET}
REQ_PATH=${REQ_PATH:-$HEALTH}
BODY=${BODY:-}
EXPECT=${EXPECT:-}
LIMIT_MB=512

docker rm -f smoke >/dev/null 2>&1 || true
docker run -d --name smoke --memory=${LIMIT_MB}m --memory-swap=${LIMIT_MB}m --cpus=0.1 \
  -e PORT=7860 -e ONNX_THREADS=1 -p 7860:7860 "$IMAGE" >/dev/null

fail() {
  echo "::error::$1"
  docker inspect -f 'OOMKilled={{.State.OOMKilled}} ExitCode={{.State.ExitCode}}' smoke 2>/dev/null
  echo "----- container log (last 60 lines) -----"
  docker logs smoke 2>&1 | tail -60
  exit 1
}

peak() { docker exec smoke cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo 0; }

start=$(date +%s)
until curl -fsS "http://localhost:7860${HEALTH}" >/dev/null 2>&1; do
  [ "$(docker inspect -f '{{.State.Running}}' smoke)" = "true" ] || fail "Container stopped before becoming healthy."
  [ $(( $(date +%s) - start )) -gt 360 ] && fail "Not healthy after 360 s."
  sleep 2
done
ready=$(( $(date +%s) - start ))
echo "Healthy after ${ready}s at 0.1 CPU."

t0=$(date +%s.%N)
if [ "$METHOD" = "POST" ]; then
  out=$(curl -sS -N --max-time 240 -X POST -H 'Content-Type: application/json' -d "$BODY" "http://localhost:7860${REQ_PATH}") || fail "Request failed."
else
  out=$(curl -sS --max-time 240 "http://localhost:7860${REQ_PATH}") || fail "Request failed."
fi
took=$(echo "$(date +%s.%N) - $t0" | bc)
[ -z "$EXPECT" ] || grep -q -- "$EXPECT" <<<"$out" || { echo "$out" | head -c 1500; fail "Response did not contain $EXPECT."; }
[ "$(docker inspect -f '{{.State.OOMKilled}}' smoke)" = "false" ] || fail "Container was OOM-killed."

peak_mb=$(( $(peak) / 1024 / 1024 ))
printf 'First request (%s %s) took %.1fs.\nPeak memory %d MB of %d MB.\n' "$METHOD" "$REQ_PATH" "$took" "$peak_mb" "$LIMIT_MB"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  {
    echo "### Render free tier check (512 MB, 0.1 CPU)"
    echo "| Start-up | First request | Peak memory |"
    echo "|---|---|---|"
    printf '| %ss | %.1fs | %d MB of %d MB |\n' "$ready" "$took" "$peak_mb" "$LIMIT_MB"
  } >> "$GITHUB_STEP_SUMMARY"
fi
docker rm -f smoke >/dev/null
