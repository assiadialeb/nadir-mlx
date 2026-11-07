#!/usr/bin/env bash
set -euo pipefail

: "${SPECTRA_URL:?}"
: "${API_KEY:?}"
: "${SCAN_ID:?}"

CE_TASK_ID=""
if [ -f .scannerwork/report-task.txt ]; then
  CE_TASK_ID=$(awk -F= '$1=="ceTaskId"{print $2}' .scannerwork/report-task.txt | tr -d '\r')
fi

curl -sf -X POST "${SPECTRA_URL%/}/api/v1/ci/scans/${SCAN_ID}/complete" \
  -H "X-CI-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"ce_task_id\":\"$CE_TASK_ID\"}"

TIMEOUT="${WAIT_TIMEOUT:-600}"
INTERVAL="${WAIT_INTERVAL:-10}"
DEADLINE=$((SECONDS + TIMEOUT))

while [ "$SECONDS" -lt "$DEADLINE" ]; do
  RESPONSE=$(curl -sS "${SPECTRA_URL%/}/api/v1/ci/scans/${SCAN_ID}/status" \
    -H "X-CI-Key: $API_KEY" || true)
  STATUS=$(echo "$RESPONSE" | jq -r '.data.status // empty' 2>/dev/null || true)
  if [ -z "$STATUS" ]; then
    sleep "$INTERVAL"
    continue
  fi
  SHOULD_FAIL=$(echo "$RESPONSE" | jq -r '.data.should_fail // false')
  case "$STATUS" in
    passed|failed|bypassed|error)
      echo "status=$STATUS" >> "$GITHUB_OUTPUT"
      echo "should_fail=$SHOULD_FAIL" >> "$GITHUB_OUTPUT"
      if [ "${FAIL_ON_GATE:-true}" = "true" ] && [ "$SHOULD_FAIL" = "true" ]; then
        echo "Spectra quality gate failed (status=$STATUS)."
        exit 1
      fi
      if [ "$STATUS" = "error" ] || [ "$STATUS" = "timeout" ]; then
        exit 1
      fi
      exit 0
      ;;
  esac
  sleep "$INTERVAL"
done

echo "status=timeout" >> "$GITHUB_OUTPUT"
echo "should_fail=true" >> "$GITHUB_OUTPUT"
echo "Timeout waiting for Spectra verdict."
exit 1
