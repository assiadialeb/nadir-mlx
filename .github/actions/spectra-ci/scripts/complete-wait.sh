#!/usr/bin/env bash
set -euo pipefail

: "${SPECTRA_URL:?}"
: "${API_KEY:?}"
: "${SCAN_ID:?}"

CE_TASK_ID=""
if [ -f .scannerwork/report-task.txt ]; then
  CE_TASK_ID=$(awk -F= '$1=="ceTaskId"{print $2}' .scannerwork/report-task.txt | tr -d '\r')
fi

# Advisory mode never blocks a merge (gate_mode from /scans registration):
# errors and timeouts are reported as warnings and the job succeeds.
ADVISORY=false
[ "${GATE_MODE:-advisory}" = "blocking" ] || ADVISORY=true

# finish <message>: fail the job in blocking mode, warn in advisory mode.
finish() {
  if [ "$ADVISORY" = "true" ]; then
    echo "::warning::$1 Advisory mode does not block the merge."
    exit 0
  fi
  echo "::error::$1"
  exit 1
}

if ! curl -sf -X POST "${SPECTRA_URL%/}/api/v1/ci/scans/${SCAN_ID}/complete" \
  -H "X-CI-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"ce_task_id\":\"$CE_TASK_ID\"}"; then
  finish "Spectra rejected the scan completion (scan ${SCAN_ID})."
fi

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
      if [ "$STATUS" = "error" ]; then
        finish "Spectra could not evaluate the scan (status=error)."
      fi
      exit 0
      ;;
  esac
  sleep "$INTERVAL"
done

echo "status=timeout" >> "$GITHUB_OUTPUT"
echo "should_fail=$([ "$ADVISORY" = "true" ] && echo false || echo true)" >> "$GITHUB_OUTPUT"
echo "Timeout waiting for Spectra verdict after ${WAIT_TIMEOUT}s (WAIT_TIMEOUT)."
echo "This is a CI wait timeout, not a Sonar analysis failure: Spectra kept the scan non-terminal."
echo "Raise wait-timeout (must exceed Spectra SONAR_CE_MAX_WAIT_SECONDS) or check the Spectra worker."
finish "Timed out waiting for the Spectra verdict."
