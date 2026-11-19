#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

if [ "${SCAN_STATUS:-}" = "bypassed" ] || [ "${SAST_ENABLED:-false}" != "true" ]; then
  exit 0
fi
if [ "${SAST_MODE:-pack}" = "skip" ]; then
  exit 0
fi

# Blocking mode fails the job; advisory mode warns and skips SAST (SPCTR-308).
# No partial report is left behind, so nothing stale is uploaded.
sast_failed() {
  rm -f sast_output.json
  spectra_step_failed "$1" "${2:-1}" || exit $?
  exit 0
}

CONFIG_ARGS=()
if [ "${SAST_MODE}" = "local" ]; then
  # Each path needs its own --config: extra positional paths become scan targets.
  for config_path in ${SAST_LOCAL_CONFIGS:-rules/sast}; do
    CONFIG_ARGS+=(--config "$config_path")
  done
else
  # A download or integrity failure never scans: a tampered or empty pack would pass the gate.
  mkdir -p .spectra
  if ! curl -sf -H "X-CI-Key: $API_KEY" -o /tmp/spectra-sast-rules.tar.gz \
    "${SPECTRA_URL%/}/api/v1/ci/sast-rules"; then
    sast_failed "SAST rule pack download failed."
  fi
  # Use openssl: macOS sha256sum lacks GNU --check/--strict; Linux CI has both.
  if [ -n "${SAST_RULES_SHA256:-}" ]; then
    _got=$(openssl dgst -sha256 /tmp/spectra-sast-rules.tar.gz | awk '{print $NF}')
    if [ "$_got" != "${SAST_RULES_SHA256}" ]; then
      sast_failed "spectra-sast-rules.tar.gz: SHA256 mismatch (expected ${SAST_RULES_SHA256}, got ${_got})."
    fi
  fi
  if ! tar -xzf /tmp/spectra-sast-rules.tar.gz -C .spectra; then
    sast_failed "SAST rule pack extraction failed."
  fi
  CONFIG_ARGS+=(--config .spectra/sast)
fi

set +e
# shellcheck disable=SC2086
# Exclude annotated rule fixtures so CI gates measure application code, not intentional test vectors.
opengrep scan --json --taint-intrafile $SAST_EXTRA_ARGS "${CONFIG_ARGS[@]}" \
  --exclude 'rules/sast/fixtures/**' \
  --output sast_output.json .
SAST_EXIT=$?
set -e
if [ "$SAST_EXIT" -ne 0 ] && [ "$SAST_EXIT" -ne 1 ]; then
  sast_failed "OpenGrep failed with exit code ${SAST_EXIT}." "$SAST_EXIT"
fi
