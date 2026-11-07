#!/usr/bin/env bash
set -euo pipefail

if [ "${SCAN_STATUS:-}" = "bypassed" ] || [ "${SAST_ENABLED:-false}" != "true" ]; then
  exit 0
fi
if [ "${SAST_MODE:-pack}" = "skip" ]; then
  exit 0
fi

CONFIG_ARGS=()
if [ "${SAST_MODE}" = "local" ]; then
  # Each path needs its own --config: extra positional paths become scan targets.
  for config_path in ${SAST_LOCAL_CONFIGS:-rules/sast}; do
    CONFIG_ARGS+=(--config "$config_path")
  done
else
  # Download and integrity check must fail the job: a tampered or empty pack would pass the gate.
  mkdir -p .spectra
  curl -sf -H "X-CI-Key: $API_KEY" -o /tmp/spectra-sast-rules.tar.gz \
    "${SPECTRA_URL%/}/api/v1/ci/sast-rules"
  # Use openssl: macOS sha256sum lacks GNU --check/--strict; Linux CI has both.
  if [ -n "${SAST_RULES_SHA256:-}" ]; then
    _got=$(openssl dgst -sha256 /tmp/spectra-sast-rules.tar.gz | awk '{print $NF}')
    if [ "$_got" != "${SAST_RULES_SHA256}" ]; then
      echo "spectra-sast-rules.tar.gz: SHA256 mismatch (expected ${SAST_RULES_SHA256}, got ${_got})" >&2
      exit 1
    fi
  fi
  tar -xzf /tmp/spectra-sast-rules.tar.gz -C .spectra
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
  echo "OpenGrep failed with exit code ${SAST_EXIT}"
  exit "$SAST_EXIT"
fi
