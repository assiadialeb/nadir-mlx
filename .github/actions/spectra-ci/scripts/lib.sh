#!/usr/bin/env bash
# Shared helpers for Spectra CI composite scripts.

spectra_is_enabled() {
  case "${1:-false}" in
    true | True | TRUE | 1 | yes | Yes | YES) return 0 ;;
    *) return 1 ;;
  esac
}

spectra_normalize_bool() {
  if spectra_is_enabled "${1:-false}"; then
    echo "true"
  else
    echo "false"
  fi
}

spectra_trivy() {
  if command -v trivy >/dev/null 2>&1; then
    command trivy "$@"
    return $?
  fi
  local fallback="${RUNNER_TEMP:-/tmp}/spectra-tools/trivy"
  if [ -x "$fallback" ]; then
    "$fallback" "$@"
    return $?
  fi
  echo "Trivy binary not found (PATH or ${fallback})." >&2
  return 127
}
