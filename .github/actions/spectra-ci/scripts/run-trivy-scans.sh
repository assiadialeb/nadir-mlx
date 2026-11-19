#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

if [ "${SCAN_STATUS:-}" = "bypassed" ]; then
  exit 0
fi

SKIP_ARGS=()
while IFS= read -r arg; do SKIP_ARGS+=("$arg"); done < <(spectra_all_skip_args)

# Run one Trivy command writing <output>. On failure, blocking mode stops the
# job with the real exit code in the message; advisory mode (SPCTR-308) drops
# the partial report so it is never uploaded, warns, and lets the other
# scanners run.
run_trivy_or_fail() {
  local label="$1"
  local output="$2"
  shift 2
  local rc=0
  spectra_trivy "$@" || rc=$?
  if [ "$rc" -ne 0 ]; then
    rm -f "$output"
    spectra_step_failed "Trivy ${label} failed (exit ${rc})." || exit 1
  fi
}

TRACKED_TREE=""
cleanup_tracked_tree() {
  if [ -n "$TRACKED_TREE" ]; then rm -rf "$TRACKED_TREE"; fi
}
trap cleanup_tracked_tree EXIT

# Secrets: scan only tracked files, never the runner workspace.
# Falls back to the workspace with cache folders skipped outside a git checkout.
run_secret_scan() {
  if TRACKED_TREE="$(spectra_export_tracked_tree)"; then
    run_trivy_or_fail "secrets scan" "$PWD/trivy_secrets.json" fs --format json --scanners secret \
      "${SKIP_ARGS[@]}" --output "$PWD/trivy_secrets.json" "$TRACKED_TREE"
    return
  fi
  echo "::warning::Not a git checkout: secrets are scanned in the workspace (cache folders skipped)."
  run_trivy_or_fail "secrets scan" trivy_secrets.json fs --format json --scanners secret \
    "${SKIP_ARGS[@]}" --output trivy_secrets.json .
}

if spectra_is_enabled "${SCA_ENABLED:-false}"; then
  # No skip-dirs for SCA: without a lockfile, installed packages (node_modules, .venv) are the inventory.
  run_trivy_or_fail "SCA scan" trivy_sca.json fs --format json --scanners vuln --output trivy_sca.json .
fi
if spectra_is_enabled "${IAC_ENABLED:-false}"; then
  run_trivy_or_fail "IaC scan" trivy_iac.json config --format json "${SKIP_ARGS[@]}" --output trivy_iac.json .
fi
if spectra_is_enabled "${SECRETS_ENABLED:-false}"; then
  run_secret_scan
fi
if spectra_is_enabled "${SBOM_ENABLED:-false}"; then
  if ! spectra_trivy fs --format cyclonedx --output sbom.cdx.json .; then
    echo "::warning::Trivy SBOM generation failed."
  elif [ ! -s sbom.cdx.json ]; then
    echo "::warning::Trivy produced an empty sbom.cdx.json."
  fi
fi
