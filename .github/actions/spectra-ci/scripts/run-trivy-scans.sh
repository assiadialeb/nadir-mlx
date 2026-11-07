#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

if [ "${SCAN_STATUS:-}" = "bypassed" ]; then
  exit 0
fi

if spectra_is_enabled "${SCA_ENABLED:-false}"; then
  if ! spectra_trivy fs --format json --scanners vuln --output trivy_sca.json .; then
    rc=$?
    echo "::error::Trivy SCA scan failed (exit ${rc})."
    exit 1
  fi
fi
if spectra_is_enabled "${IAC_ENABLED:-false}"; then
  if ! spectra_trivy config --format json --output trivy_iac.json .; then
    rc=$?
    echo "::error::Trivy IaC scan failed (exit ${rc})."
    exit 1
  fi
fi
if spectra_is_enabled "${SECRETS_ENABLED:-false}"; then
  if ! spectra_trivy fs --format json --scanners secret --output trivy_secrets.json .; then
    rc=$?
    echo "::error::Trivy secrets scan failed (exit ${rc})."
    exit 1
  fi
fi
if spectra_is_enabled "${SBOM_ENABLED:-false}"; then
  if ! spectra_trivy fs --format cyclonedx --output sbom.cdx.json .; then
    echo "::warning::Trivy SBOM generation failed."
  elif [ ! -s sbom.cdx.json ]; then
    echo "::warning::Trivy produced an empty sbom.cdx.json."
  fi
fi
