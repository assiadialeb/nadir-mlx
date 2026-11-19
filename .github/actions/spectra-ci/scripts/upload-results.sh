#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

: "${SPECTRA_URL:?}"
: "${API_KEY:?}"
: "${SCAN_ID:?}"

FAILED=0

spectra_upload_result sast_output.json sast || FAILED=1
spectra_upload_result trivy_sca.json sca || FAILED=1
spectra_upload_result trivy.json sca || FAILED=1
spectra_upload_result trivy_iac.json iac || FAILED=1
spectra_upload_result iac_output.json iac || FAILED=1
spectra_upload_result trivy_secrets.json secrets || FAILED=1
spectra_upload_result secrets_output.json secrets || FAILED=1
spectra_upload_result sbom.cdx.json sbom || FAILED=1

# Blocking mode aborts before /complete; advisory mode never blocks a merge
# (SPCTR-308): it warns and lets complete/wait report the verdict.
if [ "$FAILED" -ne 0 ]; then
  spectra_step_failed "One or more Spectra result uploads failed." || {
    echo "Aborting before /complete."
    exit 1
  }
fi

if spectra_is_enabled "${SBOM_ENABLED:-false}" && [ ! -s sbom.cdx.json ]; then
  echo "::warning::SBOM is enabled for this repository but sbom.cdx.json is missing or empty after Trivy."
fi
