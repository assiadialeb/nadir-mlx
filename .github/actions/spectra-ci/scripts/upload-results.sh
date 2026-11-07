#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

: "${SPECTRA_URL:?}"
: "${API_KEY:?}"
: "${SCAN_ID:?}"

upload_if_exists() {
  local file=$1
  local endpoint=$2
  if [ ! -f "$file" ]; then
    return 0
  fi
  local http_code
  http_code=$(curl -sS -o /tmp/spectra-upload-body.txt -w "%{http_code}" \
    -X POST "${SPECTRA_URL%/}/api/v1/ci/scans/${SCAN_ID}/results/${endpoint}" \
    -H "X-CI-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d @"$file" || echo "000")
  if [ "$http_code" -ge 200 ] && [ "$http_code" -lt 300 ]; then
    echo "Uploaded ${endpoint} from ${file} (HTTP ${http_code})."
    return 0
  fi
  echo "::error::Upload ${endpoint} failed (HTTP ${http_code}): $(head -c 400 /tmp/spectra-upload-body.txt 2>/dev/null || true)"
  return 1
}

FAILED=0

upload_if_exists sast_output.json sast || FAILED=1
upload_if_exists trivy_sca.json sca || FAILED=1
upload_if_exists trivy.json sca || FAILED=1
upload_if_exists trivy_iac.json iac || FAILED=1
upload_if_exists iac_output.json iac || FAILED=1
upload_if_exists trivy_secrets.json secrets || FAILED=1
upload_if_exists secrets_output.json secrets || FAILED=1
upload_if_exists sbom.cdx.json sbom || FAILED=1

if [ "$FAILED" -ne 0 ]; then
  echo "::error::One or more Spectra result uploads failed; aborting before /complete."
  exit 1
fi

if spectra_is_enabled "${SBOM_ENABLED:-false}" && [ ! -s sbom.cdx.json ]; then
  echo "::warning::SBOM is enabled for this repository but sbom.cdx.json is missing or empty after Trivy."
fi
