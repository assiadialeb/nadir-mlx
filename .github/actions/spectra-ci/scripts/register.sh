#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

: "${SPECTRA_URL:?}"
: "${API_KEY:?}"
: "${COMMIT_SHA:?}"
: "${BRANCH:?}"

BODY=$(jq -n \
  --arg sha "$COMMIT_SHA" \
  --arg branch "$BRANCH" \
  --arg pr "${PULL_REQUEST:-}" \
  --arg trigger "${TRIGGERED_BY:-push}" \
  '{commit_sha: $sha, branch: $branch, pull_request: $pr, triggered_by: $trigger}')

RESPONSE=$(curl -sf -X POST "${SPECTRA_URL%/}/api/v1/ci/scans" \
  -H "X-CI-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d "$BODY")

SCAN_ID=$(echo "$RESPONSE" | jq -r '.data.scan_id')
STATUS=$(echo "$RESPONSE" | jq -r '.data.status')
QUALITY=$(echo "$RESPONSE" | jq -r '.data.quality_provider // "sonar"')
SONAR_TOKEN_MANAGED=$(echo "$RESPONSE" | jq -r '.data.sonar.token // empty')
if [ -n "$SONAR_TOKEN_MANAGED" ]; then
  echo "::add-mask::$SONAR_TOKEN_MANAGED"
fi

_scanner_flag() {
  spectra_normalize_bool "$(echo "$RESPONSE" | jq -r "$1 // false")"
}

{
  echo "scan_id=$SCAN_ID"
  echo "status=$STATUS"
  echo "quality_provider=$QUALITY"
  echo "sonar_url=$(echo "$RESPONSE" | jq -r '.data.sonar.host_url // empty')"
  echo "sonar_token=$SONAR_TOKEN_MANAGED"
  echo "sonar_key=$(echo "$RESPONSE" | jq -r '.data.sonar.project_key // empty')"
  echo "sonar_organization=$(echo "$RESPONSE" | jq -r '.data.sonar.organization // empty')"
  echo "sast=$(_scanner_flag '.data.scanners.sast_enabled')"
  echo "sca=$(_scanner_flag '.data.scanners.sca_enabled')"
  echo "iac=$(_scanner_flag '.data.scanners.iac_enabled')"
  echo "secrets=$(_scanner_flag '.data.scanners.secrets_enabled')"
  echo "sbom=$(_scanner_flag '.data.scanners.sbom_enabled')"
  echo "sast_rules_sha256=$(echo "$RESPONSE" | jq -r '.data.sast_rules.sha256 // empty')"
} >> "$GITHUB_OUTPUT"
