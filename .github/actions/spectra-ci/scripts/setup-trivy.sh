#!/usr/bin/env bash
# Thin wrapper: install the pinned Trivy binary via the generic provisioner
# (SPCTR-291). Pins come from tools/allowlist.json, never from callers; legacy
# env variables are accepted only when they match the allowlist pin.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"
export SPECTRA_TOOL_ASSET="${SPECTRA_TOOL_ASSET:-${TRIVY_ASSET_NAME:-}}"
# A provisioning failure (checksum, unreachable upstream) keeps failing the job
# in blocking mode; advisory mode warns and the scan step reports the missing
# tool (SPCTR-308). The provisioner itself stays fail-closed.
rc=0
bash "${SCRIPT_DIR}/setup-tool.sh" trivy || rc=$?
if [ "$rc" -ne 0 ]; then
  spectra_step_failed "trivy provisioning failed (exit ${rc})." "$rc" || exit $?
fi
