#!/usr/bin/env bash
set -euo pipefail

TRIVY_VERSION="${TRIVY_VERSION:-0.76.2}"
TRIVY_SHA256="${TRIVY_SHA256:-bc0fc726e017b8927a62b2d3531d7a84ee01769daf5c99ed866dced54089fdbf}"

TOOLS_DIR="$RUNNER_TEMP/spectra-tools"
mkdir -p "$TOOLS_DIR"
ARCHIVE="$TOOLS_DIR/trivy_${TRIVY_VERSION}_Linux-64bit.tar.gz"
curl -fsSL --retry 5 --retry-delay 2 --retry-all-errors -o "$ARCHIVE" \
  "https://github.com/aquasecurity/trivy/releases/download/v${TRIVY_VERSION}/trivy_${TRIVY_VERSION}_Linux-64bit.tar.gz"
echo "${TRIVY_SHA256}  ${ARCHIVE}" | sha256sum --check --strict
tar -xzf "$ARCHIVE" -C "$TOOLS_DIR" trivy
chmod +x "$TOOLS_DIR/trivy"
echo "$TOOLS_DIR" >> "$GITHUB_PATH"
"$TOOLS_DIR/trivy" --version
