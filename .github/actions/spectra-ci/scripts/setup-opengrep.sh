#!/usr/bin/env bash
set -euo pipefail

OPENGREP_VERSION="${OPENGREP_VERSION:-1.33.1}"
OPENGREP_SHA256="${OPENGREP_SHA256:-46e8133cc490f9f50518061590ce2f6be5a08afaeda034471c0ceda4b3899808}"

TOOLS_DIR="$RUNNER_TEMP/spectra-tools"
mkdir -p "$TOOLS_DIR"
curl -fsSL --retry 5 --retry-delay 2 --retry-all-errors -o "$TOOLS_DIR/opengrep" \
  "https://github.com/opengrep/opengrep/releases/download/v${OPENGREP_VERSION}/opengrep_manylinux_x86"
echo "${OPENGREP_SHA256}  $TOOLS_DIR/opengrep" | sha256sum --check --strict
chmod +x "$TOOLS_DIR/opengrep"
echo "$TOOLS_DIR" >> "$GITHUB_PATH"
"$TOOLS_DIR/opengrep" --version
