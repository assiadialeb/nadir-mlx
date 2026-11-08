#!/usr/bin/env bash
# Download the pinned OpenGrep binary.
# Prefer the GitHub Releases *API* asset endpoint: the browser
# `/releases/download/...` URL has been returning regional edge 504s while
# api.github.com → release-assets.githubusercontent.com still works.
set -euo pipefail

OPENGREP_VERSION="${OPENGREP_VERSION:-1.33.1}"
OPENGREP_SHA256="${OPENGREP_SHA256:-46e8133cc490f9f50518061590ce2f6be5a08afaeda034471c0ceda4b3899808}"
OPENGREP_ASSET_NAME="${OPENGREP_ASSET_NAME:-opengrep_manylinux_x86}"
OPENGREP_REPO="${OPENGREP_REPO:-opengrep/opengrep}"

TOOLS_DIR="${RUNNER_TEMP:-/tmp}/spectra-tools"
OUT="$TOOLS_DIR/opengrep"
mkdir -p "$TOOLS_DIR"

_curl() {
  # Longer backoff than the previous 5×2s: edge 504s often last tens of seconds.
  # shellcheck disable=SC2086
  curl -fsSL --retry 8 --retry-delay 5 --retry-all-errors \
    ${GITHUB_TOKEN:+-H "Authorization: Bearer ${GITHUB_TOKEN}"} \
    "$@"
}

download_via_api() {
  local meta asset_id
  meta="$(_curl \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/${OPENGREP_REPO}/releases/tags/v${OPENGREP_VERSION}")"
  asset_id="$(
    OPENGREP_ASSET_NAME="$OPENGREP_ASSET_NAME" python3 -c '
import json, os, sys
name = os.environ["OPENGREP_ASSET_NAME"]
assets = json.load(sys.stdin).get("assets") or []
matches = [a.get("id") for a in assets if a.get("name") == name]
if not matches:
    sys.stderr.write(f"asset {name!r} not found in release payload\n")
    sys.exit(1)
print(matches[0])
' <<<"$meta"
  )"
  echo "Downloading OpenGrep v${OPENGREP_VERSION} via API asset ${asset_id}"
  _curl \
    -H "Accept: application/octet-stream" \
    -L \
    -o "$OUT" \
    "https://api.github.com/repos/${OPENGREP_REPO}/releases/assets/${asset_id}"
}

download_via_browser_url() {
  echo "Downloading OpenGrep v${OPENGREP_VERSION} via releases/download URL"
  _curl -L -o "$OUT" \
    "https://github.com/${OPENGREP_REPO}/releases/download/v${OPENGREP_VERSION}/${OPENGREP_ASSET_NAME}"
}

if ! download_via_api; then
  echo "API asset download failed; falling back to browser download URL" >&2
  download_via_browser_url
fi

OPENGREP_OUT="$OUT" OPENGREP_EXPECT="$OPENGREP_SHA256" python3 - <<'PY'
import hashlib
import os
import pathlib
import sys

path = pathlib.Path(os.environ["OPENGREP_OUT"])
digest = hashlib.sha256(path.read_bytes()).hexdigest()
expected = os.environ["OPENGREP_EXPECT"]
if digest != expected:
    sys.stderr.write(f"sha256 mismatch: {digest} != {expected}\n")
    sys.exit(1)
print(f"sha256 ok ({digest})")
PY
chmod +x "$OUT"
if [[ -n "${GITHUB_PATH:-}" ]]; then
  echo "$TOOLS_DIR" >> "$GITHUB_PATH"
fi
# Linux CI runners execute the binary; other hosts may only verify the download.
if "$OUT" --version; then
  :
else
  echo "OpenGrep binary installed at ${OUT} (not executable on this host)" >&2
fi
