#!/usr/bin/env bash
# Prefer GitHub Releases API over /releases/download (regional edge 504s).
set -euo pipefail

TRIVY_VERSION="${TRIVY_VERSION:-0.76.2}"
TRIVY_SHA256="${TRIVY_SHA256:-bc0fc726e017b8927a62b2d3531d7a84ee01769daf5c99ed866dced54089fdbf}"
TRIVY_ASSET_NAME="${TRIVY_ASSET_NAME:-trivy_${TRIVY_VERSION}_Linux-64bit.tar.gz}"
TRIVY_REPO="${TRIVY_REPO:-aquasecurity/trivy}"

TOOLS_DIR="${RUNNER_TEMP:-/tmp}/spectra-tools"
ARCHIVE="$TOOLS_DIR/${TRIVY_ASSET_NAME}"
mkdir -p "$TOOLS_DIR"

_curl() {
  curl -fsSL --retry 8 --retry-delay 5 --retry-all-errors \
    ${GITHUB_TOKEN:+-H "Authorization: Bearer ${GITHUB_TOKEN}"} \
    "$@"
}

download_via_api() {
  local meta asset_id
  meta="$(_curl \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/${TRIVY_REPO}/releases/tags/v${TRIVY_VERSION}")"
  asset_id="$(
    TRIVY_ASSET_NAME="$TRIVY_ASSET_NAME" python3 -c '
import json, os, sys
name = os.environ["TRIVY_ASSET_NAME"]
assets = json.load(sys.stdin).get("assets") or []
matches = [a.get("id") for a in assets if a.get("name") == name]
if not matches:
    sys.stderr.write(f"asset {name!r} not found in release payload\n")
    sys.exit(1)
print(matches[0])
' <<<"$meta"
  )"
  echo "Downloading Trivy v${TRIVY_VERSION} via API asset ${asset_id}"
  _curl \
    -H "Accept: application/octet-stream" \
    -L \
    -o "$ARCHIVE" \
    "https://api.github.com/repos/${TRIVY_REPO}/releases/assets/${asset_id}"
}

download_via_browser_url() {
  echo "Downloading Trivy v${TRIVY_VERSION} via releases/download URL"
  _curl -L -o "$ARCHIVE" \
    "https://github.com/${TRIVY_REPO}/releases/download/v${TRIVY_VERSION}/${TRIVY_ASSET_NAME}"
}

if ! download_via_api; then
  echo "API asset download failed; falling back to browser download URL" >&2
  download_via_browser_url
fi

ARCHIVE="$ARCHIVE" EXPECT="$TRIVY_SHA256" python3 - <<'PY'
import hashlib, os, pathlib, sys
path = pathlib.Path(os.environ["ARCHIVE"])
digest = hashlib.sha256(path.read_bytes()).hexdigest()
expected = os.environ["EXPECT"]
if digest != expected:
    sys.stderr.write(f"sha256 mismatch: {digest} != {expected}\n")
    sys.exit(1)
print(f"sha256 ok ({digest})")
PY

tar -xzf "$ARCHIVE" -C "$TOOLS_DIR" trivy
chmod +x "$TOOLS_DIR/trivy"
if [[ -n "${GITHUB_PATH:-}" ]]; then
  echo "$TOOLS_DIR" >> "$GITHUB_PATH"
fi
if "$TOOLS_DIR/trivy" --version; then
  :
else
  echo "Trivy binary installed at ${TOOLS_DIR}/trivy (not executable on this host)" >&2
fi
