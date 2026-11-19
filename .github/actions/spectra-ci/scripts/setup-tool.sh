#!/usr/bin/env bash
# Generic pinned-tool installer for the Spectra CI action (SPCTR-291).
#
# Resolution order: runner cache (flock-guarded) → Forgejo generic-package
# mirror → upstream GitHub release (API with browser fallback). Every artifact
# is SHA-256 verified against tools/allowlist.json before install or publish;
# the scanned repository can disable a tool but never change its pin.
#
# Usage: setup-tool.sh <tool-name>
# Opt-in mirror environment:
#   SPECTRA_TOOL_MIRROR_URL    Forgejo base URL (empty = mirror disabled)
#   SPECTRA_TOOL_MIRROR_OWNER  Forgejo owner holding the packages
#   SPECTRA_TOOL_MIRROR_TOKEN  Forgejo token (required to publish)
#   SPECTRA_TOOLS_CACHE_DIR    Persistent runner cache (default /var/cache/spectra-tools)
#
# Extraction formats (SPCTR-289): raw binary (extract=null), tar.gz / tar.xz /
# tar.bz2 and zip archives, via the extract member path from the allowlist.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALLOWLIST="${SPECTRA_ALLOWLIST:-${SCRIPT_DIR}/../tools/allowlist.json}"

TOOLS_DIR="${RUNNER_TEMP:-/tmp}/spectra-tools"
WORK="${RUNNER_TEMP:-/tmp}/spectra-setup-tool"
MIRROR_URL="${SPECTRA_TOOL_MIRROR_URL:-}"
MIRROR_OWNER="${SPECTRA_TOOL_MIRROR_OWNER:-vela-labs}"
MIRROR_TOKEN="${SPECTRA_TOOL_MIRROR_TOKEN:-}"
LOCK_WAIT_SECONDS=600

require_runtime_deps() {
  local dependency
  for dependency in curl python3 tar; do
    command -v "$dependency" >/dev/null 2>&1 || {
      echo "::error::${dependency} is required for Spectra tool provisioning." >&2
      exit 1
    }
  done
}

# Detect the archive kind from the artifact name so the right extractor is
# used; a mismatch between name and content fails later in extract_artifact.
archive_kind() {
  case "$1" in
    *.tar.gz|*.tgz) echo 'tar.gz' ;;
    *.tar.xz) echo 'tar.xz' ;;
    *.tar.bz2) echo 'tar.bz2' ;;
    *.zip) echo 'zip' ;;
    *) echo '' ;;
  esac
}

sha256_of() {
  python3 - "$1" <<'PY'
import hashlib, sys

digest = hashlib.sha256()
with open(sys.argv[1], "rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
}

# AC1: a checksum mismatch must fail the job with an explicit message, before
# anything is installed or published.
assert_artifact_sha256() {
  local file=$1 actual
  actual="$(sha256_of "$file")"
  if [ "$actual" != "$EXPECTED_SHA256" ]; then
    echo "::error::SHA-256 mismatch for ${TOOL} ${TOOL_VERSION} (${ASSET_NAME}): expected ${EXPECTED_SHA256}, got ${actual}. Refusing to install or publish this artifact." >&2
    exit 1
  fi
  echo "sha256 ok (${ASSET_NAME})"
}

# Fail hard when a legacy env pin (OPENGREP_VERSION, TRIVY_SHA256…) diverges
# from the allowlist: two sources of truth must never disagree silently.
check_pin_env() {
  local variable_name=$1 expected=$2 value="${!1:-}"
  if [ -n "$value" ] && [ "$value" != "$expected" ]; then
    echo "::error::Spectra tool pin mismatch: ${variable_name}=${value} diverges from the allowlist (${expected})." >&2
    exit 1
  fi
}

pin_env_suffix() {
  printf '%s%s' "$(printf '%s' "$TOOL" | tr '[:lower:]-' '[:upper:]_')" "$1"
}

parse_allowlist_entry() {
  if [ ! -f "$ALLOWLIST" ]; then
    echo "::error::Spectra tool allowlist not found at ${ALLOWLIST}." >&2
    exit 1
  fi
  local parsed
  parsed="$(SPECTRA_TOOL_NAME="$TOOL" python3 - "$ALLOWLIST" <<'PY'
import json, os, shlex, sys

tool_name = os.environ["SPECTRA_TOOL_NAME"]
try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        allowlist = json.load(handle)
except (OSError, ValueError) as exc:
    sys.stderr.write(f"unreadable allowlist {sys.argv[1]!r}: {exc}\n")
    sys.exit(1)
tool = (allowlist.get("tools") or {}).get(tool_name)
if not tool:
    sys.stderr.write(f"tool {tool_name!r} is not in the allowlist\n")
    sys.exit(1)
upstream = tool.get("upstream") or {}
if upstream.get("kind", "github-release") != "github-release":
    sys.stderr.write(f"unsupported upstream kind for {tool_name!r}\n")
    sys.exit(1)
assets = tool.get("assets") or {}
asset_name = os.environ.get("SPECTRA_TOOL_ASSET") or (next(iter(assets)) if len(assets) == 1 else "")
asset = assets.get(asset_name) if asset_name else None
if asset is None:
    sys.stderr.write(f"cannot resolve the artifact for {tool_name!r}; set SPECTRA_TOOL_ASSET\n")
    sys.exit(1)
fields = {
    "TOOL_VERSION": tool["version"],
    "ASSET_NAME": asset_name,
    "EXPECTED_SHA256": asset["sha256"],
    "EXTRACT_PATH": asset.get("extract") or "",
    "BINARY_NAME": asset.get("binary") or tool_name,
    "UPSTREAM_REPO": upstream.get("repo") or "",
    # biome tags releases as "@biomejs/biome@x.y.z"; everything else uses vX.Y.Z.
    "RELEASE_TAG": (upstream.get("tag_template") or "v{version}").replace(
        "{version}", tool["version"]
    ),
}
for key, value in fields.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"
  eval "$parsed"
  check_pin_env "$(pin_env_suffix _VERSION)" "$TOOL_VERSION"
  check_pin_env "$(pin_env_suffix _SHA256)" "$EXPECTED_SHA256"
}

prepare_dirs() {
  mkdir -p "$TOOLS_DIR" "$WORK"
  CACHE_ROOT="${SPECTRA_TOOLS_CACHE_DIR:-/var/cache/spectra-tools}"
  if ! mkdir -p "$CACHE_ROOT" 2>/dev/null || [ ! -w "$CACHE_ROOT" ]; then
    CACHE_ROOT="${WORK}/cache"
    echo "Persistent tool cache unavailable; using job-local cache (${CACHE_ROOT})."
  fi
  CACHE_DIR="${CACHE_ROOT}/${TOOL}/${TOOL_VERSION}"
  CACHE_FILE="${CACHE_DIR}/${ASSET_NAME}"
  LOCK_FILE="${CACHE_ROOT}/${TOOL}.lock"
  mkdir -p "$CACHE_DIR"
}

# AC4: all cache writes happen under an exclusive flock so two concurrent jobs
# on the same runner can never interleave writes on the same artifact.
with_lock() {
  if ! command -v flock >/dev/null 2>&1; then
    "$@"
    return
  fi
  exec 9>>"$LOCK_FILE"
  if ! flock -w "$LOCK_WAIT_SECONDS" 9; then
    exec 9>&-
    echo "::warning::Timed out waiting for the ${TOOL} cache lock; installing without cross-job caching."
    "$@"
    return
  fi
  "$@"
  local status=$?
  flock -u 9
  exec 9>&-
  return "$status"
}

resolve_locked() {
  if use_cached_artifact; then
    return 0
  fi
  if fetch_from_mirror; then
    return 0
  fi
  fetch_from_upstream
}

# AC2: a valid cache entry short-circuits before any download attempt. A
# corrupted entry self-heals by re-downloading (everything is re-verified).
use_cached_artifact() {
  [ -s "$CACHE_FILE" ] || return 1
  local actual
  actual="$(sha256_of "$CACHE_FILE")"
  if [ "$actual" != "$EXPECTED_SHA256" ]; then
    echo "::warning::Cached ${TOOL} artifact failed its checksum; deleting it and downloading again." >&2
    rm -f "$CACHE_FILE"
    return 1
  fi
  ARTIFACT_SOURCE="cache"
  echo "Using the cached ${TOOL} ${TOOL_VERSION} artifact (0 bytes downloaded)."
}

# Atomic move inside the destination directory: readers never observe a torn
# artifact, even when the cache lives on a different filesystem than RUNNER_TEMP.
cache_artifact() {
  mkdir -p "$CACHE_DIR"
  local staged="${CACHE_FILE}.tmp.$$"
  if cp "$1" "$staged"; then
    mv -f "$staged" "$CACHE_FILE"
  else
    rm -f "$staged"
    return 1
  fi
}

mirror_artifact_url() {
  printf '%s/api/packages/%s/generic/spectra-%s/%s/%s' \
    "${MIRROR_URL%/}" "$MIRROR_OWNER" "$TOOL" "$TOOL_VERSION" "$ASSET_NAME"
}

_curl_github() {
  curl -fsSL --retry 8 --retry-delay 5 --retry-all-errors \
    ${GITHUB_TOKEN:+-H "Authorization: Bearer ${GITHUB_TOKEN}"} \
    "$@"
}

_curl_mirror() {
  curl -fsSL --retry 2 --retry-delay 2 \
    ${MIRROR_TOKEN:+-H "Authorization: token ${MIRROR_TOKEN}"} \
    "$@"
}

# AC3: the mirror sits before upstream, so an unreachable upstream never fails
# a job while the internal mirror is available.
fetch_from_mirror() {
  [ -n "$MIRROR_URL" ] || return 1
  local tmp="${WORK}/${ASSET_NAME}.mirror"
  if ! _curl_mirror -o "$tmp" "$(mirror_artifact_url)"; then
    rm -f "$tmp"
    return 1
  fi
  local actual
  actual="$(sha256_of "$tmp")"
  if [ "$actual" != "$EXPECTED_SHA256" ]; then
    echo "::warning::Mirror artifact for ${TOOL} ${TOOL_VERSION} failed its checksum; falling back to upstream." >&2
    rm -f "$tmp"
    return 1
  fi
  cache_artifact "$tmp"
  ARTIFACT_SOURCE="mirror"
  echo "Installed ${TOOL} ${TOOL_VERSION} from the internal mirror."
}

download_upstream_via_api() {
  local out=$1 meta asset_id
  meta="$(_curl_github \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/${UPSTREAM_REPO}/releases/tags/${RELEASE_TAG}")" || return 1
  asset_id="$(UPSTREAM_ASSET_NAME="$ASSET_NAME" python3 -c '
import json, os, sys

name = os.environ["UPSTREAM_ASSET_NAME"]
assets = json.load(sys.stdin).get("assets") or []
matches = [a.get("id") for a in assets if a.get("name") == name]
if not matches:
    sys.stderr.write(f"asset {name!r} not found in release payload\n")
    sys.exit(1)
print(matches[0])
' <<<"$meta")" || return 1
  echo "Downloading ${ASSET_NAME} via API asset ${asset_id}"
  _curl_github \
    -H "Accept: application/octet-stream" \
    -L \
    -o "$out" \
    "https://api.github.com/repos/${UPSTREAM_REPO}/releases/assets/${asset_id}"
}

download_upstream_via_browser() {
  echo "API download failed; falling back to the browser download URL." >&2
  _curl_github -L -o "$1" \
    "https://github.com/${UPSTREAM_REPO}/releases/download/${RELEASE_TAG}/${ASSET_NAME}"
}

fetch_from_upstream() {
  local tmp="${WORK}/${ASSET_NAME}.upstream"
  if ! download_upstream_via_api "$tmp" && ! download_upstream_via_browser "$tmp"; then
    echo "::error::Upstream ${UPSTREAM_REPO} is unreachable and no mirror served ${TOOL} ${TOOL_VERSION}." >&2
    return 1
  fi
  assert_artifact_sha256 "$tmp"
  cache_artifact "$tmp"
  ARTIFACT_SOURCE="upstream"
  echo "Downloaded ${TOOL} ${TOOL_VERSION} from upstream."
}

# Best-effort self-publication: after a verified upstream download, feed the
# mirror so the next run never touches upstream again. Never fails the job.
publish_to_mirror() {
  if [ -z "$MIRROR_URL" ] || [ -z "$MIRROR_TOKEN" ] || [ "$ARTIFACT_SOURCE" != "upstream" ]; then
    return 0
  fi
  local http_code
  http_code="$(curl -sS -o /dev/null -w "%{http_code}" -X PUT \
    -H "Authorization: token ${MIRROR_TOKEN}" \
    --upload-file "$CACHE_FILE" \
    "$(mirror_artifact_url)" || echo 000)"
  case "$http_code" in
    200 | 201 | 204 | 409) echo "Published ${TOOL} ${TOOL_VERSION} to the internal mirror (HTTP ${http_code})." ;;
    *) echo "::warning::Mirror publish for ${TOOL} ${TOOL_VERSION} failed (HTTP ${http_code}); the next run will retry." >&2 ;;
  esac
}

extract_artifact() {
  local target="${TOOLS_DIR}/${BINARY_NAME}"
  local kind
  kind="$(archive_kind "$ASSET_NAME")"
  if [ -z "$EXTRACT_PATH" ] && [ -z "$kind" ]; then
    cp "$CACHE_FILE" "$target"
  elif [ -z "$EXTRACT_PATH" ]; then
    echo "::error::${TOOL}: archive ${ASSET_NAME} requires an extract member in the allowlist." >&2
    return 1
  elif [ "$kind" = "zip" ]; then
    command -v unzip >/dev/null 2>&1 || {
      echo "::error::unzip is required to extract ${ASSET_NAME}." >&2
      return 1
    }
    unzip -o -q "$CACHE_FILE" -d "$WORK"
    mv "$WORK/${EXTRACT_PATH}" "$target"
  elif [ -n "$kind" ]; then
    # tar auto-detects gzip/xz/bzip2 on GNU tar and bsdtar.
    tar -xf "$CACHE_FILE" -C "$WORK"
    mv "$WORK/${EXTRACT_PATH}" "$target"
  else
    echo "::error::${TOOL}: unexpected archive ${ASSET_NAME} with an extract member." >&2
    return 1
  fi
  chmod +x "$target"
  if [ -n "${GITHUB_PATH:-}" ]; then
    echo "$TOOLS_DIR" >> "$GITHUB_PATH"
  fi
}

# Linux CI runners execute the binary; other hosts may only verify the download.
verify_binary() {
  if "${TOOLS_DIR}/${BINARY_NAME}" --version; then
    return 0
  fi
  echo "${BINARY_NAME} installed at ${TOOLS_DIR} (not executable on this host)" >&2
}

main() {
  TOOL="${1:?usage: setup-tool.sh <tool-name>}"
  require_runtime_deps
  parse_allowlist_entry
  prepare_dirs
  with_lock resolve_locked
  # Double verification after the lock: guards a cache written by a dead job.
  assert_artifact_sha256 "$CACHE_FILE"
  extract_artifact
  verify_binary
  publish_to_mirror
  echo "Spectra tool ${TOOL} ${TOOL_VERSION} ready (source: ${ARTIFACT_SOURCE})."
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
