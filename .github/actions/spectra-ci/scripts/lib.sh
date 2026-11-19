#!/usr/bin/env bash
# Shared helpers for Spectra CI composite scripts.

spectra_is_enabled() {
  case "${1:-false}" in
    true | True | TRUE | 1 | yes | Yes | YES) return 0 ;;
    *) return 1 ;;
  esac
}

# Gate mode comes from /scans registration (register.sh output gate_mode).
# Unset means advisory, like register.sh and complete-wait.sh.
spectra_gate_is_blocking() {
  [ "${GATE_MODE:-advisory}" = "blocking" ]
}

# Report a failed step (SPCTR-308). Blocking mode keeps the fail-closed
# behaviour (::error:: and the given exit code, default 1); advisory mode
# never blocks a merge: it warns and returns 0 so the caller can go on.
spectra_step_failed() {
  local message=$1
  local code=${2:-1}
  if spectra_gate_is_blocking; then
    echo "::error::${message}"
    return "$code"
  fi
  echo "::warning::${message} Advisory mode does not block the merge; results from this step are missing from the Spectra analysis."
  return 0
}

spectra_normalize_bool() {
  if spectra_is_enabled "${1:-false}"; then
    echo "true"
  else
    echo "false"
  fi
}

# POST a JSON file to /api/v1/ci/scans/$SCAN_ID/results/<endpoint>.
# Needs SPECTRA_URL, API_KEY and SCAN_ID. Missing file: no-op success.
spectra_upload_result() {
  local file=$1
  local endpoint=$2
  if [ ! -f "$file" ]; then
    return 0
  fi
  local body_file http_code
  body_file="$(mktemp)"
  http_code=$(curl -sS -o "$body_file" -w "%{http_code}" \
    -X POST "${SPECTRA_URL%/}/api/v1/ci/scans/${SCAN_ID}/results/${endpoint}" \
    -H "X-CI-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d @"$file" || echo "000")
  if [ "$http_code" -ge 200 ] && [ "$http_code" -lt 300 ]; then
    echo "Uploaded ${endpoint} from ${file} (HTTP ${http_code})."
    rm -f "$body_file"
    return 0
  fi
  echo "::error::Upload ${endpoint} failed (HTTP ${http_code}): $(head -c 400 "$body_file" 2>/dev/null || true)"
  rm -f "$body_file"
  return 1
}

spectra_trivy() {
  if command -v trivy >/dev/null 2>&1; then
    command trivy "$@"
    return $?
  fi
  local fallback="${RUNNER_TEMP:-/tmp}/spectra-tools/trivy"
  if [ -x "$fallback" ]; then
    "$fallback" "$@"
    return $?
  fi
  echo "Trivy binary not found (PATH or ${fallback})." >&2
  return 127
}

# Workspace folders that are never repository content: caches, virtualenvs and build
# outputs left by earlier pipeline steps (pytest, npm, builds) (SPCTR-268).
SPECTRA_DEFAULT_SKIP_DIRS=(
  .git .pytest_cache .mypy_cache .ruff_cache __pycache__ .tox .venv venv node_modules .spectra
)

spectra_skip_dir_args() {
  local dir
  for dir in "${SPECTRA_DEFAULT_SKIP_DIRS[@]}"; do
    printf -- '--skip-dirs\n**/%s\n' "$dir"
  done
}

_spectra_path_like_ignore_line() {
  case "$1" in
    \#*|"") return 1 ;;
  esac
  case "$1" in
    */*|*'*'*) return 0 ;;
  esac
  return 1
}

_spectra_repo_ignore_path_lines() {
  local root="${1:-.}"
  local file line
  for file in .trivyignore .gitleaksignore; do
    [ -f "${root}/${file}" ] || continue
    while IFS= read -r line || [ -n "$line" ]; do
      line="$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
      if _spectra_path_like_ignore_line "$line"; then
        printf '%s\n' "$line"
      fi
    done < "${root}/${file}"
  done
}

_spectra_trivy_skip_from_glob() {
  local pattern="$1"
  local inner
  if [[ "$pattern" == */** ]]; then
    inner="${pattern%/**}"
    inner="${inner#/}"
    inner="${inner#**/}"
    printf -- '--skip-dirs\n**/%s/**\n' "$inner"
    return
  fi
  if [[ "$pattern" == **/* ]]; then
    printf -- '--skip-files\n%s\n' "$pattern"
    return
  fi
  if [[ "$pattern" == */* ]]; then
    printf -- '--skip-files\n**/%s\n' "${pattern#/}"
    return
  fi
  printf -- '--skip-files\n**/%s\n' "$pattern"
}

spectra_ci_secret_skip_args() {
  if [ -n "${SECRETS_SCAN_JSON:-}" ] && command -v jq >/dev/null 2>&1; then
    local pat
    while IFS= read -r pat; do
      [ -n "$pat" ] && printf -- '--skip-dirs\n%s\n' "$pat"
    done < <(echo "$SECRETS_SCAN_JSON" | jq -r '.skip_dirs[]? // empty' 2>/dev/null || true)
    while IFS= read -r pat; do
      [ -n "$pat" ] && printf -- '--skip-files\n%s\n' "$pat"
    done < <(echo "$SECRETS_SCAN_JSON" | jq -r '.skip_files[]? // empty' 2>/dev/null || true)
  fi
  local root="${GITHUB_WORKSPACE:-.}"
  while IFS= read -r pat; do
    [ -n "$pat" ] && _spectra_trivy_skip_from_glob "$pat"
  done < <(_spectra_repo_ignore_path_lines "$root")
}

spectra_all_skip_args() {
  spectra_skip_dir_args
  spectra_ci_secret_skip_args
}

# Print a temporary directory holding only the files tracked by git.
# A secret scan must read the repository, not what earlier steps wrote in the
# runner workspace. checkout-index (not git archive) so `export-ignore` in
# .gitattributes cannot hide files from the scan. Returns 1 outside a git checkout.
spectra_export_tracked_tree() {
  local target
  git rev-parse --verify HEAD >/dev/null 2>&1 || return 1
  target="$(mktemp -d "${RUNNER_TEMP:-/tmp}/spectra-tracked.XXXXXX")"
  if ! git checkout-index --all --force --prefix="${target}/"; then
    rm -rf "$target"
    return 1
  fi
  echo "$target"
}
