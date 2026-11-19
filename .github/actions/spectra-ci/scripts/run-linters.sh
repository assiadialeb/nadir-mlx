#!/usr/bin/env bash
# Spectra linters for quality_provider=linters (SPCTR-287), language phase (SPCTR-289),
# fail-closed execution manifest (SPCTR-301).
#
# Runs the pinned core linters matching the repository's tracked files, then
# uploads SARIF, metrics and the optional Cobertura report. Repository builds
# that ship their own SARIF (PMD, detekt, clippy…) are ingested through the
# repo-sarif input without installing any toolchain here.
# Trust model: tools come from the pinned allowlist (SPCTR-291) and hash-checked
# wheels in an isolated venv; the scanned repository's dependencies and tool
# configs are never used: every tool gets an explicit Spectra configuration
# from linters/ and the flag that stops it from reading the repository's own
# (SPCTR-300, table in docs/dev/LINTERS.md). SAST, secrets and taint analysis stay on OpenGrep and
# Trivy; linter taint engines are never enabled.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

: "${SPECTRA_URL:?}"
: "${API_KEY:?}"
: "${SCAN_ID:?}"

if [ "${SCAN_STATUS:-}" = "bypassed" ]; then
  exit 0
fi

ACTION_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK="${RUNNER_TEMP:-/tmp}/spectra-linters"
JSCPD_VERSION="${JSCPD_VERSION:-4.0.5}"
# Vendored cognitive_complexity 1.3.0, aligned with SonarQube (SPCTR-305).
COGNITIVE_VERSION="1.3.0+spectra.1"
SQLFLUFF_DIALECT="${SQLFLUFF_DIALECT:-ansi}"
SQLFLUFF_EXCLUDED_RULES="${SQLFLUFF_EXCLUDED_RULES:-LT01,LT02,LT03,LT04,LT05,LT06,LT07,LT08,LT09,LT10,LT11,LT12,CP01}"
BASE_PATH="$(pwd -P)"
LINTERS_CONFIG_DIR="${ACTION_DIR}/linters"
FAILED=0
mkdir -p "$WORK"
MANIFEST="$WORK/manifest.jsonl"
: > "$MANIFEST"

payload() {
  python3 "${SCRIPT_DIR}/linters_payload.py" "$@"
}

convert() {
  python3 "${SCRIPT_DIR}/lint_convert.py" "$@"
}

catalog() {
  python3 "${SCRIPT_DIR}/linters_catalog.py" "${ACTION_DIR}/linters/catalog.json" "$@"
}

# Same resolution order as setup-tool.sh: PATH first, then the runner tools dir.
tool_path() {
  local name=$1 fallback="${RUNNER_TEMP:-/tmp}/spectra-tools/${1}"
  if command -v "$name" >/dev/null 2>&1; then
    command -v "$name"
  elif [ -x "$fallback" ]; then
    echo "$fallback"
  fi
}

# First version number printed by --version (shellcheck prints it on line 2).
tool_version() {
  local version
  version="$("$1" --version 2>/dev/null | grep -oE '[0-9]+([.][0-9]+)+' | head -1)" || true
  echo "${version:-unknown}"
}

# Tracked files, one real path per line: without core.quotePath=false git
# quotes non-ASCII names ("app/caf\303\251.py") and tools get missing files.
git_ls_files() {
  git -c core.quotePath=false ls-files "$@"
}

# Default exclusions (catalog default_excludes, SPCTR-301): vendored, built
# and minified files are never linted, as git pathspec excludes.
DEFAULT_EXCLUDES=()
load_default_excludes() {
  local pattern
  while IFS= read -r pattern; do
    [ -n "$pattern" ] && DEFAULT_EXCLUDES+=(":(exclude,glob)${pattern}")
  done < <(catalog excludes)
}

# Tracked files matching pathspecs, minus the default exclusions.
linted_files() {
  git_ls_files -- "$@" ${DEFAULT_EXCLUDES[@]+"${DEFAULT_EXCLUDES[@]}"}
}

# Tracked files matching a tool's catalog pathspecs.
tool_files() {
  local pathspec
  while IFS= read -r pathspec; do
    [ -n "$pathspec" ] || continue
    linted_files "$pathspec" || true
  done < <(catalog paths "$1")
}

# TOOL_FILES array of a tool's files: a path with a space stays one argument
# (an unquoted $(tool_files …) split "scripts/deploy tools/x.sh" in two and
# made the tool fail on files that do not exist).
load_tool_files() {
  local file
  TOOL_FILES=()
  while IFS= read -r file; do
    [ -n "$file" ] && TOOL_FILES+=("$file")
  done < <(tool_files "$1")
}

upload_sarif() {
  local file=$1 tool=$2
  [ -s "$file" ] || return 0
  payload payload sarif "$file" --tool "$tool" --base-path "$BASE_PATH" \
    ${CHANGED_ARGS[@]+"${CHANGED_ARGS[@]}"} > "${WORK}/${tool}-payload.json"
  spectra_upload_result "${WORK}/${tool}-payload.json" sarif || FAILED=1
}

# ── Execution manifest (SPCTR-301) ──────────────────────────────────────────
# Every applicable tool records ok / failed / unavailable with its version,
# duration and result count; Spectra fails the gate (analysis_incomplete)
# when a tool did not run, instead of reading its silence as "no finding".

record_tool() {  # tool status version results reason started_at
  payload manifest-entry --tool "$1" --status "$2" --version "$3" --results "$4" \
    --reason "$5" --seconds "$((SECONDS - $6))" >> "$MANIFEST"
}

stderr_tail() {
  tail -c 400 "$1" 2>/dev/null | tr '\n\r' '  ' || true
}

exit_code_allowed() {  # code, comma-separated allowed codes
  case ",$2," in *",$1,"*) return 0 ;; esac
  return 1
}

# Record and upload a SARIF tool once it ran: a disallowed exit code, a
# failed conversion or an unreadable SARIF log make it "failed".
finish_sarif_tool() {  # tool version exit_code allowed_codes sarif stderr_file started_at
  local tool=$1 version=$2 code=$3 allowed=$4 sarif=$5 errors=$6 started=$7 results="" reason=""
  if [ "$code" = "conversion" ]; then
    reason="report conversion failed: $(stderr_tail "$errors")"
  elif ! exit_code_allowed "$code" "$allowed"; then
    reason="exit ${code}: $(stderr_tail "$errors")"
  elif ! results="$(payload count-results "$sarif" 2>>"$errors")"; then
    reason="unreadable SARIF: $(stderr_tail "$errors")"
  fi
  if [ -n "$reason" ]; then
    echo "::error::${tool} failed (${reason}); Spectra reports the analysis as incomplete."
    record_tool "$tool" failed "$version" "" "$reason" "$started"
    return 0
  fi
  upload_sarif "$sarif" "$tool"
  record_tool "$tool" ok "$version" "$results" "" "$started"
}

# Convert a native report to SARIF; returns "conversion" on failure.
convert_report() {  # tool report sarif version stderr_file exit_code
  if convert "$1" "$2" "$3" --version "$4" 2>>"$5"; then
    echo "$6"
  else
    echo conversion
  fi
}

# Setup each pinned binary through the generic provisioner (SPCTR-291) once.
setup_core_tool() {
  [ -n "$(tool_path "$1")" ] && return 0
  SPECTRA_TOOL_ASSET="$(catalog asset "$1")" \
    bash "${SCRIPT_DIR}/setup-tool.sh" "$1" || {
      echo "::error::$1 could not be provisioned; Spectra reports the analysis as incomplete."
      record_tool "$1" unavailable "" "" "the pinned binary could not be provisioned" "$SECONDS"
      return 1
    }
}

# ── Per-tool runners (one function per catalog tool) ────────────────────────
# Allowed exit codes: success plus "findings reported"; anything else is a
# tool failure. stderr goes to $WORK/<tool>.stderr for the manifest reason.

run_shellcheck() {
  local binary version code=0 started=$SECONDS errors="$WORK/shellcheck.stderr"
  binary="$(tool_path shellcheck)"
  version="$(tool_version "$binary")"
  echo "Running shellcheck ${version}"
  load_tool_files shellcheck
  # --norc: the repository's .shellcheckrc can never disable checks.
  "$binary" --norc -f json1 "${TOOL_FILES[@]}" > "$WORK/shellcheck.json" 2>"$errors" || code=$?
  code="$(convert_report shellcheck "$WORK/shellcheck.json" "$WORK/shellcheck.sarif" "$version" "$errors" "$code")"
  finish_sarif_tool shellcheck "$version" "$code" 0,1 "$WORK/shellcheck.sarif" "$errors" "$started"
}

run_hadolint() {
  local binary version code=0 started=$SECONDS errors="$WORK/hadolint.stderr"
  binary="$(tool_path hadolint)"
  version="$(tool_version "$binary")"
  echo "Running hadolint ${version}"
  load_tool_files hadolint
  "$binary" --config "${LINTERS_CONFIG_DIR}/hadolint.yaml" --format sarif "${TOOL_FILES[@]}" \
    > "$WORK/hadolint.sarif" 2>"$errors" || code=$?
  finish_sarif_tool hadolint "$version" "$code" 0,1 "$WORK/hadolint.sarif" "$errors" "$started"
}

run_biome() {
  local binary version code=0 started=$SECONDS errors="$WORK/biome.stderr"
  binary="$(tool_path biome)"
  version="$(tool_version "$binary")"
  echo "Running biome ${version}"
  load_tool_files biome
  # --config-path: Spectra's root biome.json, never the repository's.
  "$binary" lint --config-path="${LINTERS_CONFIG_DIR}/biome.json" \
    --reporter=sarif --reporter-file="$WORK/biome.sarif" \
    --max-diagnostics=none "${TOOL_FILES[@]}" >/dev/null 2>"$errors" || code=$?
  finish_sarif_tool biome "$version" "$code" 0,1 "$WORK/biome.sarif" "$errors" "$started"
}

run_tflint() {
  local binary version code=0 started=$SECONDS errors="$WORK/tflint.stderr"
  binary="$(tool_path tflint)"
  version="$(tool_version "$binary")"
  echo "Running tflint ${version}"
  # --config: the repository's .tflint.hcl is never read (no plugin it names
  # is ever enabled); --call-module-type=none keeps the run offline.
  # --recursive: every directory holding .tf files is a module to lint, not
  # only the root (SPCTR-304); it cannot be combined with --chdir.
  # Exit 2 means issues found; errors reported in the JSON fail the conversion.
  (cd "$BASE_PATH" && "$binary" --recursive --config="${LINTERS_CONFIG_DIR}/tflint.hcl" --format json \
    --call-module-type=none) > "$WORK/tflint.json" 2>"$errors" || code=$?
  code="$(convert_report tflint "$WORK/tflint.json" "$WORK/tflint.sarif" "$version" "$errors" "$code")"
  finish_sarif_tool tflint "$version" "$code" 0,2 "$WORK/tflint.sarif" "$errors" "$started"
}

run_buf() {
  local binary version file file_code code=0 started=$SECONDS errors="$WORK/buf.stderr"
  binary="$(tool_path buf)"
  version="$(tool_version "$binary")"
  echo "Running buf ${version}"
  : > "$WORK/buf.jsonl"
  : > "$errors"
  # Exit 100 means lint failures; the first other code fails the tool.
  while IFS= read -r file; do
    file_code=0
    "$binary" lint --config "${LINTERS_CONFIG_DIR}/buf.yaml" --error-format=json "$file" \
      >> "$WORK/buf.jsonl" 2>>"$errors" || file_code=$?
    if [ "$code" -eq 0 ] && ! exit_code_allowed "$file_code" 0,100; then
      code=$file_code
    fi
  done < <(tool_files buf)
  code="$(convert_report buf "$WORK/buf.jsonl" "$WORK/buf.sarif" "$version" "$errors" "$code")"
  finish_sarif_tool buf "$version" "$code" 0 "$WORK/buf.sarif" "$errors" "$started"
}

run_yamllint() {
  local binary version code=0 started=$SECONDS errors="$WORK/yamllint.stderr"
  binary="${WORK}/venv/bin/yamllint"
  version="$(tool_version "$binary")"
  echo "Running yamllint ${version}"
  load_tool_files yamllint
  # -c: Spectra's configuration, never the repository's .yamllint.
  # Exit 1 means errors found, 2 warnings only.
  "$binary" -f parsable -c "${LINTERS_CONFIG_DIR}/yamllint.yaml" "${TOOL_FILES[@]}" \
    > "$WORK/yamllint.txt" 2>"$errors" || code=$?
  code="$(convert_report yamllint "$WORK/yamllint.txt" "$WORK/yamllint.sarif" "$version" "$errors" "$code")"
  finish_sarif_tool yamllint "$version" "$code" 0,1,2 "$WORK/yamllint.sarif" "$errors" "$started"
}

run_sqlfluff() {
  local binary version code=0 started=$SECONDS errors="$WORK/sqlfluff.stderr"
  binary="${WORK}/venv/bin/sqlfluff"
  version="$(tool_version "$binary")"
  echo "Running sqlfluff ${version}"
  load_tool_files sqlfluff
  # Cosmetic layout rules are disabled; the dialect is an action input.
  # --ignore-local-config: no .sqlfluff / setup.cfg / tox.ini / pyproject.toml
  # from the repository, so its Jinja library_path is never imported; the
  # Spectra config pins the raw templater. .sqlfluffignore cannot hide files.
  # Exit 1 means violations found.
  "$binary" lint --ignore-local-config --config "${LINTERS_CONFIG_DIR}/spectra.sqlfluff" \
    --disregard-sqlfluffignores --format json --dialect "${SQLFLUFF_DIALECT}" --nocolor \
    --exclude-rules "${SQLFLUFF_EXCLUDED_RULES}" "${TOOL_FILES[@]}" \
    > "$WORK/sqlfluff.json" 2>"$errors" || code=$?
  code="$(convert_report sqlfluff "$WORK/sqlfluff.json" "$WORK/sqlfluff.sarif" "$version" "$errors" "$code")"
  finish_sarif_tool sqlfluff "$version" "$code" 0,1 "$WORK/sqlfluff.sarif" "$errors" "$started"
}

# ── Phases ──────────────────────────────────────────────────────────────────

# Python phase: ruff, cognitive complexity and jscpd (SPCTR-287, SPCTR-305),
# each recorded in the manifest. Lizard runs in the complexity phase.
run_python_phase() {
  linted_files '*.py' > "$WORK/py-files.txt"
  [ -s "$WORK/py-files.txt" ] || return 0
  ensure_venv
  run_ruff
  run_cognitive
  run_jscpd
}

# Complexity phase (SPCTR-305): Lizard on every language it reads, by file
# extension; the gate picks each function's language from its file.
run_complexity_phase() {
  local pattern patterns=()
  # Read into an array: an unquoted $(...) would glob-expand "*.py" here.
  while IFS= read -r pattern; do
    [ -n "$pattern" ] && patterns+=("$pattern")
  done < <(catalog paths lizard)
  linted_files "${patterns[@]}" > "$WORK/lizard-files.txt"
  [ -s "$WORK/lizard-files.txt" ] || return 0
  ensure_venv
  run_lizard
}

run_ruff() {
  local version code=0 started=$SECONDS errors="$WORK/ruff.stderr"
  version="$(tool_version "$WORK/venv/bin/ruff")"
  echo "Running ruff ${version}"
  "$WORK/venv/bin/ruff" check --config "${ACTION_DIR}/linters/ruff.toml" \
    --output-format sarif --output-file "$WORK/ruff.sarif" --exit-zero . 2>"$errors" || code=$?
  finish_sarif_tool ruff "$version" "$code" 0 "$WORK/ruff.sarif" "$errors" "$started"
}

# Record and upload a metrics tool (Lizard, jscpd) once it produced its report.
finish_metrics_tool() {  # tool version exit_code report stderr_file started_at
  local tool=$1 version=$2 code=$3 report=$4 errors=$5 started=$6 reason=""
  if [ "$code" != "0" ]; then
    reason="exit ${code}: $(stderr_tail "$errors")"
  elif [ ! -s "$report" ]; then
    reason="no report written: $(stderr_tail "$errors")"
  fi
  if [ -n "$reason" ]; then
    echo "::error::${tool} failed (${reason}); Spectra reports the analysis as incomplete."
    record_tool "$tool" failed "$version" "" "$reason" "$started"
    return 0
  fi
  payload payload metrics "$report" --tool "$tool" --base-path "$BASE_PATH" \
    ${CHANGED_ARGS[@]+"${CHANGED_ARGS[@]}"} > "$WORK/${tool}-payload.json"
  spectra_upload_result "$WORK/${tool}-payload.json" metrics || FAILED=1
  record_tool "$tool" ok "$version" "" "" "$started"
}

run_lizard() {
  local version code=0 started=$SECONDS errors="$WORK/lizard.stderr"
  version="$(tool_version "$WORK/venv/bin/lizard")"
  "$WORK/venv/bin/lizard" --csv -f "$WORK/lizard-files.txt" > "$WORK/lizard.csv" 2>"$errors" || code=$?
  finish_metrics_tool lizard "$version" "$code" "$WORK/lizard.csv" "$errors" "$started"
}

# Python cognitive complexity, the SonarQube measure (SPCTR-305): the source
# is parsed with ast by Spectra's vendored module, never imported or run.
run_cognitive() {
  local code=0 started=$SECONDS errors="$WORK/cognitive.stderr"
  python3 "${SCRIPT_DIR}/cognitive_report.py" --files "$WORK/py-files.txt" --base "$BASE_PATH" \
    > "$WORK/cognitive.json" 2>"$errors" || code=$?
  finish_metrics_tool cognitive "$COGNITIVE_VERSION" "$code" "$WORK/cognitive.json" "$errors" "$started"
}

# jscpd reads .jscpd.json from its working directory and the "jscpd" key of
# package.json, whatever --config says: it runs from a neutral directory on
# the explicit list of tracked Python files, with Spectra's config.
run_jscpd() {
  local file files=() code=0 started=$SECONDS errors="$WORK/jscpd.stderr"
  if ! command -v npx >/dev/null 2>&1; then
    echo "::error::Node.js (npx) not found: jscpd cannot measure duplication; Spectra reports the analysis as incomplete."
    record_tool jscpd unavailable "$JSCPD_VERSION" "" "Node.js (npx) not found on the runner" "$started"
    return 0
  fi
  while IFS= read -r file; do
    files+=("${BASE_PATH}/${file}")
  done < "$WORK/py-files.txt"
  mkdir -p "$WORK/jscpd-cwd"
  (cd "$WORK/jscpd-cwd" && npx --yes "jscpd@${JSCPD_VERSION}" --config "${LINTERS_CONFIG_DIR}/jscpd.json" \
    --reporters json --output "$WORK/jscpd" --silent --absolute --format python "${files[@]}" \
    >/dev/null 2>"$errors") || code=$?
  finish_metrics_tool jscpd "$JSCPD_VERSION" "$code" "$WORK/jscpd/jscpd-report.json" "$errors" "$started"
}

ensure_venv() {
  [ -x "${WORK}/venv/bin/pip" ] && return 0
  python3 -m venv "$WORK/venv"
  "$WORK/venv/bin/pip" install --quiet --disable-pip-version-check \
    --require-hashes --no-deps -r "${ACTION_DIR}/linters/requirements.txt"
}

# YAML/SQL phase shares the venv with the Python phase (yamllint, sqlfluff).
run_yaml_and_sql_phase() {
  local yaml_count sql_count
  yaml_count="$(linted_files '*.yml' '*.yaml' | wc -l)"
  sql_count="$(linted_files '*.sql' | wc -l)"
  [ "$yaml_count" -gt 0 ] || [ "$sql_count" -gt 0 ] || return 0
  ensure_venv
  [ "$yaml_count" -gt 0 ] && run_yamllint
  [ "$sql_count" -gt 0 ] && run_sqlfluff
  return 0
}

# SARIF files produced by the repository's own build job (PMD, detekt, clippy…).
# Comma-separated git pathspecs; only tracked files are ever ingested.
upload_repo_sarif() {
  [ -n "${REPO_SARIF:-}" ] || return 0
  local pathspec file
  while IFS= read -r pathspec; do
    [ -n "$pathspec" ] || continue
    while IFS= read -r file; do
      [ -n "$file" ] || continue
      echo "Ingesting repository-provided SARIF: ${file}"
      local tool results started=$SECONDS
      tool="repo-$(basename "${file}" .sarif | tr '[:upper:] .-' '[:lower:]___')"
      if results="$(payload count-results "$file" 2>"$WORK/${tool}.stderr")"; then
        upload_sarif "$file" "$tool"
        record_tool "$tool" ok "" "$results" "" "$started"
      else
        echo "::error::${file} is not a readable SARIF log; Spectra reports the analysis as incomplete."
        record_tool "$tool" failed "" "" "unreadable SARIF: $(stderr_tail "$WORK/${tool}.stderr")" "$started"
      fi
    done < <(git_ls_files -- "$pathspec")
  done < <(echo "${REPO_SARIF}" | tr ',' '\n')
}

# ── Main sequence ───────────────────────────────────────────────────────────

# ── New code reference (SPCTR-303): see new-code.sh ─────────────────────────
CHANGED_ARGS=()
bash "${SCRIPT_DIR}/new-code.sh" "$WORK"
if [ -f "$WORK/changed-lines.json" ]; then
  CHANGED_ARGS=(--changed-lines "$WORK/changed-lines.json")
fi
CHANGED_ARGS+=(--new-code "$WORK/new-code.json")

load_default_excludes
run_python_phase
run_complexity_phase
run_yaml_and_sql_phase

# Generic phase: pinned binaries whose catalog pathspecs match tracked files.
while IFS= read -r tool; do
  [ -n "$tool" ] || continue
  case "$tool" in
    ruff|lizard|cognitive|jscpd|yamllint|sqlfluff) continue ;;  # covered by the phases above
    shellcheck|hadolint|biome|tflint|buf) ;;
    *) continue ;;  # a catalog tool without a runner here must not abort the scan
  esac
  [ -n "$(tool_files "$tool" | head -1)" ] || continue
  if setup_core_tool "$tool"; then
    "run_${tool}"
  fi
done < <(catalog tools)

upload_repo_sarif

if [ -n "${COVERAGE_REPORT:-}" ] && [ -f "${COVERAGE_REPORT}" ]; then
  payload payload coverage "${COVERAGE_REPORT}" --base-path "$BASE_PATH" \
    ${CHANGED_ARGS[@]+"${CHANGED_ARGS[@]}"} > "$WORK/coverage-payload.json"
  spectra_upload_result "$WORK/coverage-payload.json" coverage || FAILED=1
elif [ -n "${COVERAGE_REPORT:-}" ]; then
  echo "::warning::Coverage report ${COVERAGE_REPORT} not found; coverage is not measured."
fi

# The manifest goes last: Spectra judges completeness from it (SPCTR-301).
catalog excludes > "$WORK/default-excludes.txt"
payload payload manifest "$MANIFEST" --excludes "$WORK/default-excludes.txt" --new-code "$WORK/new-code.json" \
  > "$WORK/manifest-payload.json"
spectra_upload_result "$WORK/manifest-payload.json" manifest || FAILED=1

# Advisory mode never blocks a merge: a failed upload is reported, and the
# verdict marks the analysis incomplete instead of failing the job.
if [ "$FAILED" -ne 0 ]; then
  if [ "${GATE_MODE:-advisory}" = "blocking" ]; then
    echo "::error::One or more Spectra linter uploads failed; aborting before /complete."
    exit 1
  fi
  echo "::warning::One or more Spectra linter uploads failed; advisory mode does not block, Spectra reports the analysis as incomplete."
fi
