#!/usr/bin/env bash
# Local gate for the vela-vibe-loop skill.
# Runs tests, linters and the same scanners Spectra uses on the current branch,
# and writes .vela/gate-report.json.
#
# Exit codes: 0 = all checks passed, 1 = at least one failed,
#             2 = nothing failed but at least one check was skipped (not green).
#
# Configuration via environment (defaults in brackets):
#   BASE_BRANCH      [main]
#   TEST_CMD         [pytest -q --junitxml=.vela/junit.xml]
#   LINT_CMD         [] (empty = skipped)
#   OPENGREP_CONFIG  [.opengrep]
#   SEVERITY         [HIGH,CRITICAL]
set -uo pipefail

BASE_BRANCH="${BASE_BRANCH:-main}"
TEST_CMD="${TEST_CMD:-pytest -q --junitxml=.vela/junit.xml}"
LINT_CMD="${LINT_CMD:-}"
OPENGREP_CONFIG="${OPENGREP_CONFIG:-.opengrep}"
SEVERITY="${SEVERITY:-HIGH,CRITICAL}"
export AGENT_NO_ANALYTICS=1

OUT=".vela"
mkdir -p "$OUT"
# Remove previous reports so scanners never read their own output.
rm -f "$OUT"/*.json "$OUT"/*.log "$OUT"/*.xml
REPORT="$OUT/gate-report.json"
HEAD_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
MERGE_BASE="$(git merge-base "origin/$BASE_BRANCH" HEAD 2>/dev/null || git merge-base "$BASE_BRANCH" HEAD 2>/dev/null || echo "")"

mapfile -t CHANGED < <(
  if [ -n "$MERGE_BASE" ]; then git diff --name-only --diff-filter=ACMR "$MERGE_BASE"; fi
  git diff --name-only --diff-filter=ACMR --cached
  git ls-files --others --exclude-standard
)
mapfile -t CHANGED < <(printf '%s\n' "${CHANGED[@]}" | grep -v '^$' | sort -u)

RESULTS=()
FAILED=0
SKIPPED=0

record() { # name status detail
  local detail="${3//\\/\\\\}"; detail="${detail//\"/\\\"}"
  RESULTS+=("{\"check\":\"$1\",\"status\":\"$2\",\"detail\":\"$detail\"}")
  case "$2" in fail) FAILED=1 ;; skipped) SKIPPED=1 ;; esac
  printf '%-10s %-8s %s\n' "$1" "$2" "$3"
}

have() { command -v "$1" >/dev/null 2>&1; }

# 1. Tests
if [ -n "$TEST_CMD" ]; then
  if bash -c "$TEST_CMD" >"$OUT/tests.log" 2>&1; then record tests pass "$TEST_CMD"
  else record tests fail "see $OUT/tests.log"; fi
else record tests skipped "TEST_CMD empty"; fi

# 2. Lint / complexity
if [ -n "$LINT_CMD" ]; then
  if bash -c "$LINT_CMD" >"$OUT/lint.log" 2>&1; then record lint pass "$LINT_CMD"
  else record lint fail "see $OUT/lint.log"; fi
else record lint skipped "LINT_CMD empty"; fi

# 3. Secrets: commits on this branch + working tree
if have gitleaks; then
  rc=0
  if [ -n "$MERGE_BASE" ]; then
    gitleaks git --log-opts="$MERGE_BASE..HEAD" --no-banner --redact \
      --report-format json --report-path "$OUT/gitleaks-history.json" >/dev/null 2>&1 || rc=1
  fi
  gitleaks dir . --no-banner --redact \
    --report-format json --report-path "$OUT/gitleaks-tree.json" >/dev/null 2>&1 || rc=1
  if [ $rc -eq 0 ]; then record secrets pass "gitleaks"
  else record secrets fail "see $OUT/gitleaks-*.json (redacted)"; fi
else record secrets skipped "gitleaks not installed"; fi

# 4. Dependencies + misconfig + secrets in files
if have trivy; then
  if trivy fs --quiet --skip-dirs .vela --scanners vuln,secret,misconfig --severity "$SEVERITY" \
       --exit-code 1 --format json -o "$OUT/trivy.json" . >/dev/null 2>&1
  then record trivy pass "no $SEVERITY"
  else record trivy fail "see $OUT/trivy.json"; fi
else record trivy skipped "trivy not installed"; fi

# 5. SAST on changed files only (pre-existing findings are not this change's)
if have opengrep; then
  if [ ${#CHANGED[@]} -eq 0 ]; then record sast pass "no changed files"
  elif [ ! -e "$OPENGREP_CONFIG" ]; then record sast skipped "config $OPENGREP_CONFIG missing"
  else
    existing=()
    for f in "${CHANGED[@]}"; do [ -f "$f" ] && existing+=("$f"); done
    if [ ${#existing[@]} -eq 0 ]; then record sast pass "only deletions"
    elif opengrep scan --quiet --config "$OPENGREP_CONFIG" --severity ERROR \
         --error --json --output "$OUT/opengrep.json" "${existing[@]}" >/dev/null 2>&1
    then record sast pass "${#existing[@]} files"
    else record sast fail "see $OUT/opengrep.json"; fi
  fi
else record sast skipped "opengrep not installed"; fi

# 6. Conventional Commits on this branch (Spectra dev health)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATOR="$SCRIPT_DIR/validate_commit_msg.sh"
if [ ! -x "$VALIDATOR" ]; then
  record commits skipped "validate_commit_msg.sh missing"
elif [ -z "$MERGE_BASE" ]; then
  record commits skipped "no merge base"
else
  bad_subjects=()
  while IFS= read -r sha; do
    subject="$(git log -1 --format=%s "$sha")"
    if ! printf '%s\n' "$subject" | "$VALIDATOR" --subject 2>/dev/null; then
      bad_subjects+=("$sha $subject")
    fi
  done < <(git rev-list --no-merges "$MERGE_BASE..HEAD")
  if [ ${#bad_subjects[@]} -eq 0 ]; then
    record commits pass "Conventional Commits"
  else
    detail=$(printf '%s; ' "${bad_subjects[@]}")
    record commits fail "non-CC subjects: ${detail%; }"
  fi
fi

# Report
{
  printf '{"head":"%s","merge_base":"%s","changed_files":%d,"checks":[' \
    "$HEAD_SHA" "$MERGE_BASE" "${#CHANGED[@]}"
  (IFS=,; printf '%s' "${RESULTS[*]}")
  printf ']}\n'
} >"$REPORT"
echo "report: $REPORT (HEAD $HEAD_SHA)"

if [ $FAILED -ne 0 ]; then exit 1; fi
if [ $SKIPPED -ne 0 ]; then exit 2; fi
exit 0
