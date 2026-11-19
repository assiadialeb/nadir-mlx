#!/usr/bin/env bash
# New code reference for quality_provider=linters (SPCTR-303).
#
# Usage: new-code.sh WORK_DIR   (run from the repository checkout)
# Writes WORK_DIR/new-code.json (always) and WORK_DIR/changed-lines.json
# (only when new code is known). Environment:
#   NEW_CODE_REFERENCE  explicit reference commit (action input)
#   BASE_REF            pull request target branch
#   PUSH_BEFORE         commit before the push (github.event.before)
#   DEFAULT_BRANCH      repository default branch (new-branch pushes)
#   CURRENT_BRANCH      branch being scanned
#
# New code is the diff between an explicit reference commit and HEAD:
#   - pull request: merge-base with the target branch;
#   - push: the commit before the push (every pushed commit is new code);
#   - push of a new branch: merge-base with the default branch;
#   - NEW_CODE_REFERENCE (action input) overrides all of them.
# A shallow clone is deepened until the reference is reachable. When no
# reference can be found the scan reports "new code unknown": no finding is
# counted as new and the gate names the problem, instead of failing on the
# whole existing debt.
set -euo pipefail

NEW_CODE_REF=""
NEW_CODE_METHOD="none"
NEW_CODE_REASON=""
NULL_SHA="0000000000000000000000000000000000000000"

is_sha() {
  [[ "${1:-}" =~ ^[0-9a-f]{40}$ ]] && [ "$1" != "$NULL_SHA" ]
}

has_commit() {
  git cat-file -e "${1}^{commit}" 2>/dev/null
}

# Make a commit known locally: a push "before" SHA is fetched by id.
fetch_commit() {
  has_commit "$1" && return 0
  if [ "$(git rev-parse --is-shallow-repository 2>/dev/null)" = "true" ]; then
    git fetch --quiet --no-tags --depth=1 origin "$1" 2>/dev/null || true
  else
    git fetch --quiet --no-tags origin "$1" 2>/dev/null || true
  fi
  has_commit "$1"
}

# Merge-base of HEAD and a remote branch, deepening a shallow clone if needed.
merge_base_with() {
  local branch=$1 depth base
  git fetch --quiet --no-tags origin "+refs/heads/${branch}:refs/remotes/origin/${branch}" 2>/dev/null || return 1
  for depth in 50 200 1000 unshallow; do
    if base="$(git merge-base "origin/${branch}" HEAD 2>/dev/null)"; then
      echo "$base"
      return 0
    fi
    [ "$(git rev-parse --is-shallow-repository 2>/dev/null)" = "true" ] || return 1
    echo "Shallow clone: deepening history (${depth}) to find the merge-base with ${branch}." >&2
    if [ "$depth" = "unshallow" ]; then
      git fetch --quiet --no-tags --unshallow origin HEAD "+refs/heads/${branch}:refs/remotes/origin/${branch}" 2>/dev/null || return 1
    else
      git fetch --quiet --no-tags --deepen="$depth" origin HEAD "+refs/heads/${branch}:refs/remotes/origin/${branch}" 2>/dev/null || return 1
    fi
  done
  git merge-base "origin/${branch}" HEAD 2>/dev/null
}

resolve_new_code_reference() {
  local before="${PUSH_BEFORE:-}" base
  if [ -n "${NEW_CODE_REFERENCE:-}" ]; then
    if is_sha "$NEW_CODE_REFERENCE" && fetch_commit "$NEW_CODE_REFERENCE"; then
      NEW_CODE_REF="$NEW_CODE_REFERENCE"; NEW_CODE_METHOD="explicit"; return
    fi
    NEW_CODE_REASON="new-code-reference ${NEW_CODE_REFERENCE} is not a reachable commit"; return
  fi
  if [ -n "${BASE_REF:-}" ]; then
    if base="$(merge_base_with "$BASE_REF")"; then
      NEW_CODE_REF="$base"; NEW_CODE_METHOD="merge-base"; return
    fi
    NEW_CODE_REASON="no merge-base between HEAD and ${BASE_REF}"; return
  fi
  if is_sha "$before"; then
    if fetch_commit "$before"; then
      NEW_CODE_REF="$before"; NEW_CODE_METHOD="push-before"; return
    fi
    NEW_CODE_REASON="commit before the push (${before}) is not reachable"; return
  fi
  # New branch (before = 0000…) or no push event: fork point from the default branch.
  if [ -n "${DEFAULT_BRANCH:-}" ] && [ "${DEFAULT_BRANCH}" != "${CURRENT_BRANCH:-}" ] \
      && base="$(merge_base_with "$DEFAULT_BRANCH")" && [ "$base" != "$(git rev-parse HEAD)" ]; then
    NEW_CODE_REF="$base"; NEW_CODE_METHOD="default-branch-merge-base"; return
  fi
  if git rev-parse --verify --quiet HEAD~1 >/dev/null; then
    NEW_CODE_REF="$(git rev-parse HEAD~1)"; NEW_CODE_METHOD="parent"; return
  fi
  NEW_CODE_REASON="no reference commit: first commit of the repository"
}

WORK="${1:?usage: new-code.sh WORK_DIR}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$WORK"
rm -f "$WORK/changed-lines.json"

payload() {
  python3 "${SCRIPT_DIR}/linters_payload.py" "$@"
}

resolve_new_code_reference
if [ -n "$NEW_CODE_REF" ]; then
  # core.quotePath=false: non-ASCII paths stay literal (café.py, not "caf\303\251.py").
  if git -c core.quotePath=false diff -U0 --no-color --no-ext-diff "$NEW_CODE_REF" HEAD > "$WORK/diff.txt" 2>"$WORK/diff.stderr"; then
    payload changed-lines < "$WORK/diff.txt" > "$WORK/changed-lines.json"
    echo "New code: diff ${NEW_CODE_REF:0:12}..HEAD (${NEW_CODE_METHOD})."
  else
    NEW_CODE_REASON="git diff ${NEW_CODE_REF:0:12}..HEAD failed: $(tail -c 300 "$WORK/diff.stderr" | tr '\n' ' ')"
    NEW_CODE_REF=""
  fi
fi
if [ -z "$NEW_CODE_REF" ]; then
  echo "::warning::New code is unknown (${NEW_CODE_REASON}); no linter finding counts as new and Spectra reports it on the gate."
fi
payload new-code --reference "$NEW_CODE_REF" --method "$NEW_CODE_METHOD" --reason "$NEW_CODE_REASON" \
  > "$WORK/new-code.json"
