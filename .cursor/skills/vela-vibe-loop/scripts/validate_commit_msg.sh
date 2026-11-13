#!/bin/sh
# Conventional Commits check (Vela dev health / Spectra classification).
# Hook: validate_commit_msg.sh "$1"   Branch audit: git log … | xargs -I{} … --subject
set -eu

CC_PATTERN='^(feat|fix|docs|refactor|test|style|chore|perf|build|ci|revert)(\([^)]+\))?(!)?: .{8,}$'

validate_subject() {
    line=$1
    case "$line" in
        Merge\ *) return 0 ;;
        Revert\ \"*) return 0 ;;
    esac
  if printf '%s\n' "$line" | grep -Eq "$CC_PATTERN"; then
    return 0
  fi
  return 1
}

fail() {
    line=$1
    echo "commit-msg: subject must be Conventional Commits (type(scope): subject, ≥8 chars after colon)" >&2
    echo "  types: feat, fix, docs, refactor, test, style, chore, perf, build, ci, revert" >&2
    echo "  got: $line" >&2
    exit 1
}

if [ "${1:-}" = "--subject" ]; then
    IFS= read -r line || exit 1
    validate_subject "$line" || fail "$line"
    exit 0
fi

msg_file=${1:-}
if [ -z "$msg_file" ] || [ ! -f "$msg_file" ]; then
    echo "usage: $0 <commit-msg-file> | --subject" >&2
    exit 2
fi

line=$(sed -n '1p' "$msg_file")
validate_subject "$line" || fail "$line"
