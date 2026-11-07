#!/usr/bin/env bash
set -euo pipefail

if [ "${QUALITY_PROVIDER:-sonar}" = "none" ]; then
  echo "Sonar skipped (quality_provider=none)."
  exit 0
fi

if [ "${SCAN_STATUS:-}" = "bypassed" ]; then
  exit 0
fi

if [ -n "${SONAR_TOKEN_MANAGED:-}" ]; then
  SONAR_TOKEN="$SONAR_TOKEN_MANAGED"
else
  SONAR_TOKEN="${SONAR_TOKEN_CI:-}"
fi

if [ -n "${SONAR_TOKEN:-}" ]; then
  echo "::add-mask::$SONAR_TOKEN"
fi

if [ -z "${SONAR_HOST_URL:-}" ] || [ -z "${SONAR_PROJECT_KEY:-}" ]; then
  echo "Sonar host URL or project key missing."
  exit 1
fi
if [ -z "${SONAR_TOKEN:-}" ]; then
  echo "Missing Sonar token."
  exit 1
fi

SONAR_SCANNER_VERSION="${SONAR_SCANNER_VERSION:-8.3.1.16491}"
SONAR_SCANNER_SHA256="${SONAR_SCANNER_SHA256:-0f304e4c5982f0cfbd06762306d6e60e6ed9c8e3853629ae3b8ec2f4dc9bdc04}"
curl -fsSL -o sonar-scanner.zip \
  "https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-${SONAR_SCANNER_VERSION}-linux-x64.zip"
echo "${SONAR_SCANNER_SHA256}  sonar-scanner.zip" | sha256sum --check --strict
unzip -q sonar-scanner.zip

BASE_ARGS="-Dsonar.host.url=$SONAR_HOST_URL"
BASE_ARGS="$BASE_ARGS -Dsonar.token=$SONAR_TOKEN"
BASE_ARGS="$BASE_ARGS -Dsonar.projectKey=$SONAR_PROJECT_KEY"
if [ -n "${SONAR_ORGANIZATION:-}" ]; then
  BASE_ARGS="$BASE_ARGS -Dsonar.organization=$SONAR_ORGANIZATION"
fi
BASE_ARGS="$BASE_ARGS -Dsonar.projectVisibility=private"
BASE_ARGS="$BASE_ARGS -Dsonar.python.version=${PYTHON_VERSION:-3.12}"
BASE_ARGS="$BASE_ARGS -Dsonar.sources=${SONAR_SOURCES:-.}"
BASE_ARGS="$BASE_ARGS -Dsonar.scm.revision=${SCM_REVISION:-$GITHUB_SHA}"
if [ -n "${COVERAGE_REPORT:-}" ] && [ -f "${COVERAGE_REPORT}" ]; then
  BASE_ARGS="$BASE_ARGS -Dsonar.python.coverage.reportPaths=${COVERAGE_REPORT}"
fi

if [ -n "${SONAR_EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2086
  BASE_ARGS="$BASE_ARGS $SONAR_EXTRA_ARGS"
fi

BRANCH_NAME="${BRANCH:-$GITHUB_REF_NAME}"
BASE_BRANCH="${BASE_REF:-main}"
if [ -n "${PULL_REQUEST:-}" ]; then
  PR_ARGS="-Dsonar.pullrequest.key=$PULL_REQUEST"
  PR_ARGS="$PR_ARGS -Dsonar.pullrequest.branch=$BRANCH_NAME"
  PR_ARGS="$PR_ARGS -Dsonar.pullrequest.base=$BASE_BRANCH"
  PR_ARGS="$PR_ARGS -Dsonar.newCode.referenceBranch=$BASE_BRANCH"
  # shellcheck disable=SC2086
  ./sonar-scanner-*/bin/sonar-scanner $BASE_ARGS $PR_ARGS
else
  # shellcheck disable=SC2086
  ./sonar-scanner-*/bin/sonar-scanner $BASE_ARGS -Dsonar.branch.name="$BRANCH_NAME"
fi
