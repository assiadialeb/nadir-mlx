# Reading each tool

Tool names below are the MCP tool names as exposed today; your client may
prefix them (e.g. `mcp__Spectra__spectra_get_scan`). If a server is not
connected, use the CLI fallback and say in the handoff that the gate was
verified locally.

## Contents
1. Gravity (tickets)
2. Spectra (security & quality scans)
3. SonarQube
4. ReportPortal (test evidence)
5. CI
6. Path normalisation

---

## 1. Gravity

| Need | Tool |
|---|---|
| ticket by key | `gravity_issue_get_by_key(issue_key)` → `id`, `project_id`, `description`, `state`, `parent_issue_id` |
| project ids/codes | `gravity_projects_list` (codes: GP, GRVT, MLX, SPCTR, VLSK) |
| workflow states | `gravity_states_list(project_id)` — use `slug` (`in-progress`, `test`, `done`) |
| change state | `gravity_issue_patch(project_id, issue_id, '{"state_slug":"test"}')` |
| append evidence | read `description`, append section, `gravity_issue_patch(... '{"description": "<full text>"}')` |

Notes:
- Some triage states have an empty slug; use `state_id` for those.
- There is no comment tool yet, so evidence is appended to the description.
  Re-read right before patching to avoid overwriting a concurrent edit.
- Acceptance criteria are prose under `## Acceptance`; extract them into AC1..n.

## 2. Spectra

Spectra has two modules. Know which one you are reading:

| Module | What it is | Commit-attributable? | Use in the loop |
|---|---|---|---|
| **CI integration** | runs in the pipeline on each push/PR; aggregates Sonar metrics with Spectra policies (secrets, SCA, IaC, SBOM); posts "Spectra — Quality Gate" on the PR; UI at `/ci/<project>/scans/<id>` with branch + short SHA | yes | **the remote gate** |
| **Manual/scheduled audit** | full audit (daily or more), MCP tools below | no SHA/branch on the scan | baseline, posture, trend |

Reading the CI gate (until the MCP exposes CI scans):
1. On the PR, find the latest "Spectra — Quality Gate" comment or check and
   confirm it refers to HEAD (comment posted after your last push, SHA in the
   linked scan). Older comments describe older commits.
2. Its failed conditions are the work list. Open the linked scan for
   file/line details if needed.
3. A failed Spectra gate is blocking for you **even if the forge lets the PR
   merge** (the check may be advisory). Never treat "merge button is green"
   as "gate is green".
4. Conditions may be computed on the whole project rather than new code
   (e.g. "Leaked secrets: 7" on a PR that added none). Prove which items your
   diff introduced (local gitleaks on `<base>..HEAD`); fix those; report the
   rest as NOT OURS + rotation needed, rather than claiming the gate is fine.

Manual-audit MCP tools:

| Need | Tool |
|---|---|
| find project | `spectra_list_projects(integration_source="all")`; use the id in loop.yaml |
| recent audits | `spectra_list_project_scans(project_id, limit)` newest first |
| summary | `spectra_get_scan(scan_id)` → severity counts, `secrets_count`, `sca_vulnerable_count`, grades |
| findings | `spectra_list_scan_issues(scan_id, types?, include_ignored=false)` |
| health | `spectra_suite_health` |

Finding fields: `type` (vulnerability, secret, quality, cloud), `severity`,
`tool` (sast, trivy, gitleaks…), `rule_id`, `fingerprint`, `file_path`,
`line_number` (often null), `description`.

- Fingerprints can repeat inside one scan; dedupe before counting.
- **SCA findings** have `file_path` = the ecosystem ("Python") and the fixed
  version only in `description`. Parse it, find the manifest
  (`requirements*.txt`, `pyproject.toml`, `uv.lock`, `package.json`), bump to
  the lowest fixed version that satisfies all findings for that package.
- Do not trust grades as a pass signal; use counts and gate conditions.

CLI fallback (same engines):
```
gitleaks git --log-opts="<base>..HEAD" --report-format json --report-path .vela/gitleaks.json
trivy fs --scanners vuln,secret,misconfig --severity HIGH,CRITICAL --format json -o .vela/trivy.json .
opengrep scan --config "$OPENGREP_CONFIG" --json --output .vela/opengrep.json <changed files>
```

## 3. SonarQube

| Need | Tool |
|---|---|
| project key | `sonarqube_list_projects(q)` (keys are snake_case: `spectra`, `nadir_mlx`, `gitpulse`, `vela_stack`) |
| branches analysed | `sonarqube_list_branches(project)` — lists branches only, **not PRs** |
| gate | `sonarqube_get_quality_gate_status(projectKey, branch?|pullRequest?)` → `status`, failing `conditions` |
| last analysis time | `sonarqube_list_project_analyses(project)` |
| issues to fix | `sonarqube_search_issues(componentKeys, branch?|pullRequest?, issueStatuses="OPEN,CONFIRMED", impactSoftwareQualities?, createdAfter?)` |
| metric values | `sonarqube_get_measures(component, "new_coverage,new_software_quality_security_rating,...")` |

Notes:
- Don't pass the optional `_intent` field; it isn't needed.
- **PR analysis.** The server analyses branches and PRs (branch plugin).
  Query the PR with `pullRequest="<number>"`, not `branch`; PRs don't
  appear in `sonarqube_list_branches`. Never read main's gate as your PR's.
- **Check the conditions, not just the status.** A PR gate can hold fewer
  conditions than main's (e.g. no coverage condition when no coverage report
  was uploaded for the PR). If a condition you expect is missing, the PR
  isn't fully evaluated: say so, and verify coverage locally.
- PR gate green does not mean main stays green: main's "new code" period can
  span days of other changes. Your responsibility is the PR gate.
- **Freshness.** The gate is only meaningful if the last analysis date is
  after your push.
- Gate conditions are on new code (`new_coverage`, `new_*_rating`). Fix in
  this order: security rating → reliability rating → coverage → duplication.
  Coverage: add tests for the lines you added, not unrelated files.
- Reliability issues in templates are often accessibility rules (Web:S68xx):
  fix them properly (real interactive elements, required ARIA attributes);
  public-sector customers require accessibility.

## 4. ReportPortal

No MCP server may be connected. Expected conventions:
- Launch attributes: `event`, `branch`, `commit` (full SHA), `ticket` when run
  for one ticket.
- Test items carry `ticket:<KEY>` from `@pytest.mark.ticket("KEY")`.
- Pass condition: launch for HEAD exists, `failed = 0`, `to_investigate = 0`,
  and each AC maps to ≥1 passed item.

Fallback: run the suite locally with JUnit output
(`pytest -q --junitxml=.vela/junit.xml`) and map AC from markers. Say so.

## 5. CI

Read the required checks for HEAD from the forge (GitHub/GitLab MCP or `gh pr
checks` / `glab ci status`). A pending check is not a pass. A flaky test is a
failure until a rerun passes; mention the flake in the handoff.

## 6. Path normalisation

Tools report paths differently (Spectra may prefix the repo name,
`Spectra/app/...`; Sonar uses `app/...`; component keys are `spectra:app/...`).
Normalise to repo-relative paths before comparing findings with your diff.
