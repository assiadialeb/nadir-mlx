# Triage: what to do with each red item

For every failing item, decide one of three outcomes and record it:

- **FIX** — caused by the change (or in a file it touched and cheap to fix
  safely); fix the root cause.
- **NEEDS HUMAN** — the right answer is a suppression, a threshold change, a
  rotation, a product decision or a design trade-off. Propose it; don't do it.
- **NOT OURS** — pre-existing, proven by the baseline (same fingerprint /
  same Sonar issue before the change). List it as a follow-up; don't fix it in
  this PR unless it is ≥ HIGH in a touched file and the fix is small.

Order: secrets → failing tests → new CRITICAL/HIGH security → Sonar security
rating → reliability → coverage → MEDIUM/LOW → style.

## Secrets (gitleaks, Spectra `type=secret`, Sonar S2068/S6418)
1. Is it a real credential? Fake values in test fixtures that exist to test a
   scanner (`fixtures/**/vuln_*`) are intentional: NEEDS HUMAN (propose a
   `.gitleaksignore` fingerprint entry with justification).
2. Real, and not yet committed: remove it, read it from config/env, add a
   placeholder in `.env.example`.
3. Real and already in history (any branch pushed): NEEDS HUMAN — it must be
   **rotated**; removing it from the file is not enough. Still remove it from
   the code in your change. Agent instruction files (`.cursor/rules`,
   `AGENTS.md`, `CLAUDE.md`) count: they are read by every agent and sent to
   models, so a key there is exposed.

## Failing tests
- Read the failure; reproduce locally with the narrowest command.
- Your change broke it → fix the code, not the test, unless the test asserts
  behaviour the ticket deliberately changes (then update it and say which AC
  justifies it).
- Fails on base too → NOT OURS; confirm by running it on the base commit.
- Flaky (passes on rerun without changes) → note it; don't hide it.

## Dependency vulnerabilities (trivy / Spectra SCA)
- Fixed version is in the description. Upgrade to the lowest version fixing
  all advisories for that package; check the changelog for breaking changes;
  run the full test suite.
- No fixed version, or upgrade breaks things → NEEDS HUMAN with a reachability
  note (is the vulnerable function used?).
- A new dependency you added that carries a known HIGH/CRITICAL → pick another
  version or library.

## SAST (opengrep / Spectra sast / Sonar vulnerabilities)
- Fix the pattern (parameterised queries, list-arg subprocess, escaping,
  safe deserialisation). Don't rename variables to dodge a rule.
- Believed false positive → NEEDS HUMAN with the reasoning and the exact
  suppression comment you propose.

## IaC / cloud (trivy misconfig, Spectra `type=cloud`, AWS-xxxx rules)
- Infra is high-risk: fix only if the ticket is about this infra; otherwise
  list it. Never relax an IAM/MFA/encryption control to pass.

## Sonar quality gate
- `new_software_quality_security_rating` / `reliability_rating` > 1: list
  new issues on new code for the branch, fix each.
- `new_coverage` below threshold: add meaningful tests for added lines
  (asserting behaviour, not just executing code). Don't add coverage
  exclusions.
- `new_duplicated_lines_density`: extract the shared code.

## Complexity / lint
- Function over the complexity limit (repo rules: ≤ 15): split by
  responsibility; keep behaviour identical; tests must still pass.

## When nothing converges
If the same items stay red after two iterations, stop. In the handoff explain
what you tried, why it didn't work, and what decision is needed.
