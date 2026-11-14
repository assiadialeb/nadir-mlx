# Evidence block

Use this exact structure in the PR body and in the ticket's
`## Delivery evidence` section. Keep it factual; a reviewer should be able to
check every line by clicking or re-running something.

```markdown
## Delivery evidence — <KEY> @ <short SHA>

**Status:** READY FOR REVIEW | BLOCKED
**Risk:** low | medium | high (<why>)   **Iterations:** <n>/<max>

### Acceptance criteria
| AC | Statement (testable form) | Evidence |
|----|---------------------------|----------|
| AC1 | … | `tests/test_x.py::test_ac1_…` passed (RP launch <id> / local junit) |
| AC2 | … | … |

### Gates for <full SHA>
| Gate | Result | Source |
|------|--------|--------|
| Tests / CI | ✅ 214 passed, 0 failed | CI run <link> |
| Secrets (gitleaks) | ✅ 0 new | local gate / Spectra scan <id> |
| Dependencies (trivy) | ✅ 0 new ≥ HIGH | … |
| SAST (opengrep) | ✅ 0 new ≥ HIGH | … |
| SonarQube gate | ✅ OK / ⚠️ verified locally / ❌ not evaluable before merge | … |
| ReportPortal | ✅ 0 failed, 0 to investigate | launch <id> |
| Custom gates | ✅ navitia-nonreg: 0 unexplained diffs | … |
| Conventional Commits (dev health) | ✅ all branch subjects | local gate `commits` check / `validate_commit_msg.sh` |

### Needs a human
- <suppression / rotation / threshold / decision, with the exact proposal>

### Not caused by this change (follow-ups)
- <finding, where, proof it pre-exists>
```

Rules:
- One SHA per block. If you push again, write a new block; don't edit old
  results.
- "Verified locally" and "not evaluable" are legitimate results; hiding them is not.
- No secrets, tokens or customer data in the block, even partially masked.
