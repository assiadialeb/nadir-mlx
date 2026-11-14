# Commit messages (Conventional Commits)

Spectra **dev health** classifies each commit from its subject. Prefer types that
map cleanly to metrics (`feat` → feature, `perf` → refactor, `ci`/`build` →
chore). Vague subjects become `other` and may trigger the small-model fallback.

## Format

```
<type>(<scope>): <imperative subject>
```

- **Types:** `feat`, `fix`, `docs`, `refactor`, `test`, `style`, `chore` (also
  `perf`, `build`, `ci`, `revert` for tooling)
- **Subject:** English, specific, at least **8 characters** after `:`
- **Ticket:** put the Gravity key in scope or `Refs: SPCTR-123` in the body

Examples:

```
feat(qa): add release verdict strip (SPCTR-202)
fix(dev-health): correct period query on tab links
docs: document commit-type backfill CLI
```

## Optional `commit-msg` hook

From the repository root (Spectra ships the hook under `.cursor/skills/`):

```bash
chmod +x .cursor/skills/vela-vibe-loop/githooks/commit-msg
chmod +x .cursor/skills/vela-vibe-loop/scripts/validate_commit_msg.sh
git config core.hooksPath .cursor/skills/vela-vibe-loop/githooks
```

To validate a single subject without installing the hook:

```bash
printf '%s\n' 'feat(auth): refresh session cookie' | \
  .cursor/skills/vela-vibe-loop/scripts/validate_commit_msg.sh --subject
```

The **local gate** (`scripts/local_gate.sh`) also audits every non-merge commit
on the branch against the same rules before you push.
