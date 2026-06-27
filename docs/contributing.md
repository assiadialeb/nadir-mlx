# Contributing

Thank you for contributing to Nadir MLX.

## Development setup

```bash
git clone https://github.com/assiadialeb/nadir-mlx.git
cd nadir-mlx
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python manage.py migrate
```

Optional quality benchmark dependencies:

```bash
pip install -r requirements-quality.txt
```

Documentation build dependencies:

```bash
pip install -r requirements-docs.txt
```

## Running tests

```bash
pip install -r requirements-test.txt
export DJANGO_SETTINGS_MODULE=mlx_orchestrator.settings
pytest orchestrator/tests -q
```

Most tests use mocks and do not require a running MLX model or Apple Silicon GPU in CI.

### API contract tests (MLX-47)

OpenAPI contract tests for gateway endpoints marked ✅ in the coverage matrix:

```bash
pytest -m contract orchestrator/tests/contracts -q
```

Live smoke tests (optional, against a running gateway):

```bash
export NADIR_SMOKE_GATEWAY_URL=http://127.0.0.1:11380
export NADIR_SMOKE_MODEL_ALIAS=<text-alias>   # for chat smoke only
pytest -m smoke orchestrator/tests/smoke -q
```

Contract suite scope is defined in `openapi/nadir-curated.yaml` (endpoints marked ✅ in the coverage matrix).

Disable the instance watchdog during one-off commands:

```bash
export MLX_DISABLE_INSTANCE_WATCHDOG=1
```

## Documentation

Build and preview locally:

```bash
pip install -r requirements-docs.txt
mkdocs serve
# open http://127.0.0.1:8000 (MkDocs dev server — not Django)
```

Strict build (same as CI):

```bash
mkdocs build --strict
```

Published docs: [https://assiadialeb.github.io/nadir-mlx/](https://assiadialeb.github.io/nadir-mlx/)

When adding or moving pages under `docs/`, update `nav` in `mkdocs.yml`.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add gateway health endpoint
fix: handle empty SBOM in dependency parser
docs: update STT runbook
test: add unit tests for wake timeout
chore: bump mlx-lm dependency
```

Allowed types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`, `style`, `perf`.

## Pull requests

1. Branch from `main` (e.g. `feat/my-feature`)
2. Keep changes focused; update docs when behaviour changes
3. Run `pytest` and `mkdocs build --strict` when touching docs
4. Open a PR against `main`

## Code standards

- Type hints on public Python functions (PEP 484)
- Business logic in `services.py` / `selectors.py`; thin views
- No raw stack traces in user-facing errors
- Timezone-aware datetimes (`datetime.now(timezone.utc)`)

## License

By contributing, you agree that your contributions will be licensed under the [GNU Affero General Public License v3.0](https://github.com/assiadialeb/nadir-mlx/blob/main/LICENSE).
