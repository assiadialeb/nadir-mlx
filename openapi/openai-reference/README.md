# OpenAI reference (curated)

Nadir does not target 100% of the OpenAI API. Contract tests validate only endpoints marked ✅ in `docs/usage/nadir-gateway-coverage-matrix.md`.

The canonical contract file is `openapi/nadir-curated.yaml`, shaped after OpenAI request/response schemas with explicit allowance for Nadir `metadata` extensions.

When adding a new ✅ endpoint to the coverage matrix, extend `nadir-curated.yaml` and add a `pytest -m contract` module under `orchestrator/tests/contracts/`.
