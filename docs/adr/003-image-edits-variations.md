# ADR 003: Image edits / variations — v1 scope

**Date:** 2026-06-22  
**Status:** Accepted (deferred)

## Context

We need OpenAI-compatible `POST /v1/images/edits` and `POST /v1/images/variations` in addition to `response_format: url` for generations.

Local image inference uses **mflux** with **txt2img** profiles only (`orchestrator/image_model_loader.py`). There is no img2img / inpainting path wired for FLUX, Z-Image, or Klein in Nadir MLX today.

## Decision

**v1 ships:**

- `POST /v1/images/generations` with `response_format: url` — PNG stored under `data/generated_images/`, served at `GET /v1/images/files/{id}` via Nadir Gateway (air-gapped, no external URL).
- `POST /v1/images/edits` and `POST /v1/images/variations` return **501 Not Implemented** with a clear message.

**Deferred:**

- img2img / inpainting when mflux (or an alternate local backend) exposes a stable API for our supported families.
- Gateway multipart proxy for edits is stubbed; routes exist so clients get explicit errors instead of 404.

## Consequences

- OpenAI-compatible clients that require edits must use `b64_json` generations or an external provider.
- `url` responses always point at the **gateway** base (`NADIR_GATEWAY_PUBLIC_BASE_URL`), not instance ports.

## References

- `orchestrator/image_server.py`
- `orchestrator/image_assets.py`
