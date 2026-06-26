# ADR 005 — VLM vision E2E via Nadir Gateway

**Date:** 2026-06-22  
**Status:** Accepted  
**Ticket:** MLX-37

## Context

MULTIMODAL instances run **mlx-vlm** and are reachable through `POST /v1/chat/completions` on the gateway (`:11380`). Clients send OpenAI-style messages with `content` arrays containing `image_url` blocks (base64 data URIs, local paths, or HTTP URLs).

MLX-37 required end-to-end qualification: gateway relay, streaming, OpenAI-compatible client support, and documented limits — without implementing vision in the gateway itself.

## Decision

1. **Gateway:** reuse the chat proxy (`prepare_chat_upstream_body` deep-copy). Only `model` is rewritten to the instance upstream id. Multimodal `messages` (including large base64 blobs) are forwarded unchanged.
2. **Vision execution:** entirely delegated to **mlx-vlm** on the instance port.
3. **Recommended image transport (air-gapped):**
   - `data:image/jpeg;base64,...` or `data:image/png;base64,...` inline in JSON
   - Absolute local file path (`/path/to/image.jpg`) when the VLM process can read the filesystem
   - Gateway-hosted PNG from MLX-34 (`http://127.0.0.1:11380/v1/images/files/{id}`) when the client already generated an image locally
4. **Documentation:** dedicated runbook `gateway-runbooks/vlm.md`, model matrix `vlm-vision-model-matrix.md`, gateway + qualification tests.

## Known upstream limits (mlx-vlm)

| Topic | Status |
|-------|--------|
| Single image per message (OpenAI `image_url` shape) | ✅ primary path |
| Multiple `image_url` blocks in one message | ⚠️ some mlx-vlm versions keep **only the last** image ([issue #1084](https://github.com/Blaizzy/mlx-vlm/issues/1084)) |
| External `https://` image URLs | ⚠️ requires network on the Mac (not air-gapped) |
| SSE `stream: true` with vision | ✅ gateway relays; mlx-vlm dependent |
| `/v1/completions` on VLM alias | ❌ gateway returns 400 by design |

## Alternatives

| Option | Why rejected |
|--------|----------------|
| Resize / re-encode images in gateway | Out of scope; duplicates mlx-vlm |
| Reject oversized base64 at gateway | Breaks transparent proxy; clients vary |

## Consequences

- **Positive:** Vision chat routes work through Nadir with zero gateway changes.
- **Positive:** Operators have curl + Python examples for Mac Studio QA.
- **Negative:** Very large inline images can hit `NADIR_GATEWAY_PROXY_TIMEOUT_SECONDS` (default 300s).
- **Negative:** Model-specific vision quality still varies (Gemma 3/4 VLM, Qwen-VL, etc.).

## References

- `orchestrator/mlx_vlm_launcher.py`
- `orchestrator/gateway/chat_extensions.py`
- `docs/usage/gateway-runbooks/vlm.md`
