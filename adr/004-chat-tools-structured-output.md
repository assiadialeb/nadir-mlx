# ADR 004 — Chat tools & structured output (qualification)

**Date:** 2026-06-22  
**Status:** Accepted  

## Context

Nadir Gateway proxies `POST /v1/chat/completions` to **mlx-lm** (TEXT) or **mlx-vlm** (MULTIMODAL) without interpreting the OpenAI payload. Clients (agents, SDKs) increasingly send:

- `tools` / `tool_choice` / `parallel_tool_calls`
- `response_format` with `json_object` or `json_schema`

We needed to qualify what works end-to-end vs what is relay-only, document curl examples, and ensure the gateway does not strip complex JSON.

## Decision

1. **Gateway behaviour (unchanged, now explicit):** deep-copy the request body, rewrite only `model` → upstream instance id, forward all other fields (including large `tools` arrays and nested `response_format`).
2. **Tools:** delegated entirely to **mlx-lm** / **mlx-vlm**. Tool calling works when the model tokenizer exposes `has_tool_calling` (typically via `tool_parser_type` in `tokenizer_config.json`). Otherwise mlx-lm logs a warning and ignores tools.
3. **Structured JSON:** gateway relays `response_format` as-is. **mlx-lm server does not enforce** OpenAI JSON mode or JSON Schema validation in v1 — output quality depends on the model and prompt. Document as *relay + best-effort*, not strict parity.
4. **Documentation:** runbook `gateway-runbooks/chat.md`, model matrix `chat-tools-model-matrix.md`, unit + gateway passthrough tests.

## Alternatives

| Option | Why rejected |
|--------|----------------|
| Implement tools in the gateway | Out of scope; duplicates mlx-lm |
| Reject `tools` at gateway when model unknown | Breaks transparent proxy; wrong layer |
| Post-process JSON schema in gateway | Heavy, model-specific, not Mac-killer-app v1 |

## Consequences

- **Positive:** Agent clients can send full tool payloads through `:11380` without gateway truncation.
- **Positive:** Clear operator docs on which model families support native tool parsing.
- **Negative:** `response_format: json_schema` may not produce valid JSON on all models — clients must handle parse errors.
- **Negative:** Model matrix requires manual updates when new tool-capable MLX weights are validated.

## References

- mlx-lm PR #217 (server tool calling)
- `orchestrator/gateway/chat_extensions.py`
- `docs/usage/gateway-runbooks/chat.md`
