# ADR 007 — Gemma 4 MTP and assistant quantization

**Date:** 2026-06-24  
**Status:** Accepted  

## Context

Gemma 4 E2B/E4B vision-language models support **Multi-Token Prediction (MTP)** in mlx-vlm via a separate **assistant** checkpoint (`draft_model`, `draft_kind: mtp`). Operators want QAT/4bit **targets** for memory efficiency while maximizing throughput with MTP.

Incident observed on mlx-vlm 0.6.x:

- Target: `gemma-4-E4B-it-qat-4bit` (loads OK)
- Assistant: `gemma-4-E4B-it-qat-assistant-4bit` (loads OK)
- First `POST /v1/chat/completions` → `ValueError: [reshape] Cannot reshape array...` in `masked_embedder.py`

Root cause: assistants with `model_type: gemma4_assistant`, `use_ordered_embeddings: true`, and quantization bits &lt; 16 are incompatible with MTP `MaskedEmbedder` in current mlx-vlm.

## Decision

1. **Allow** MTP with **quantized target + bf16 assistant** (validated ~180 agg tok/s on reference hardware).
2. **Reject at save** any MULTIMODAL config with `draft_kind: mtp` and a quantized assistant (path heuristic + local `config.json` inspection via `mtp_draft_validation.py`).
3. **Ship UX defaults:**
   - Registry `gemma4_vlm.advanced.draft_kind: mtp`
   - Performance profiles for E4B/E2B QAT → bf16 assistant
   - UI auto-suggest and warning on QAT assistant paths
4. **Document** runbook, compatibility matrix, and gateway ops paths (MLX-76/77).

TEXT launch mode remains on classic `draft_model` / `num_draft_tokens` until mlx-lm exposes MTP (see MLX-67).

## Alternatives considered

| Alternative | Why rejected |
|-------------|--------------|
| Allow QAT assistant and patch mlx-vlm at runtime | Fragile; upstream fix belongs in mlx-vlm |
| Silent fallback to non-MTP generation | Hides misconfiguration; violates fail-fast ops |
| Block all quantized targets with MTP | Validated perf path uses QAT target + bf16 assistant |
| Second endpoint for draft model | Contradicts Nadir model: draft is same-process `advanced` config |

## Consequences

**Positive**

- Operators cannot save configs known to crash at generation time
- One-click profiles reduce JSON errors
- Smoke test `NADIR_SMOKE_MTP_ALIAS` guards regressions

**Negative**

- Assistant must be downloaded separately (bf16 folder)
- Slight memory increase vs QAT assistant (not usable anyway)
- Validation depends on mlx-vlm 0.6.x behaviour; re-verify on major upgrades

**Follow-up**

- Revisit when mlx-vlm supports quantized ordered assistants
- mlx-lm MTP for TEXT (MLX-67)
- Golden benchmark CI for MTP configs (MLX-74)
