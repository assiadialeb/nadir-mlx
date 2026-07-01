# Runbook — Gemma 4 MTP (QAT target + bf16 assistant)

Operational procedure for **Multi-Token Prediction (MTP)** on Gemma 4 E2B/E4B via mlx-vlm, through Nadir MLX.

!!! warning "Assistant quantization"
    The **target** may stay QAT/4bit. The **draft assistant** must be **bf16** (`*-assistant-bf16`). Quantized assistants load but **crash at first generation** with a `MaskedEmbedder` reshape error in mlx-vlm 0.6.x.

## Prerequisites

- Django control plane (`:8000`) and Nadir Gateway (`:11380`) running
- Local model folders installed under `MODELS_DIR`:
  - Target, e.g. `gemma-4-E4B-it-qat-4bit`
  - Assistant, e.g. `gemma-4-E4B-it-assistant-bf16`
- Apple Silicon host with enough unified memory for both checkpoints

## 1. Create the MULTIMODAL server (UI)

1. Open **Servers** → **Create server**
2. **Server type:** Multimodal
3. **Model:** `gemma-4-E4B-it-qat-4bit` (or E2B QAT variant)
4. Expand **Server configuration**
5. Either:
   - Select performance profile **Gemma 4 E4B MTP (perf)**, or
   - Rely on auto-suggest (QAT target → bf16 assistant hint), or
   - Paste Advanced JSON manually (see below)
6. Set **Gateway alias** (e.g. `gemma-4-e4b-mtp`)
7. **Start server** and wait until status is **Running**

**Advanced JSON (manual):**

```json
{
  "draft_kind": "mtp",
  "draft_model": "gemma-4-E4B-it-assistant-bf16"
}
```

Use the **folder name** under `MODELS_DIR`, or an absolute path. Nadir rejects QAT/4bit assistant paths at save time.

## 2. Verify registration

```bash
curl -s http://127.0.0.1:11380/v1/models | python3 -m json.tool
```

Confirm your alias appears with `"launch_mode": "MULTIMODAL"`.

## 3. Smoke generation (gateway)

```bash
curl -s http://127.0.0.1:11380/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<alias>",
    "messages": [{"role": "user", "content": "Reply with the word ok."}],
    "max_tokens": 16
  }' | python3 -m json.tool
```

Expected: `object: chat.completion` with non-empty `choices[0].message.content`.

Automated smoke (optional):

```bash
export NADIR_SMOKE_GATEWAY_URL=http://127.0.0.1:11380
export NADIR_SMOKE_MTP_ALIAS=<alias>
pytest -m smoke orchestrator/tests/smoke/test_gateway_smoke.py::test_smoke_multimodal_mtp_generation -q
```

## 4. Benchmark MTP (optional)

Use the **Benchmark** page with the same instance, or compare draft on/off in a future A/B UI (MLX-84). Reference: ~180 aggregate tok/s reported for E4B QAT + bf16 assistant on Apple Silicon (hardware-dependent).

## Symptom → cause → fix

| Symptom | Likely cause | Fix |
|---------|----------------|-----|
| Save rejected: MTP assistant error | QAT/4bit `draft_model` | Use `*-assistant-bf16` folder |
| Server **Running**, first chat crashes | Assistant still quantized or wrong pairing | Check Advanced JSON; read instance log |
| Log: `[reshape] Cannot reshape array...` | mlx-vlm MTP + quantized ordered assistant | Switch assistant to bf16; restart server |
| `503 model_unavailable` | Instance stopped or loading | Start server; wait for **Running** |
| `503 model_waking_timeout` | `on_demand` cold start too slow | Increase `NADIR_GATEWAY_WAKE_TIMEOUT_SECONDS` |
| Empty or slow first token | Cold load + MTP draft load | Normal on first request; retry after warm |

## Read logs

```bash
# From Django UI: Servers → instance → View logs
# Or on disk:
tail -f logs/<model-folder>_<port>.log
```

Search for `MaskedEmbedder`, `reshape`, or `draft_kind`. Nadir logs MTP-related failures with model id context when detected.

## Deep health (optional)

```bash
export NADIR_DEEP_INSTANCE_HEALTH=1
```

The orchestrator may run a minimal `/v1/chat/completions` probe on RUNNING TEXT/MULTIMODAL instances (rate-limited). A process that answers `/health` but fails generation is marked **DEGRADED** in the UI.

## Related docs

- [Draft / MTP compatibility matrix](../draft-mtp-compatibility-matrix.md)
- [Server config reference — MULTIMODAL advanced](../server-config-reference.md#advanced-json-multimodal-mlx-vlm)
