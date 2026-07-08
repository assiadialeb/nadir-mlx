# Draft / MTP compatibility matrix

Reference for speculative decoding options in Nadir MLX by **modality** and **model family**.

## Modality overview

| Modality | Launch mode | Mechanism | `advanced` keys |
|----------|-------------|-----------|-----------------|
| Text LLM | `TEXT` | mlx-lm speculative decoding | `draft_model`, `num_draft_tokens` |
| Vision-language | `MULTIMODAL` | mlx-vlm drafters | `draft_model`, `draft_kind`, `draft_block_size`, … |

!!! note "TEXT MTP"
    mlx-lm does not expose `--draft-kind mtp` in current releases. TEXT instances use classic `draft_model` + `num_draft_tokens` only. MTP on Gemma 4 is **MULTIMODAL-only** today.

## MULTIMODAL — `draft_kind` values

| `draft_kind` | mlx-vlm support | Nadir whitelist | Notes |
|--------------|-----------------|-----------------|-------|
| *(omit)* | Auto-detect | Yes | Server infers from checkpoint |
| `dflash` | Yes | Yes | DFlash drafter |
| `eagle3` | Yes | Yes | Eagle3 drafter |
| `mtp` | Gemma 4 assistants | Yes | Requires compatible assistant checkpoint |

## Gemma 4 MTP pairing (E2B / E4B)

| Target checkpoint | Assistant `draft_model` | Generation | Nadir save |
|-------------------|-------------------------|------------|------------|
| `*-qat-4bit` / QAT | `*-assistant-bf16` | Supported | Allowed |
| `*-qat-4bit` / QAT | `*-qat-assistant-4bit` | **Crash** (reshape) | **Rejected** |
| `*-4bit` (non-QAT) | `*-assistant-bf16` | Use bf16 assistant | Allowed if assistant unquantized |
| bf16 target | bf16 assistant | Supported | Allowed |

**UI helpers:**

- Registry default: `draft_kind: mtp` for `gemma4_vlm` family
- Performance profile: **Gemma 4 E4B/E2B MTP (perf)**
- Auto-suggest: QAT target → bf16 assistant folder name

## Other families (draft / speculative)

| Family | TEXT `draft_model` | MULTIMODAL `draft_kind` | Verified in Nadir |
|--------|--------------------|-------------------------|-------------------|
| Llama 3.x | Optional second checkpoint | `dflash` / `eagle3` if supported by mlx-vlm | Generic registry |
| Qwen 3 | Optional second checkpoint | Per mlx-vlm model card | Generic registry |
| Gemma 3 VLM | N/A | `dflash` / `eagle3` typical | Registry `gemma3_vlm` |
| Gemma 4 VLM | N/A | **MTP** (E2B/E4B) | Curated profiles + validation |

!!! tip "When in doubt"
    Start without draft, confirm baseline chat works, then enable MTP with bf16 assistant only. See [Gemma 4 MTP runbook](gateway-runbooks/gemma4-mtp.md).

## TEXT speculative decoding

| Field | Description |
|-------|-------------|
| `draft_model` | Local folder or HF id for draft model (same process) |
| `num_draft_tokens` | Tokens per speculative step |

No `draft_kind` on TEXT until mlx-lm ships MTP server support (tracked **MLX-67**).

### MLX-67 preview flag

When testing mlx-lm nightlies that expose `--draft-kind`, set:

```bash
export NADIR_TEXT_MTP_PREVIEW=1
```

This extends the TEXT advanced whitelist with `draft_kind` and `draft_block_size`, and maps them to mlx-lm CLI flags. **Do not enable in production** until mlx-lm MTP is officially released.

| TEXT field (preview) | CLI flag | Notes |
|----------------------|----------|-------|
| `draft_kind` | `--draft-kind` | e.g. `mtp` when mlx-lm supports it |
| `draft_block_size` | `--draft-block-size` | Paired with MTP drafts |
| `draft_model` | `--draft-model` | Always available (classic speculative) |
| `num_draft_tokens` | `--num-draft-tokens` | Always available |

## Validation layers

1. **UI** — Advanced JSON whitelist per launch mode
2. **Save** — `validate_mtp_draft_advanced()` rejects known-bad Gemma 4 QAT assistants
3. **Runtime** — mlx-vlm may still fail on unsupported combos; use smoke tests and deep health

## See also

- [Server config reference](server-config-reference.md)
- [Gemma 4 MTP runbook](gateway-runbooks/gemma4-mtp.md)
