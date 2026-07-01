# Server configuration reference

Reference for Nadir inference instance `server_config`: standard form fields, lifecycle (`ops`), and the **Advanced (JSON)** block in the Servers UI.

!!! note "Living document"
    Keys and wiring follow `orchestrator/server_config_schema.py` and `orchestrator/server_manager.py`. Update this page when the whitelist or launch commands change.

Last updated: June 2026 (lifecycle modes).

## Where to configure

1. Open **Servers** in the Nadir UI.
2. Create a server or **Edit** an existing instance.
3. Expand **Server configuration**.
4. Standard fields render as form inputs; expert flags go in **Advanced (JSON)**.
5. **Stop** and **Start** (or create fresh) after changes — flags are passed at process launch only.

The UI shows **Allowed keys** for the selected launch mode under the Advanced textarea.

## `server_config` shape

Stored on each `InferenceInstance` as JSON:

```json
{
  "host": "127.0.0.1",
  "model_id": "gemma-4-e2b",
  "max_tokens": 512,
  "trust_remote_code": false,
  "advanced": {},
  "ops": {
    "lifecycle_mode": "always_on",
    "idle_minutes": 30,
    "auto_restart": false,
    "auto_restart_max_retries": 3
  }
}
```

| Section | Edited in UI | Purpose |
|---------|--------------|---------|
| Top-level fields (`host`, `model_id`, …) | Form inputs | Defaults for every request on this instance |
| `advanced` | Advanced (JSON) textarea | Backend-specific flags → MLX CLI arguments |
| `ops` | Lifecycle / auto-restart fields | Wake, idle offload, health recovery (not forwarded to mlx-lm / mlx-vlm) |

Unknown keys inside `advanced` are **rejected** at save time for the selected launch mode.

---

## Standard form fields (not in Advanced JSON)

These are **not** part of `advanced`; use the main form.

### All launch modes

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `host` | `0.0.0.0` \| `127.0.0.1` | `MLX_DEFAULT_SERVER_HOST` | Bind address. Use `127.0.0.1` when only the gateway on the same machine should connect. |
| `model_id` | string | model folder name | **Gateway alias** — value clients send in the OpenAI `model` field (TEXT instances are rewritten to `default_model` upstream). |

### TEXT & MULTIMODAL

| Field | Type | Description |
|-------|------|-------------|
| `max_tokens` | int (1–131072) | Server default when the client omits `max_tokens`. |
| `trust_remote_code` | bool | Pass `--trust-remote-code` to mlx-lm / mlx-vlm (custom tokenizers / architectures). |

### MULTIMODAL only

| Field | Type | Description |
|-------|------|-------------|
| `max_kv_size` | int (512–1000000) | `--max-kv-size` — caps KV cache size in tokens (large VLMs, long context). |

### RERANKER

| Field | Type | Description |
|-------|------|-------------|
| `disable_batching` | bool | `--disable-batching` for local-reranker debugging. |

### IMAGE

| Field | Type | Description |
|-------|------|-------------|
| `default_quality` | `fast` \| `balanced` \| `quality` | Default when the API omits `quality`. |

### TTS (Kokoro)

| Field | Type | Description |
|-------|------|-------------|
| `voice_id` | string | Default voice when the client omits `voice`. |
| `speaking_rate` | float (0.25–4.0) | Speed multiplier. |
| `lang_code` | string | Kokoro phonetic language code (e.g. `a`, `f`). |

### STT (Whisper)

| Field | Type | Description |
|-------|------|-------------|
| `language` | string | ISO code (`en`, `fr`, …). Empty = auto-detect. |
| `chunk_duration` | float (1–120) | Audio window in seconds. |

### Lifecycle (`ops`)

| Field | Values | Description |
|-------|--------|-------------|
| `lifecycle_mode` | `always_on` \| `on_demand` | `on_demand` stops the process after `idle_minutes` without gateway traffic; next request triggers wake. |
| `idle_minutes` | 5–1440 | Idle threshold when `lifecycle_mode` is `on_demand`. |
| `auto_restart` | bool | Relaunch when health checks report DOWN. |
| `auto_restart_max_retries` | 1–20 | Cap before a 1-hour backoff. |

See [instance-lifecycle.md](instance-lifecycle.md) for gateway wake behaviour.

---

## Advanced JSON — TEXT (`mlx-lm`)

Allowed keys: `adapter_path`, `draft_model`, `num_draft_tokens`, `chat_template_args`, `temp`, `top_p`, `top_k`, `min_p`.

Each key maps to a `--kebab-case` CLI flag on `mlx_lm.server` at instance start.

| Key | CLI flag | Type | Description |
|-----|----------|------|-------------|
| `adapter_path` | `--adapter-path` | string | Path to LoRA / adapter weights. |
| `draft_model` | `--draft-model` | string | Speculative decoding draft model path or HF id. |
| `num_draft_tokens` | `--num-draft-tokens` | int | Draft tokens per speculative step. |
| `chat_template_args` | `--chat-template-args` | object | JSON object passed to the tokenizer `apply_chat_template` (serialized as JSON string). Use for template toggles, e.g. `{"enable_thinking": false}` on thinking-capable text models. |
| `temp` | `--temp` | float | Default sampling temperature (mlx-lm default 0.0). |
| `top_p` | `--top-p` | float | Nucleus sampling (default 1.0). |
| `top_k` | `--top-k` | int | Top-k sampling (0 = disabled). |
| `min_p` | `--min-p` | float | Min-p sampling (0 = disabled). |

!!! tip "Thinking on TEXT models"
    There is **no** `enable_thinking` key for TEXT in Nadir. Use `chat_template_args` when the mlx-lm chat template supports it, or rely on model-native thinking tokens in the stream (`delta.reasoning`).

**Example — Qwen with adapter and sampling:**

```json
{
  "adapter_path": "/path/to/lora",
  "temp": 0.7,
  "top_p": 0.9,
  "top_k": 40
}
```

**Example — disable thinking in chat template:**

```json
{
  "chat_template_args": {
    "enable_thinking": false
  }
}
```

---

## Advanced JSON — MULTIMODAL (`mlx-vlm`)

Allowed keys: `adapter_path`, `draft_model`, `draft_kind`, `draft_block_size`, `kv_bits`, `kv_quant_scheme`, `kv_group_size`, `enable_thinking`, `thinking_budget`.

| Key | CLI flag | Type | Description |
|-----|----------|------|-------------|
| `adapter_path` | `--adapter-path` | string | Adapter weights for the VLM. |
| `draft_model` | `--draft-model` | string | Speculative drafter (e.g. DFlash, Gemma assistant). |
| `draft_kind` | `--draft-kind` | `dflash` \| `eagle3` \| `mtp` | Drafter family; auto-detected if omitted. |
| `draft_block_size` | `--draft-block-size` | int | Override drafter block size. |
| `kv_bits` | `--kv-bits` | number | KV cache quantization bit width (e.g. `3.5` for TurboQuant). |
| `kv_quant_scheme` | `--kv-quant-scheme` | `uniform` \| `turboquant` | KV quantization backend. |
| `kv_group_size` | `--kv-group-size` | int | Group size for uniform KV quant. |
| `enable_thinking` | `--enable-thinking` | bool | Default thinking mode when the client does not set `enable_thinking`. Streams reasoning in `delta.reasoning`. |
| `thinking_budget` | `--thinking-budget` | int | Max tokens inside a thinking block (client can override per request). |

!!! warning "Agents and streaming clients"
    With `enable_thinking: true`, most output may land in `reasoning` instead of `content`. Ensure your client aggregates `reasoning` / `reasoning_content`, not only `content`.

**Example — Gemma 4 VLM with thinking:**

```json
{
  "enable_thinking": true,
  "thinking_budget": 8192
}
```

**Example — Gemma 4 MTP (E4B / E2B, bf16 assistant only):**

```json
{
  "draft_model": "/path/to/models/gemma-4-E4B-it-assistant-bf16",
  "draft_kind": "mtp",
  "draft_block_size": 6
}
```

!!! warning "MTP + quantized assistants"
    For E2B/E4B assistants with `use_ordered_embeddings`, mlx-vlm 0.6.x MTP requires an **unquantized** drafter (`*-assistant-bf16`). Pairing a QAT/4bit target with a QAT/4bit assistant (e.g. `gemma-4-E4B-it-qat-assistant-4bit`) loads successfully but **crashes at first generation** with a `MaskedEmbedder` reshape error. Keep the target quantized if needed; only the `draft_model` must be bf16.

    See [Gemma 4 MTP runbook](gateway-runbooks/gemma4-mtp.md), [compatibility matrix](draft-mtp-compatibility-matrix.md), and [ADR 007](../../adr/007-mtp-quantization.md).

**Example — KV quant for long context:**

```json
{
  "kv_bits": 4,
  "kv_quant_scheme": "uniform",
  "kv_group_size": 64
}
```

---

## Advanced JSON — IMAGE

Allowed keys: `quantize_override`.

| Key | Type | Description |
|-----|------|-------------|
| `quantize_override` | int | Intended override for mflux quantize bits when auto-detection from the folder name is wrong. |

!!! warning "Not wired yet"
    `quantize_override` is accepted in `advanced` JSON validation but is **not** passed to the image launcher today. Quantization is inferred from the model folder name (`image_model_profiles.py`). Track wiring before relying on this key.

---

## Advanced JSON — TTS

Allowed keys: `response_format`.

| Key | Type | Description |
|-----|------|-------------|
| `response_format` | string | Intended default audio codec (`wav`, `mp3`, `opus`, `aac`, …). |

!!! warning "Not wired yet"
    `response_format` in `advanced` is validated but **not** applied at server launch. Clients should send `response_format` per `POST /v1/audio/speech` request, or use Kokoro defaults.

---

## Advanced JSON — EMBEDDING, RERANKER, STT

No advanced keys are whitelisted. Use an empty object `{}` or omit `advanced`.

---

## Validation rules

- `advanced` must be a JSON **object** (not an array or string).
- Keys must match the whitelist for the instance `launch_mode`.
- `null` values are stripped on save.
- Boolean `true` for flags like `enable_thinking` becomes `--enable-thinking` with no value.
- Object values (e.g. `chat_template_args`) are JSON-serialized into a single CLI argument.

## Restart checklist

1. Edit configuration → Save.
2. **Stop** the instance (wait until status is `STOPPED`).
3. **Start** again.
4. Confirm **Running** and probe via gateway: `GET /v1/models` or a smoke `POST /v1/chat/completions`.

On-demand instances that were offloaded wake on the next gateway request with the updated config.

## Related docs

- [instance-lifecycle.md](instance-lifecycle.md) — wake, idle offload, `ops` fields
- [chat-tools-model-matrix.md](chat-tools-model-matrix.md) — tools / structured output per model family
- [vlm-vision-model-matrix.md](vlm-vision-model-matrix.md) — vision inputs for MULTIMODAL
- [nadir-gateway.md](nadir-gateway.md) — gateway aliases and client setup

## Source of truth (code)

| File | Role |
|------|------|
| `orchestrator/server_config_schema.py` | Field specs, `ADVANCED_WHITELIST`, validation |
| `orchestrator/server_manager.py` | `_build_launch_command`, CLI mapping |
| `orchestrator/templates/orchestrator/servers.html` | Advanced JSON UI and allowed-keys hint |
