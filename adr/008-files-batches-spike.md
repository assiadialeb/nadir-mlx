# ADR 008: OpenAI `/v1/files` + `/v1/batches` — local overnight batch (spike)

**Date:** 2026-11-04  
**Status:** Accepted (conditional go — phased MVP; not in current sprint delivery)  
**Ticket:** MLX-117

## Context

OpenAI exposes:

| Surface | Role |
|---------|------|
| **`/v1/files`** | Upload / list / download / delete opaque file objects (`purpose`: `batch`, `fine-tune`, `assistants`, …) |
| **`/v1/batches`** | Async jobs: JSONL input file → many `/v1/chat/completions` (or other) calls → JSONL output file |

Nadir Gateway already covers interactive inference (`chat`, embeddings, image, TTS, STT, façades). It does **not** persist request queues or multi-hour jobs. Operators on a Mac Studio still want **privacy-first overnight runs** (eval suites, bulk rewrite, embedding backfills) without keeping a laptop client open or sending JSONL to the cloud.

This spike answers: *Should Nadir implement a local Files + Batches surface, and at what cost?*

## Current building blocks

| Asset | Relevance |
|-------|-----------|
| Gateway FastAPI `:11380` | Natural place for `/v1/files` and `/v1/batches` routes |
| SQLite via Django ORM | Persist `FileObject` / `BatchJob` metadata |
| `data/generated_images/` pattern | Precedent for UUID-keyed local blobs + TTL |
| Upstream concurrency queue | Cap parallel batch lines (`NADIR_GATEWAY_MAX_CONCURRENT_UPSTREAM`) |
| `on_demand` wake + LRU | Batches must tolerate cold starts and long wall time |
| LiteLLM (Team LAN) | Can front keys/metrics; **not** a substitute for local overnight batch on loopback |

## Options considered

| Option | Description | Verdict |
|--------|-------------|---------|
| **A — Full OpenAI Files + Batches parity** | All `purpose` values, Assistants/vector stores, fine-tune uploads, 24h cloud SLAs, every batchable endpoint | **No-go** — large surface, little local value (fine-tune/assistants unused on Nadir) |
| **B — Thin local MVP** | `purpose=batch` files only; batches for `POST /v1/chat/completions` (+ optional embeddings); JSONL in/out; SQLite status; background worker | **Conditional go** |
| **C — External only** | Document “use a script / LiteLLM / cron curl”; no Nadir API | **Fallback** if we decline product ownership |
| **D — Control-plane UI jobs only** | Django “batch run” without OpenAI shapes | Rejected for MLX-117 goal (SDK/client compatibility) |

## Feasibility (Option B)

### Files (`purpose=batch`)

**Feasible with low risk.**

- Store blobs under `data/files/<uuid>` (or content-addressed), metadata in ORM (`id`, `filename`, `bytes`, `purpose`, `created_at`).
- Routes: `POST /v1/files`, `GET /v1/files`, `GET /v1/files/{id}`, `GET /v1/files/{id}/content`, `DELETE /v1/files/{id}`.
- Reuse gateway auth (`NADIR_GATEWAY_API_KEY`), body size limits, Trusted Host / Origin guards.
- Reject non-`batch` purposes with `400` (explicit no-go for fine-tune/assistants in v1).

### Batches

**Feasible with medium engineering cost.**

Minimum viable semantics:

1. `POST /v1/batches` with `input_file_id`, `endpoint` ∈ {`/v1/chat/completions`}, `completion_window` ignored or accepted as soft hint.
2. Worker process (management command or gateway background task group) reads JSONL lines (`custom_id` + request body).
3. Each line → internal call through existing gateway routing (alias wake, queue, proxy).
4. Write output JSONL (`id`, `custom_id`, `response` / `error`) to a result file; set `status`: `validating` → `in_progress` → `completed` | `failed` | `cancelled`.
5. `GET /v1/batches/{id}`, list, cancel (best-effort stop after current line).

Hard requirements for correctness on Apple Silicon:

- **Persistence**: job state must survive gateway restart (ORM + file paths), not only in-memory.
- **Single-writer worker**: one batch runner per host (lock file or DB lease) to avoid double-processing.
- **Back-pressure**: respect concurrency env / per-instance caps; default low concurrency overnight.
- **Timeouts**: per-line timeout ≠ interactive `NADIR_GATEWAY_PROXY_TIMEOUT_SECONDS` only — document separate `NADIR_BATCH_LINE_TIMEOUT_SECONDS`.
- **Partial completion**: on cancel/crash, mark completed lines in output; allow resume **or** fail closed (choose fail-closed for v1 simplicity).

Out of scope for MVP:

- Batching `/v1/images/generations`, TTS, STT, Anthropic/Ollama façades
- OpenAI `completion_window` enforcement / billing metadata
- Multi-Mac distributed batch
- Webhook `POST` on completion (nice-to-have later)

## Effort estimate

| Slice | Rough effort |
|-------|----------------|
| Files CRUD + disk store + tests | **1–2 days** |
| Batch model + create/get/list/cancel API | **1–2 days** |
| Worker + JSONL execute via gateway + tests | **2–3 days** |
| Docs, runbook, coverage matrix, ops knobs | **1 day** |
| **MVP total** | **~5–8 engineer-days** |
| Resume-after-crash + webhooks + more endpoints | **+3–5 days** (phase 2) |

## Decision

1. **Product**: **Conditional go** on Option B — local `purpose=batch` Files + Batches for **chat completions** only.
2. **Timing**: **Not** a current-sprint delivery item after façades/security/MCP. Schedule as a dedicated story when overnight bulk jobs are a concrete customer/operator need (or when Spectra/GitPulse ask for private eval batches).
3. **Until then**: operators use scripts/`xargs`/cron against `:11380`, or LiteLLM for multi-key LAN — documented as the interim path.
4. **No-go** for Assistants / fine-tune file purposes and full cloud Batch parity.

## Consequences

- Coverage matrix should mark `/v1/files` and `/v1/batches` as **spike / deferred (ADR 008)**, not “missing by accident”.
- Implementing MVP implies a **long-lived worker** next to gateway (or gateway lifespan task) — ops docs must list a third process unless folded into `run_gateway`.
- Privacy story stays intact: JSONL never leaves the Mac.
- Risk: under-used API surface if no internal consumer; mitigate by waiting for a named use case before coding.

## Go / no-go summary

| Question | Answer |
|----------|--------|
| Technically feasible on Nadir? | **Yes** (thin MVP) |
| Worth building now? | **No** — defer until a concrete overnight-batch consumer |
| Recommended next ticket when unblocked | `feat(gateway): purpose=batch files + chat batches MVP` (~1 sprint slice) |
| Interim workaround | Shell/cron against gateway; optional LiteLLM for team auth |

## Implementation notes (when unblocked)

Design constraints to bake into the first MVP ticket (review follow-up):

1. **Upload size** — gateway `MaxBodySizeMiddleware` (50 MiB Content-Length) is too small for OpenAI’s ~200 MiB batch files; use a dedicated streaming multipart path for `POST /v1/files` that writes to disk without buffering the whole body.
2. **Content-Type** — `/v1/batches` stays JSON; `/v1/files` is `multipart/form-data` and must not go through `parse_json_body` (Origin guard still applies for browsers).
3. **Durable worker** — ORM status machine + JSONL output/error files; resume via startup reconcile; not an in-process asyncio-only queue.
4. **Interactive fairness** — share concurrency slots with lower priority; optional `NADIR_BATCH_WINDOW=22:00-07:00`.
5. **Wake / eviction** — group JSONL lines by model; wake one model at a time; rely on activity heartbeats so LRU does not thrash.
6. **Throughput** — batch fan-out should align with mlx-lm decode concurrency once that knobs surface is exposed (performance track).

## References

- OpenAI Batch API (JSONL `custom_id`, `/v1/chat/completions`)
- OpenAI Files API (`purpose=batch`)
- Local precedents: `orchestrator/gateway/routes/image_files.py`, ADR 002 (spike no-go pattern)
- MLX-117
