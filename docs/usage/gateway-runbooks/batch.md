# Runbook — Gateway batch (Files + Batches)

End-to-end validation of local OpenAI **`/v1/files`** and **`/v1/batches`** through Nadir Gateway (`:11380`). Batch jobs fan out to **`POST /v1/chat/completions`** with `X-Nadir-Priority: batch`.

!!! note "Scope (v1)"
    - **`purpose=batch`** files only (no fine-tune / assistants uploads).
    - **`endpoint`**: `/v1/chat/completions` only.
    - No resume/checkpoint after partial crash — re-submit input JSONL (orphan `in_progress` jobs are reconciled to `failed` on gateway restart).

## Prerequisites

- Django + gateway: `python manage.py serve` (recommended) or `run_gateway` in a second terminal.
- At least one **TEXT** instance **Running** (or **`on_demand`** — the worker wakes aliases per line).
- Known **gateway alias** matching `body.model` in every JSONL line.
- `NADIR_GATEWAY_BATCH_WORKER_ENABLED=true` (default) unless you plan to run `process_gateway_batches` manually.
- Optional: `Authorization: Bearer …` when `NADIR_GATEWAY_API_KEY` is set.

## 1. Prepare input JSONL

Each line is one OpenAI Batch request (UTF-8 JSONL):

```json
{"custom_id":"eval-1","method":"POST","url":"/v1/chat/completions","body":{"model":"<alias>","messages":[{"role":"user","content":"Reply with exactly: ok"}],"max_tokens":16}}
```

Save as `batch-input.jsonl` (one object per line, trailing newline). Use your real gateway **alias** in `body.model`.

!!! tip "Minimal line shape"
    Tests and the worker also accept lines **without** `method` when `url` and `body` are present (same as OpenAI batch export).

## 2. Upload file (streaming)

```bash
curl -sS http://127.0.0.1:11380/v1/files \
  -H "Authorization: Bearer sk-local" \
  -F purpose=batch \
  -F file=@batch-input.jsonl | python3 -m json.tool
```

**Expected:** HTTP 200, `"object": "file"`, `"purpose": "batch"`, `"id": "file-…"`. Body is written to disk under `NADIR_BATCH_FILES_DIR` (default `data/batch_files`), not held fully in RAM.

**Failure:** `413` if the file exceeds `NADIR_BATCH_FILE_MAX_MB` (default 200).

## 3. Create batch job

```bash
curl -sS http://127.0.0.1:11380/v1/batches \
  -H "Authorization: Bearer sk-local" \
  -H "Content-Type: application/json" \
  -d '{
    "input_file_id": "file-…",
    "endpoint": "/v1/chat/completions",
    "completion_window": "24h"
  }' | python3 -m json.tool
```

**Expected:** HTTP 200, `"object": "batch"`, `"status": "validating"`, `"id": "batch_…"`.

## 4. Wait for worker + poll status

With `serve` / `run_gateway`, the embedded worker polls every `NADIR_GATEWAY_BATCH_POLL_SECONDS` (default 15s) and processes one validating job per cycle (cross-process lock).

```bash
curl -sS http://127.0.0.1:11380/v1/batches/batch_… \
  -H "Authorization: Bearer sk-local" | python3 -m json.tool
```

**Expected progression:**

| Field | While running | When done |
|-------|----------------|-----------|
| `status` | `in_progress` | `completed` (or `failed` / `cancelled`) |
| `request_counts` | `total`, `completed`, `failed` increase | `completed + failed == total` |
| `output_file_id` | set early | same |
| `error_file_id` | set if any line errors | optional |

Output JSONL is **appended line-by-line** during `in_progress` (safe to download partial results for long jobs).

## 5. Download results

```bash
curl -sS "http://127.0.0.1:11380/v1/files/<output_file_id>/content" \
  -H "Authorization: Bearer sk-local"
```

**Expected per line (success):** OpenAI batch result object with:

- `"custom_id"` matching input
- `"id"` starting with `batch_req_`
- `"response"` wrapping HTTP 200 chat completion body

If some lines fail, fetch **`error_file_id`** content the same way — failed lines appear in the error JSONL; job-level `errors` uses OpenAI **list** shape on `failed` jobs.

## 6. Cancel (optional)

```bash
curl -sS -X POST http://127.0.0.1:11380/v1/batches/batch_…/cancel \
  -H "Authorization: Bearer sk-local" | python3 -m json.tool
```

**Expected:** `validating` → `cancelled` immediately. `in_progress` → `cancelling`, then `cancelled` after in-flight lines finish or hit `NADIR_BATCH_LINE_TIMEOUT_SECONDS` (default 600s). Lines already written to output/error files are kept.

## 7. Manual worker (debug)

When the auto worker is disabled or you need a one-shot drain:

```bash
python manage.py process_gateway_batches --max-jobs 1
```

After a gateway crash with stuck `in_progress` jobs:

```bash
python manage.py process_gateway_batches --reconcile-stale
```

The auto worker also reconciles stale `in_progress` jobs on the **first** cycle after `run_gateway` starts.

## 8. Re-use same input file

Create a second batch with the same `input_file_id`. Retention purge (`NADIR_BATCH_RETENTION_DAYS`) deletes **terminal jobs** first and removes file blobs only when **no** job references them — shared inputs remain valid for newer batches.

## 9. List and cleanup

```bash
curl -sS http://127.0.0.1:11380/v1/batches?limit=20 -H "Authorization: Bearer sk-local"
curl -sS http://127.0.0.1:11380/v1/files?limit=20 -H "Authorization: Bearer sk-local"
```

Delete an unreferenced file:

```bash
curl -sS -X DELETE http://127.0.0.1:11380/v1/files/file-… \
  -H "Authorization: Bearer sk-local"
```

**Expected:** `400` if a batch job still references the file.

## Environment knobs

| Variable | Default | Role |
|----------|---------|------|
| `NADIR_GATEWAY_BATCH_WORKER_ENABLED` | `true` | Background worker in gateway |
| `NADIR_GATEWAY_BATCH_POLL_SECONDS` | `15` | Poll interval |
| `NADIR_BATCH_LINE_TIMEOUT_SECONDS` | `600` | Per-line upstream timeout |
| `NADIR_BATCH_FILE_MAX_MB` | `200` | Upload cap |
| `NADIR_BATCH_RETENTION_DAYS` | `30` | Purge old terminal jobs |
| `NADIR_GATEWAY_MAX_CONCURRENT_GLOBAL` | `8` | Shared with interactive traffic |
| `NADIR_GATEWAY_BATCH_MAX_SHARE` | `0.5` | Max global slots for batch priority |

See [Configuration — OpenAI batch API](../../configuration.md#openai-batch-api-gateway).

## Troubleshooting

| Symptom | Cause | Action |
|---------|--------|--------|
| Stays `validating` | Worker off or gateway not running | Enable worker; use `serve` / `run_gateway` or manual `process_gateway_batches` |
| `failed` — input missing | Deleted blob or bad `input_file_id` | Re-upload JSONL; do not delete input while jobs reference it |
| Stuck `in_progress` after kill | Orphan job | Restart gateway or `--reconcile-stale`; re-submit batch |
| All lines `503` | Alias stopped (`always_on`) | Start instance or use `on_demand` |
| Slow under chat load | Global cap + interactive priority | Tune global cap / batch share; run overnight |
| `404 model_not_found` in output lines | Wrong `body.model` | Match gateway alias exactly |

## References

- [Nadir Gateway — Local batch jobs](../nadir-gateway.md#local-batch-jobs-openai-compatible)
- [Integration recipes — Overnight bulk jobs](../integration-recipes.md#overnight-bulk-jobs-openai-batch-api)
- [API coverage matrix](../nadir-gateway-coverage-matrix.md)
- [ADR 008 — Files + Batches spike](https://github.com/assiadialeb/nadir-mlx/blob/main/adr/008-files-batches-spike.md)
