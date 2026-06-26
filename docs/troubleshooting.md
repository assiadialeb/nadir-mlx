# Troubleshooting

Common operator issues on macOS.

## Port still in use after stop

The stop flow waits up to `MLX_STOP_PORT_RELEASE_TIMEOUT_SECONDS` (default **12**) for the kernel to release the port.

```bash
lsof -nP -iTCP:<PORT> -sTCP:LISTEN
kill -9 <PID>
```

The UI verifies port release before marking an instance as stopped. During a manual stop, the health watchdog pauses for that instance to avoid races with auto-restart.

## Gateway returns 404 / Not Found

| Symptom | Fix |
|---------|-----|
| `{"detail":"Not Found"}` on `/v1/chat/completions` | Restart gateway: `python manage.py run_gateway` |
| `404 model_not_found` | Check gateway alias in UI; `curl …/v1/models` |
| `400 unsupported_endpoint` | Use the route matching launch mode (embeddings vs chat, etc.) |

See [Nadir Gateway — Troubleshooting](usage/nadir-gateway.md#troubleshooting).

## Gateway returns 503

| Code | When | Action |
|------|------|--------|
| `model_unavailable` | `always_on` instance stopped or failed | Start server in UI; wait for **Running** |
| `model_waking_timeout` | `on_demand` cold start too slow | Increase `NADIR_GATEWAY_WAKE_TIMEOUT_SECONDS` and client timeout |
| Immediate 503 on stopped alias | Wrong lifecycle | Set `on_demand` if you expect wake-on-request |

See [Instance lifecycle](usage/instance-lifecycle.md).

## Model download stuck

Check `./logs/` and the Downloads section on the dashboard. Incomplete folders in `./models/` are detected automatically on restart.

## Reranker fails to load

- `jina-reranker-v3-mlx` style models → need separate `projector.safetensors`
- `JinaForRanking` mxfp4 models → handled by `reranker_server.py` (projector from bundled weights)

See [Reranker runbook](usage/gateway-runbooks/reranker.md).

## STT: ffmpeg or format errors

WAV and MP3 work in memory. M4A, FLAC, OGG, Opus, and WebM require **ffmpeg** on the host:

```bash
brew install ffmpeg
```

## Quality benchmark failures

- Install `requirements-quality.txt` for industry presets
- Ensure the target instance stays **Running** for the full job
- See [Quality benchmarks](usage/quality-benchmarks.md)

## Django / database

| Symptom | Check |
|---------|-------|
| `No module named 'django'` | `pip install -r requirements.txt` |
| PostgreSQL connection refused | `NADIR_DB_*` values; server running on configured port |
| CSRF error in production | `DJANGO_CSRF_TRUSTED_ORIGINS` with full URL |

## Getting help

Open an issue on [GitHub](https://github.com/assiadialeb/nadir-mlx/issues) with logs from `./logs/`, gateway response body, and launch mode.
