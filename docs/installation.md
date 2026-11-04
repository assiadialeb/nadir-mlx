# Installation

Step-by-step setup on Apple Silicon macOS.

## Prerequisites

| Item | Requirement |
|------|-------------|
| Hardware | Apple Silicon Mac (M1–M4) |
| OS | macOS 14+ |
| Python | **3.12.x** recommended (mflux, local-reranker, lm-eval) |
| Disk | Model-dependent — plan tens of GB per large checkpoint |
| Optional | **ffmpeg** (`brew install ffmpeg`) for STT on M4A/WebM |

## 1. Clone the repository

```bash
git clone https://github.com/assiadialeb/nadir-mlx.git
cd nadir-mlx
```

## 2. Create a virtual environment

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

## 3. Install core dependencies

Required for the dashboard, gateway, inference launchers, and **performance** benchmarks:

```bash
pip install -r requirements.txt
```

!!! note "Python 3.14 + local-reranker"
    `local-reranker` pins `Python <3.14`. If pip refuses the install:

    ```bash
    pip install -r requirements.txt --ignore-requires-python
    ```

## 4. Install quality benchmarks (optional)

Required only for **industry** quality tasks (IFEval, GSM8K via [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)). Platform quality suites run without this extra.

```bash
pip install -r requirements-quality.txt
```

| Install set | Enables |
|-------------|---------|
| `requirements.txt` only | Dashboard, gateway, perf benchmarks, platform quality suites |
| `+ requirements-quality.txt` | Industry presets (`industry_lite`: IFEval + GSM8K) |

See [Quality benchmarks](usage/quality-benchmarks.md) for presets and troubleshooting.

## 5. Configure environment

```bash
cp .env.example .env
```

Edit `.env` as needed. See [Configuration](configuration.md) for all variables.

PostgreSQL is optional — SQLite is the default when `NADIR_DB_HOST` is unset.

## 6. Start UI + gateway (one command)

```bash
python manage.py serve
```

On first run, `serve`:

- generates `DJANGO_SECRET_KEY` into `data/django_secret_key` (mode `600`) when unset
- runs `migrate` before starting children

You can still set `DJANGO_SECRET_KEY` in `.env` yourself. Create an admin when ready:

```bash
python manage.py createsuperuser
```

This supervises:

- Django UI on `DJANGO_HTTP_PORT` (default **8000**)
- Nadir Gateway (`run_gateway`, default **11380**)

Optional MCP in the same stack:

```bash
python manage.py serve --with-mcp
```

Ctrl+C stops every supervised process (cleanup runs on the main thread, then reclaim any orphaned listeners on the UI/gateway ports). Background reconcile / watchdog / idle-offload run only in the gateway child. If `serve` is killed with `SIGKILL`, children exit shortly after (parent-PID watch); the next `serve` also reclaims stale `manage.py` listeners before binding. You can still run `runserver` / `run_gateway` / `run_mcp` in separate terminals if you prefer.

Open **http://127.0.0.1:8000** and sign in with your superuser account.

Health check:

```bash
curl http://127.0.0.1:11380/health
curl http://127.0.0.1:11380/v1/models
```

Keep `NADIR_GATEWAY_HOST=127.0.0.1` unless you intentionally expose the LAN. Off-loopback binds require `NADIR_GATEWAY_API_KEY` — see [Nadir Gateway — Security](usage/nadir-gateway.md#security).

Full operator guide: [Nadir Gateway](usage/nadir-gateway.md). Client recipes: [Integration recipes](usage/integration-recipes.md).

Use **`http://127.0.0.1:11380/v1`** as `api_base` in your client. Pass the **gateway alias** (shown on each server card) in the `model` field.

## 7. Download and launch a model

1. Go to **Search** and find a model (e.g. `mlx-community/Qwen2.5-7B-Instruct-4bit`)
2. Click **Download** — files land in `./models/<model-name>/`
3. On the **Dashboard**, pick a launch mode and click **Start**
4. Point your client at the gateway (`:11380/v1`) or the instance port for debugging

## Verify installation

| Check | Expected |
|-------|----------|
| `curl http://127.0.0.1:8000/` | Login page or redirect |
| `curl http://127.0.0.1:11380/health` | `{"status":"ok"}` or similar |
| Start a TEXT instance | Status **Running** in UI |
| `GET /v1/models` on gateway | Alias listed when instance is running |

## Next steps

- [Configuration](configuration.md) — environment variables
- [Instance lifecycle](usage/instance-lifecycle.md) — `always_on` vs `on_demand`
- [Gateway runbooks](usage/gateway-runbooks/chat.md) — E2E validation per mode
