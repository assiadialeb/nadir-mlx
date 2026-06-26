# Quality benchmarks

Nadir supports three benchmark modes against **TEXT** or **MULTIMODAL** instances via the gateway (`:11380`):

| Mode | Description |
|------|-------------|
| **PERF** | Latency and throughput via vendored `llmbench.py` (default) |
| **QUALITY** | Industry tasks (optional `lm-eval`) + platform suites (`qualitybench`) |
| **COMPLETE** | Runs **PERF** then **QUALITY** sequentially on the same instance |

!!! note "Privacy-first"
    All evaluation traffic stays on loopback. No cloud APIs. Platform fixtures live under `orchestrator/data/quality_suites/`.

## Prerequisites

Core Nadir install is enough for **PERF** and **platform quality** suites.

For **industry** metrics (IFEval, GSM8K), install the optional extra:

```bash
pip install -r requirements-quality.txt
```

The Nadir runner passes `--apply_chat_template` to lm-eval (required for `local-chat-completions` against the gateway).

This pulls in [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) with API backends, plus `langdetect` and `immutabledict` required by IFEval.

!!! warning "Chat-completions API"
    Industry tasks use `local-chat-completions` against the Nadir gateway. Tasks that require **loglikelihood** scoring (e.g. MMLU) are excluded from `industry_lite` because the chat API does not support them.

## Launch from the UI

1. Open **Benchmark** in the Nadir dashboard.
2. Choose **Performance**, **Quality**, or **Complete**.
3. Select a **RUNNING** TEXT/MULTIMODAL instance (gateway alias is resolved automatically).
4. For quality modes, keep preset **`industry_lite`** unless you have a custom runner preset.

**Complete** creates a parent run plus two child runs (perf → quality). The parent page shows an aggregate summary and links to each phase.

## Preset `industry_lite`

| Task | Metric | Typical use |
|------|--------|-------------|
| `ifeval` | `prompt_level_strict_acc` | Instruction following |
| `gsm8k` | `exact_match` | Grade-school math |

Parameters: `temperature=0`, `num_concurrent=1`, `limit=100`, gateway chat-completions URL with `--apply_chat_template`.

If `lm_eval` is not installed, industry tasks are **skipped** with a clear reason; platform suites still run.

## Platform suites

| Suite | Cases | Scorers |
|-------|-------|---------|
| `text_platform` | 10 | `regex`, `contains`, `json_schema_valid` |

Results appear as `text_platform_pass_rate` in quality metrics.

!!! warning "VLM / OCR"
    Vision quality suites are **not** enabled while mlx-vlm HTTP serving has known thread-safety issues. Use PERF/multimodal chat benchmarks instead.

## Artifacts

| File | Content |
|------|---------|
| `logs/benchmarks/bench_<id>.json` | PERF llmbench output |
| `logs/benchmarks/bench_<id>_quality.json` | Normalized quality envelope |
| `logs/benchmarks/quality_<id>/` | Raw lm-eval output directory |

## CLI (lm-eval runner)

The wrapper mirrors llmbench invocation:

```bash
python -c "
from pathlib import Path
from orchestrator.vendor.lm_eval_runner import run_lm_eval
print(run_lm_eval('127.0.0.1', 11380, 'my-alias', Path('logs/benchmarks/manual_lm_eval')))
"
```

## Interpreting a complete run

Example summary on the parent run:

```
Perf:     aggregate_tps 38 · TTFT p50 420 ms
Quality:  gsm8k_exact_match 74% · ifeval_strict_acc 72% · text_platform_pass_rate 90%
```

Use **Benchmark history** to filter by `benchmark_kind` (PERF / QUALITY / COMPLETE).

## Related docs

- [Instance lifecycle](instance-lifecycle.md) — ensure the instance stays RUNNING for long quality jobs
- [Nadir gateway](nadir-gateway.md) — alias routing used by all benchmark modes
