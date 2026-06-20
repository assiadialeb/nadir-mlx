#!/usr/bin/env python3
"""Generate orchestrator/data/model_registry.json from Hugging Face top models.

Examples:
  # Top 50 mlx-community (default), merge into bundled registry
  python scripts/build_model_registry.py

  # Top 100, write to a review file first
  python scripts/build_model_registry.py --limit 100 --output /tmp/model_registry.json

  # Also parse local ./models README files
  python scripts/build_model_registry.py --include-local

  # Fetch upstream README too (slower, more HF requests)
  python scripts/build_model_registry.py --limit 30 --fetch-upstream-readme

  # Preview without writing
  python scripts/build_model_registry.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orchestrator.registry_builder import build_registry_file
from orchestrator.model_registry import REGISTRY_PATH


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build model_registry.json from Hugging Face mlx-community metadata.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of top mlx-community models to fetch (default: 50).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REGISTRY_PATH,
        help=f"Output JSON path (default: {REGISTRY_PATH}).",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Local models directory for --include-local.",
    )
    parser.add_argument(
        "--include-local",
        action="store_true",
        help="Merge entries built from local ./models README files.",
    )
    parser.add_argument(
        "--fetch-upstream-readme",
        action="store_true",
        help="Also download upstream README.md (extra network calls).",
    )
    parser.add_argument(
        "--fresh-models",
        action="store_true",
        help="Replace the models section entirely instead of merging into existing entries.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and merge but do not write the output file.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.limit < 1:
        print("Error: --limit must be >= 1.", file=sys.stderr)
        return 1

    print(
        f"Fetching top {args.limit} mlx-community models from Hugging Face...",
        flush=True,
    )
    if args.fetch_upstream_readme:
        print("Upstream README fetching enabled (slower).", flush=True)

    merged = build_registry_file(
        limit=args.limit,
        output_path=args.output,
        models_dir=args.models_dir,
        include_local=args.include_local,
        fetch_upstream_readme=args.fetch_upstream_readme,
        fresh_models=args.fresh_models,
        dry_run=args.dry_run,
    )

    model_count = len(merged.get("models") or {})
    action = "Would write" if args.dry_run else "Wrote"
    print(f"{action} {model_count} model entries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
