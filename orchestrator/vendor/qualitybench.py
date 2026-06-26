"""Platform quality suites executed via the Nadir gateway chat API."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx

SUITE_DIR = Path(__file__).resolve().parent.parent / "data" / "quality_suites"
DEFAULT_SUITE_NAMES = ("text_platform",)


def list_suite_paths() -> list[Path]:
    """Return available suite JSON files."""
    if not SUITE_DIR.is_dir():
        return []
    return sorted(SUITE_DIR.glob("*.json"))


def load_suite(name: str) -> dict[str, Any]:
    """Load a suite definition by file stem name."""
    path = SUITE_DIR / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"Quality suite not found: {name}")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def score_response(text: str, scorer: dict[str, Any]) -> bool:
    """Apply a deterministic scorer to model output."""
    scorer_type = scorer.get("type", "")
    cleaned = text.strip()

    if scorer_type == "contains":
        return str(scorer.get("value", "")).lower() in cleaned.lower()

    if scorer_type == "regex":
        pattern = str(scorer.get("pattern", ""))
        flags = re.IGNORECASE if scorer.get("ignore_case") else 0
        return bool(re.search(pattern, cleaned, flags))

    if scorer_type == "json_schema_valid":
        return _json_schema_valid(cleaned, scorer)

    return False


def _json_schema_valid(text: str, scorer: dict[str, Any]) -> bool:
    payload = _extract_json(text)
    if payload is None:
        return False

    required_keys = scorer.get("required_keys") or []
    for key in required_keys:
        if key not in payload:
            return False

    types_map = scorer.get("types") or {}
    for key, expected_type in types_map.items():
        if key not in payload:
            return False
        if not _value_matches_type(payload[key], str(expected_type)):
            return False
    return True


def _extract_json(text: str) -> Any | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(line for line in lines if not line.strip().startswith("```")).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                return None
        start = stripped.find("[")
        end = stripped.rfind("]")
        if start >= 0 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _value_matches_type(value: Any, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    return True


def run_suite_case(
    host: str,
    port: int,
    model: str,
    case: dict[str, Any],
    *,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Execute one suite case against the chat-completions API."""
    base_url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": case["prompt"]}],
        "temperature": 0,
        "max_tokens": int(case.get("max_tokens", 128)),
    }
    response = httpx.post(base_url, json=payload, timeout=timeout_seconds)
    response.raise_for_status()
    body = response.json()
    content = (
        body.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    passed = score_response(str(content), case.get("scorer", {}))
    return {
        "id": case.get("id", ""),
        "passed": passed,
        "response_preview": str(content)[:200],
    }


def run_platform_suites(
    host: str,
    port: int,
    model: str,
    *,
    suite_names: tuple[str, ...] = DEFAULT_SUITE_NAMES,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Run all platform suites and return pass rates."""
    suites: dict[str, Any] = {}

    for suite_name in suite_names:
        suite = load_suite(suite_name)
        case_results: list[dict[str, Any]] = []
        for case in suite.get("cases", []):
            case_results.append(
                run_suite_case(
                    host,
                    port,
                    model,
                    case,
                    timeout_seconds=timeout_seconds,
                )
            )
        passed = sum(1 for item in case_results if item["passed"])
        total = len(case_results)
        suites[suite_name] = {
            "pass_rate": round((passed / total) * 100, 1) if total else 0.0,
            "passed": passed,
            "total": total,
            "cases": case_results,
        }

    return {"suites": suites, "source": "qualitybench"}


def summarize_platform_metrics(platform: dict[str, Any]) -> dict[str, float | None]:
    """Flatten suite pass rates for UI display."""
    metrics: dict[str, float | None] = {}
    for suite_name, payload in (platform.get("suites") or {}).items():
        metrics[f"{suite_name}_pass_rate"] = payload.get("pass_rate")
    return metrics
