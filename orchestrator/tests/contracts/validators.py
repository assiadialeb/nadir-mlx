"""OpenAPI contract validation helpers for gateway pytest suites."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
CURATED_SPEC_PATH = _PROJECT_ROOT / "openapi" / "nadir-curated.yaml"


def load_curated_spec() -> dict[str, Any]:
    """Load the curated OpenAPI document from disk."""
    with CURATED_SPEC_PATH.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid OpenAPI document: {CURATED_SPEC_PATH}")
    return payload


def _resolve_ref(spec: dict[str, Any], ref: str) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise ValueError(f"Only local refs are supported: {ref}")
    node: Any = spec
    for part in ref.lstrip("#/").split("/"):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"Unresolved $ref: {ref}")
        node = node[part]
    if not isinstance(node, dict):
        raise ValueError(f"$ref does not point to a schema object: {ref}")
    return node


def _inline_refs(spec: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve local component $ref entries for jsonschema validation."""
    if "$ref" in schema:
        resolved = _resolve_ref(spec, str(schema["$ref"]))
        return _inline_refs(spec, resolved.copy())

    cloned = schema.copy()
    for key, value in schema.items():
        if key == "$ref":
            continue
        if isinstance(value, dict):
            cloned[key] = _inline_refs(spec, value)
        elif isinstance(value, list):
            cloned[key] = [
                _inline_refs(spec, item) if isinstance(item, dict) else item
                for item in value
            ]
    return cloned


def response_schema(spec: dict[str, Any], path: str, method: str, status: str) -> dict[str, Any]:
    """Return the JSON schema for a response status on a given path."""
    paths = spec.get("paths", {})
    operation = paths.get(path, {}).get(method.lower(), {})
    responses = operation.get("responses", {})
    response = responses.get(status)
    if not response:
        raise KeyError(f"No response {status} for {method.upper()} {path}")
    content = response.get("content", {}).get("application/json", {})
    schema = content.get("schema")
    if not isinstance(schema, dict):
        raise ValueError(f"Missing JSON schema for {method.upper()} {path} {status}")
    return _inline_refs(spec, schema)


def request_schema(spec: dict[str, Any], path: str, method: str) -> dict[str, Any]:
    """Return the JSON schema for a request body on a given path."""
    paths = spec.get("paths", {})
    operation = paths.get(path, {}).get(method.lower(), {})
    body = operation.get("requestBody", {})
    content = body.get("content", {}).get("application/json", {})
    schema = content.get("schema")
    if not isinstance(schema, dict):
        raise ValueError(f"Missing request schema for {method.upper()} {path}")
    return _inline_refs(spec, schema)


def validate_against_schema(instance: Any, schema: dict[str, Any]) -> None:
    """Validate JSON data against a JSON Schema; raises ValidationError on failure."""
    validator = Draft202012Validator(schema)
    validator.validate(instance)


def validate_response_body(
    spec: dict[str, Any],
    *,
    path: str,
    method: str,
    status_code: int,
    body: Any,
) -> None:
    """Validate a response body against the curated OpenAPI contract."""
    schema = response_schema(spec, path, method, str(status_code))
    validate_against_schema(body, schema)


def format_validation_error(exc: ValidationError) -> str:
    """Return a concise validation error message for test assertions."""
    location = ".".join(str(part) for part in exc.absolute_path) or "<root>"
    return f"{location}: {exc.message}"
