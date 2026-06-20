"""OpenAI-compatible image generation server powered by mflux."""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import threading
import time
from pathlib import Path
from typing import Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from orchestrator.image_model_loader import (
    generate_image_bytes,
    load_image_model,
    resolve_image_profile,
)
from orchestrator.image_model_profiles import ImageModelProfile

app = FastAPI(title="MLX Image Server")
_state: dict[str, object] = {}
_generation_lock = threading.Lock()


class ImageGenerationRequest(BaseModel):
    prompt: str
    model: str = "default_model"
    n: int = Field(default=1, ge=1, le=4)
    size: Optional[str] = None
    response_format: Literal["url", "b64_json"] = "b64_json"
    quality: Literal["fast", "balanced", "quality"] = "balanced"
    user: Optional[str] = None
    seed: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance: Optional[float] = None
    negative_prompt: Optional[str] = None


def _get_profile() -> ImageModelProfile:
    profile = _state.get("profile")
    if profile is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return profile  # type: ignore[return-value]


def _parse_size(size: str | None, profile: ImageModelProfile) -> tuple[int, int]:
    if not size:
        return profile.default_width, profile.default_height

    try:
        width_str, height_str = size.lower().split("x", maxsplit=1)
        width, height = int(width_str), int(height_str)
    except (ValueError, AttributeError) as exc:
        raise HTTPException(status_code=400, detail="size must be WIDTHxHEIGHT.") from exc

    if width < 256 or height < 256 or width > 2048 or height > 2048:
        raise HTTPException(status_code=400, detail="size must be between 256 and 2048.")
    return width, height


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "model": str(_state.get("model_id", ""))}


@app.get("/v1/image/defaults")
def image_defaults() -> dict[str, object]:
    """Return recommended inference parameters for the loaded checkpoint."""
    profile = _get_profile()
    return {"object": "image_generation_defaults", **profile.as_api_dict()}


@app.get("/v1/models")
def list_models() -> dict[str, object]:
    profile = _state.get("profile")
    metadata: dict[str, object] = {}
    if isinstance(profile, ImageModelProfile):
        metadata["image_defaults"] = profile.as_api_dict()

    return {
        "object": "list",
        "data": [
            {
                "id": _state.get("model_id", "default_model"),
                "object": "model",
                "created": int(_state.get("created", time.time())),
                **metadata,
            }
        ],
    }


@app.post("/v1/images/generations")
def create_images(body: ImageGenerationRequest) -> dict[str, object]:
    model = _state.get("model")
    profile = _get_profile()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if not body.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt must not be empty.")
    if body.response_format == "url":
        raise HTTPException(
            status_code=400,
            detail="url response_format is not supported; use b64_json.",
        )

    width, height = _parse_size(body.size, profile)
    steps = profile.resolve_steps(body.quality, body.num_inference_steps)
    guidance = profile.default_guidance if body.guidance is None else body.guidance

    data: list[dict[str, str]] = []
    with _generation_lock:
        print(
            f"[image pid={os.getpid()}] generation started: quality={body.quality} "
            f"steps={steps} size={width}x{height} n={body.n}",
            flush=True,
        )
        try:
            for index in range(body.n):
                seed = body.seed if body.seed is not None else random.randint(0, 2**31 - 1)
                png_bytes = generate_image_bytes(
                    model,
                    profile,
                    prompt=body.prompt,
                    seed=seed,
                    num_inference_steps=steps,
                    width=width,
                    height=height,
                    guidance=guidance,
                    negative_prompt=body.negative_prompt,
                )
                data.append({"b64_json": base64.b64encode(png_bytes).decode("ascii")})
                print(f"[image] completed image {index + 1}/{body.n}", flush=True)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"created": int(time.time()), "data": data}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLX image generation server")
    parser.add_argument("--model", required=True, help="Local path to the image model")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11400)
    parser.add_argument("--model-id", default=None, help="Model ID exposed via /v1/models")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("TQDM_DISABLE", "1")
    args = _parse_args()
    model_path = Path(args.model).resolve()
    if not model_path.is_dir():
        raise SystemExit(f"Model path not found: {model_path}")

    profile = resolve_image_profile(model_path)
    print(f"Resolved image profile: {json.dumps(profile.as_api_dict(), indent=2)}")
    print(f"Loading image model ({profile.family}/{profile.config_attr}) from {model_path} ...")
    model = load_image_model(model_path, profile)

    _state["model"] = model
    _state["profile"] = profile
    _state["model_id"] = args.model_id or model_path.name
    _state["created"] = time.time()

    print(f"MLX image server PID {os.getpid()} ready on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
