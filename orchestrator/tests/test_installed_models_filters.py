"""Tests for installed models filtering and sorting."""

from orchestrator.ui_selectors import (
    apply_installed_models_filters,
    build_installed_models_query,
    filter_installed_models,
    format_disk_size,
    parse_installed_models_query,
    sort_installed_models,
)


def _sample_models() -> list[dict]:
    return [
        {
            "name": "z-model",
            "disk_size_bytes": 1000,
            "supports_text": True,
            "supports_image": False,
        },
        {
            "name": "a-flux",
            "disk_size_bytes": 5000,
            "supports_text": False,
            "supports_image": True,
        },
        {
            "name": "chat-alpha",
            "disk_size_bytes": 2000,
            "supports_text": True,
            "supports_multimodal": True,
        },
    ]


def test_parse_installed_models_query_defaults() -> None:
    query, cap, sort, fits_mac = parse_installed_models_query({})
    assert query == ""
    assert cap == ""
    assert sort == "name_asc"
    assert fits_mac is False


def test_parse_installed_models_query_rejects_invalid_cap_and_sort() -> None:
    query, cap, sort, fits_mac = parse_installed_models_query({
        "q": "  llama ",
        "cap": "INVALID",
        "sort": "bad",
        "fits_mac": "1",
    })
    assert query == "llama"
    assert cap == ""
    assert sort == "name_asc"
    assert fits_mac is True


def test_filter_installed_models_by_name() -> None:
    result = filter_installed_models(_sample_models(), query="flux")
    assert [model["name"] for model in result] == ["a-flux"]


def test_filter_installed_models_by_capability() -> None:
    result = filter_installed_models(_sample_models(), capability="IMAGE")
    assert [model["name"] for model in result] == ["a-flux"]

    text_models = filter_installed_models(_sample_models(), capability="TEXT")
    assert {model["name"] for model in text_models} == {"z-model", "chat-alpha"}


def test_sort_installed_models_by_name_and_size() -> None:
    models = _sample_models()
    assert [m["name"] for m in sort_installed_models(models, "name_asc")] == [
        "a-flux",
        "chat-alpha",
        "z-model",
    ]
    assert [m["name"] for m in sort_installed_models(models, "name_desc")][0] == "z-model"
    assert [m["name"] for m in sort_installed_models(models, "size_desc")][0] == "a-flux"
    assert [m["name"] for m in sort_installed_models(models, "size_asc")][0] == "z-model"


def test_apply_installed_models_filters_combines_query_and_sort() -> None:
    result = apply_installed_models_filters(
        _sample_models(),
        query="a",
        capability="TEXT",
        sort="name_desc",
    )
    assert [model["name"] for model in result] == ["chat-alpha"]


def test_build_installed_models_query_omits_empty_values() -> None:
    assert build_installed_models_query() == "tab=installed"
    assert build_installed_models_query(query="flux", capability="IMAGE", sort="size_desc") == (
        "tab=installed&q=flux&cap=IMAGE&sort=size_desc"
    )


def test_format_disk_size_human_readable() -> None:
    assert format_disk_size(512) == "512 B"
    assert format_disk_size(2048) == "2.0 KB"
    assert format_disk_size(5 * 1024 * 1024 * 1024) == "5.0 GB"


def test_infer_hf_model_capabilities_for_flux() -> None:
    from orchestrator.ui_selectors import infer_hf_model_capabilities

    caps = infer_hf_model_capabilities(
        "Flux-1.lite-8B-MLX-Q4",
        pipeline_tag="text-to-image",
    )
    assert caps["supports_image"] is True
    assert caps["supports_text"] is False


def test_infer_hf_model_capabilities_for_kokoro() -> None:
    from orchestrator.ui_selectors import infer_hf_model_capabilities

    caps = infer_hf_model_capabilities("Kokoro-82M-6bit")
    assert caps["supports_tts"] is True


def test_parse_hf_model_from_api_includes_capabilities() -> None:
    from orchestrator.ui_selectors import parse_hf_model_from_api

    parsed = parse_hf_model_from_api(
        {
            "id": "mlx-community/jina-reranker-v3-4bit-mxfp4",
            "tags": [],
            "pipeline_tag": "text-ranking",
            "downloads": 10,
            "likes": 2,
        },
        {},
    )
    assert parsed["supports_rerank"] is True
    assert parsed["repo_id"] == "mlx-community/jina-reranker-v3-4bit-mxfp4"


def test_hf_fetch_limit_for_filters() -> None:
    from orchestrator.ui_selectors import HF_FETCH_LIMIT_FILTERED, HF_FETCH_LIMIT_DEFAULT, hf_fetch_limit_for_filters

    assert hf_fetch_limit_for_filters("", "", "name_asc") == HF_FETCH_LIMIT_DEFAULT
    assert hf_fetch_limit_for_filters("", "IMAGE", "name_asc") == HF_FETCH_LIMIT_FILTERED


def test_fetch_hf_models_requests_full_hub_expand_fields(mocker) -> None:
    from orchestrator.ui_selectors import HF_HUB_MODEL_EXPAND_FIELDS, fetch_hf_models

    mock_get = mocker.patch("orchestrator.ui_selectors.huggingface_get")
    mock_get.return_value.raise_for_status = lambda: None
    mock_get.return_value.json.return_value = []

    fetch_hf_models()

    params = mock_get.call_args.kwargs["params"]
    assert params["expand"] == list(HF_HUB_MODEL_EXPAND_FIELDS)
    assert set(params["expand"]) >= {
        "safetensors",
        "tags",
        "pipeline_tag",
        "likes",
        "downloads",
        "library_name",
    }


def test_parse_hf_model_from_api_minimal_expand_payload() -> None:
    """Contract: HF expand-only payloads still parse (capabilities from name)."""
    from orchestrator.ui_selectors import parse_hf_model_from_api

    parsed = parse_hf_model_from_api(
        {
            "id": "mlx-community/parakeet-tdt-0.6b-v3",
            "downloads": 42,
            "safetensors": {"total": 600_000_000},
        },
        {},
    )
    assert parsed["supports_stt"] is True
    assert parsed["downloads"] == 42
    assert parsed["likes"] == 0
    assert parsed["param_size_b"] == 0.6

