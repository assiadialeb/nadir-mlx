"""Tests for global tokenizer compatibility hooks."""

from __future__ import annotations

from orchestrator.tokenizer_compat import install_auto_fix_mistral_regex


def test_install_auto_fix_mistral_regex_defaults_flag(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeAutoTokenizer:
        _nadir_fix_mistral_regex_installed = False

        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> str:
            captured["path"] = path
            captured["kwargs"] = dict(kwargs)
            return "tokenizer"

    monkeypatch.setattr("transformers.AutoTokenizer", FakeAutoTokenizer)

    install_auto_fix_mistral_regex()
    result = FakeAutoTokenizer.from_pretrained("/models/example")

    assert result == "tokenizer"
    assert captured["path"] == "/models/example"
    assert captured["kwargs"] == {"fix_mistral_regex": True}


def test_install_auto_fix_mistral_regex_respects_explicit_false(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeAutoTokenizer:
        _nadir_fix_mistral_regex_installed = False

        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> str:
            captured["kwargs"] = dict(kwargs)
            return "tokenizer"

    monkeypatch.setattr("transformers.AutoTokenizer", FakeAutoTokenizer)

    install_auto_fix_mistral_regex()
    FakeAutoTokenizer.from_pretrained("/models/example", fix_mistral_regex=False)

    assert captured["kwargs"] == {"fix_mistral_regex": False}


def test_install_auto_fix_mistral_regex_is_idempotent(monkeypatch) -> None:
    call_count = {"value": 0}

    class FakeAutoTokenizer:
        _nadir_fix_mistral_regex_installed = False

        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> str:
            call_count["value"] += 1
            return "tokenizer"

    monkeypatch.setattr("transformers.AutoTokenizer", FakeAutoTokenizer)

    install_auto_fix_mistral_regex()
    install_auto_fix_mistral_regex()
    FakeAutoTokenizer.from_pretrained("/models/example")

    assert call_count["value"] == 1
