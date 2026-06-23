"""Global Hugging Face tokenizer compatibility hooks for Nadir inference processes."""

from __future__ import annotations

from typing import Any


def install_auto_fix_mistral_regex() -> None:
    """Default ``fix_mistral_regex=True`` when loading tokenizers.

    Transformers detects incorrect Mistral-style ``tokenizer.json`` regex patterns
    on many community checkpoints (Qwen, Mistral-derived, etc.) and warns unless
    ``fix_mistral_regex=True`` is passed to ``AutoTokenizer.from_pretrained``.

    Installed once per inference process at launcher startup — no per-model UI config.
    """
    from transformers import AutoTokenizer

    if getattr(AutoTokenizer, "_nadir_fix_mistral_regex_installed", False):
        return

    original = AutoTokenizer.__dict__["from_pretrained"].__func__

    def from_pretrained_with_mistral_fix(
        cls: type[Any],
        pretrained_model_name_or_path: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        kwargs.setdefault("fix_mistral_regex", True)
        return original(cls, pretrained_model_name_or_path, *args, **kwargs)

    AutoTokenizer.from_pretrained = classmethod(from_pretrained_with_mistral_fix)  # type: ignore[method-assign]
    AutoTokenizer._nadir_fix_mistral_regex_installed = True
