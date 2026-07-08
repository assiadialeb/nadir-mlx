"""Tests for gateway alias validation and lookup."""

from django.test import TestCase

from orchestrator.gateway_aliases import (
    find_instance_by_gateway_alias,
    instance_gateway_alias,
    instance_gateway_aliases,
    normalize_gateway_alias,
    normalize_gateway_aliases_list,
    validate_gateway_alias_format,
    validate_gateway_alias_unique,
    validate_instance_gateway_aliases,
)
from orchestrator.models import InferenceInstance
from orchestrator.server_config_schema import validate_and_normalize_server_config


class GatewayAliasTests(TestCase):
    def test_normalize_gateway_alias_strips_whitespace(self) -> None:
        self.assertEqual(normalize_gateway_alias("  llama-chat  "), "llama-chat")

    def test_validate_gateway_alias_format_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            validate_gateway_alias_format("")

    def test_validate_gateway_alias_format_rejects_invalid_characters(self) -> None:
        with self.assertRaises(ValueError):
            validate_gateway_alias_format("bad alias")

    def test_validate_gateway_alias_format_accepts_hf_style_names(self) -> None:
        validate_gateway_alias_format("org/model-name:v1")

    def test_validate_sets_model_id_from_folder_name(self) -> None:
        config = validate_and_normalize_server_config("TEXT", {}, "Llama-3-8B")
        self.assertEqual(config["model_id"], "Llama-3-8B")

    def test_validate_gateway_alias_unique_rejects_duplicate(self) -> None:
        InferenceInstance.objects.create(
            model_name="model-a",
            port=11400,
            server_config={"model_id": "shared-alias"},
            status="STOPPED",
        )
        with self.assertRaises(ValueError):
            validate_gateway_alias_unique("shared-alias")

    def test_validate_gateway_alias_unique_ignores_same_instance(self) -> None:
        instance = InferenceInstance.objects.create(
            model_name="model-a",
            port=11400,
            server_config={"model_id": "shared-alias"},
            status="STOPPED",
        )
        validate_gateway_alias_unique("shared-alias", exclude_instance_id=instance.pk)

    def test_validate_gateway_alias_unique_is_case_insensitive(self) -> None:
        InferenceInstance.objects.create(
            model_name="model-a",
            port=11400,
            server_config={"model_id": "Gemma-4B"},
            status="STOPPED",
        )
        with self.assertRaises(ValueError):
            validate_gateway_alias_unique("gemma-4b")

    def test_instance_gateway_alias_falls_back_to_folder_name(self) -> None:
        instance = InferenceInstance.objects.create(
            model_name="folder-name",
            port=11401,
            server_config={},
            status="STOPPED",
        )
        self.assertEqual(instance_gateway_alias(instance), "folder-name")

    def test_find_instance_by_gateway_alias_returns_match(self) -> None:
        instance = InferenceInstance.objects.create(
            model_name="folder-name",
            port=11402,
            server_config={"model_id": "lite-llm-name"},
            status="RUNNING",
        )
        found = find_instance_by_gateway_alias("lite-llm-name")
        self.assertEqual(found.pk, instance.pk)

    def test_instance_gateway_aliases_includes_primary_and_extras(self) -> None:
        instance = InferenceInstance.objects.create(
            model_name="folder-name",
            port=11403,
            server_config={
                "model_id": "primary-alias",
                "gateway_aliases": ["dev-alias", "prod-alias", "primary-alias"],
            },
            status="STOPPED",
        )
        self.assertEqual(
            instance_gateway_aliases(instance),
            ["primary-alias", "dev-alias", "prod-alias"],
        )

    def test_normalize_gateway_aliases_list_rejects_invalid_type(self) -> None:
        with self.assertRaises(ValueError):
            normalize_gateway_aliases_list({"bad": "type"}, primary_alias="primary")

    def test_validate_instance_gateway_aliases_rejects_cross_instance_collision(self) -> None:
        InferenceInstance.objects.create(
            model_name="model-a",
            port=11404,
            server_config={"model_id": "taken-alias"},
            status="STOPPED",
        )
        with self.assertRaises(ValueError):
            validate_instance_gateway_aliases(
                {"model_id": "new-primary", "gateway_aliases": ["taken-alias"]},
                "model-b",
            )

    def test_find_instance_by_gateway_alias_matches_extra_alias(self) -> None:
        instance = InferenceInstance.objects.create(
            model_name="folder-name",
            port=11405,
            server_config={
                "model_id": "primary-alias",
                "gateway_aliases": ["legacy-alias"],
            },
            status="RUNNING",
        )
        found = find_instance_by_gateway_alias("legacy-alias")
        self.assertEqual(found.pk, instance.pk)
