"""Tests for gateway model discovery."""

from django.test import TestCase

from orchestrator.gateway.selectors import list_running_gateway_models
from orchestrator.models import InferenceInstance


class GatewayModelsSelectorTests(TestCase):
    def test_list_running_gateway_models_returns_running_aliases(self) -> None:
        InferenceInstance.objects.create(
            model_name="Llama-3-8B",
            port=11400,
            launch_mode="TEXT",
            server_config={"model_id": "llama-chat"},
            status="RUNNING",
        )
        InferenceInstance.objects.create(
            model_name="embed-model",
            port=11401,
            launch_mode="EMBEDDING",
            server_config={"model_id": "local-embed"},
            status="RUNNING",
        )
        InferenceInstance.objects.create(
            model_name="offline",
            port=11402,
            launch_mode="TEXT",
            server_config={"model_id": "offline"},
            status="STOPPED",
        )

        payload = list_running_gateway_models()
        self.assertEqual(payload["object"], "list")
        ids = {entry["id"] for entry in payload["data"]}
        self.assertEqual(ids, {"llama-chat", "local-embed"})
        text_entry = next(item for item in payload["data"] if item["id"] == "llama-chat")
        self.assertEqual(text_entry["owned_by"], "nadir")
        self.assertEqual(text_entry["metadata"]["launch_mode"], "TEXT")

    def test_list_running_gateway_models_uses_folder_name_without_alias(self) -> None:
        InferenceInstance.objects.create(
            model_name="gemma-4-12B-it-4bit",
            port=11403,
            launch_mode="TEXT",
            server_config={},
            status="RUNNING",
        )
        payload = list_running_gateway_models()
        self.assertEqual(payload["data"][0]["id"], "gemma-4-12B-it-4bit")
