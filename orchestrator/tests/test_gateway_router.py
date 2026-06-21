"""Tests for gateway alias routing."""

from django.test import TestCase

from orchestrator.gateway.router import (
    CHAT_COMPLETIONS_PATH,
    EMBEDDINGS_PATH,
    GatewayRouteError,
)
from orchestrator.gateway.selectors import resolve_gateway_target
from orchestrator.models import InferenceInstance


class GatewayRouterTests(TestCase):
    def test_resolve_gateway_target_returns_running_text_instance(self) -> None:
        instance = InferenceInstance.objects.create(
            model_name="Llama-3-8B",
            port=11400,
            launch_mode="TEXT",
            server_config={"model_id": "llama-chat", "host": "127.0.0.1"},
            status="RUNNING",
        )
        target = resolve_gateway_target("llama-chat")
        self.assertEqual(target.instance_id, instance.pk)
        self.assertEqual(target.launch_mode, "TEXT")
        self.assertEqual(target.host, "127.0.0.1")
        self.assertEqual(target.port, 11400)
        self.assertEqual(target.upstream_model, "default_model")
        self.assertEqual(target.api_path, CHAT_COMPLETIONS_PATH)
        self.assertEqual(target.upstream_url, "http://127.0.0.1:11400/v1/chat/completions")

    def test_resolve_gateway_target_maps_embedding_mode(self) -> None:
        InferenceInstance.objects.create(
            model_name="embed-model",
            port=11401,
            launch_mode="EMBEDDING",
            server_config={"model_id": "local-embed"},
            status="RUNNING",
        )
        target = resolve_gateway_target("local-embed")
        self.assertEqual(target.api_path, EMBEDDINGS_PATH)
        self.assertEqual(target.upstream_model, "local-embed")

    def test_resolve_gateway_target_unknown_alias_returns_404(self) -> None:
        with self.assertRaises(GatewayRouteError) as ctx:
            resolve_gateway_target("missing-alias")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.code, "model_not_found")

    def test_resolve_gateway_target_stopped_instance_returns_503(self) -> None:
        InferenceInstance.objects.create(
            model_name="offline-model",
            port=11402,
            launch_mode="TEXT",
            server_config={"model_id": "offline"},
            status="STOPPED",
        )
        with self.assertRaises(GatewayRouteError) as ctx:
            resolve_gateway_target("offline")
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ctx.exception.code, "model_unavailable")

    def test_resolve_gateway_target_failed_instance_returns_503(self) -> None:
        InferenceInstance.objects.create(
            model_name="broken-model",
            port=11403,
            launch_mode="TEXT",
            server_config={"model_id": "broken"},
            status="FAILED",
        )
        with self.assertRaises(GatewayRouteError) as ctx:
            resolve_gateway_target("broken")
        self.assertEqual(ctx.exception.status_code, 503)

    def test_resolve_gateway_target_loading_instance_returns_503(self) -> None:
        InferenceInstance.objects.create(
            model_name="loading-model",
            port=11404,
            launch_mode="TEXT",
            server_config={"model_id": "loading"},
            status="LOADING",
        )
        with self.assertRaises(GatewayRouteError) as ctx:
            resolve_gateway_target("loading")
        self.assertEqual(ctx.exception.status_code, 503)

    def test_resolve_gateway_target_empty_alias_returns_400(self) -> None:
        with self.assertRaises(GatewayRouteError) as ctx:
            resolve_gateway_target("   ")
        self.assertEqual(ctx.exception.status_code, 400)

    def test_resolve_gateway_target_maps_zero_host_to_localhost(self) -> None:
        InferenceInstance.objects.create(
            model_name="bound-all",
            port=11405,
            launch_mode="MULTIMODAL",
            server_config={"model_id": "vlm-alias", "host": "0.0.0.0"},
            status="RUNNING",
        )
        target = resolve_gateway_target("vlm-alias")
        self.assertEqual(target.host, "127.0.0.1")
