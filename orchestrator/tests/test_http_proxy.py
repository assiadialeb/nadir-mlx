"""Tests for gateway HTTP proxy helpers."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from django.test import override_settings
from django.utils import timezone

from orchestrator.gateway.router import CHAT_COMPLETIONS_PATH, GatewayRouteError, GatewayTarget
from orchestrator.gateway.services.http_proxy import (
    forward_request_headers,
    gateway_error,
    passthrough_response_headers,
    prepare_upstream_body,
    proxy_binary_post,
    proxy_json_post,
    proxy_timeout_seconds,
    read_upstream_error,
    resolve_target_from_body,
    resolve_target_from_model,
    proxy_sse_post,
    rewrite_sse_line_model,
    upstream_url_for_path,
    validate_target_launch_mode,
)
from orchestrator.gateway.upstream_concurrency import UpstreamQueueTimeoutError

TARGET = GatewayTarget(
    alias="demo",
    instance_id=1,
    launch_mode="TEXT",
    host="127.0.0.1",
    port=11400,
    upstream_model="default_model",
    api_path=CHAT_COMPLETIONS_PATH,
    max_concurrent_upstream=1,
)


class HttpProxyHelperTests(IsolatedAsyncioTestCase):
    def test_gateway_error_returns_openai_shape(self) -> None:
        response = gateway_error(503, "model_unavailable", "Model is not running.")
        payload = json.loads(response.body)
        self.assertEqual(payload["error"]["code"], "model_unavailable")
        self.assertEqual(payload["error"]["type"], "server_error")
        self.assertEqual(response.status_code, 503)

    def test_read_upstream_error_parses_json_body(self) -> None:
        upstream = httpx.Response(
            400,
            json={"error": {"message": "bad request", "type": "invalid_request_error"}},
        )
        response = read_upstream_error(upstream)
        self.assertEqual(response.status_code, 400)

    def test_read_upstream_error_falls_back_to_text(self) -> None:
        upstream = httpx.Response(502, content=b"upstream exploded")
        response = read_upstream_error(upstream)
        payload = json.loads(response.body)
        self.assertIn("upstream exploded", payload["error"]["message"])

    @override_settings(NADIR_GATEWAY_PROXY_TIMEOUT_SECONDS=120.0)
    def test_proxy_timeout_seconds_reads_settings(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("NADIR_GATEWAY_PROXY_TIMEOUT_SECONDS", None)
            self.assertEqual(proxy_timeout_seconds(), 120.0)


class HttpProxyJsonPostTests(IsolatedAsyncioTestCase):
    async def test_proxy_json_post_returns_upstream_json(self) -> None:
        mock_response = httpx.Response(
            200,
            json={"choices": [], "model": "default_model"},
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        with patch(
            "orchestrator.gateway.services.http_proxy.get_http_client",
            return_value=mock_client,
        ), patch(
            "orchestrator.lifecycle_services.touch_instance_last_used_at",
        ):
            response = await proxy_json_post(
                TARGET,
                "http://127.0.0.1:11400/v1/chat/completions",
                {"model": "default_model"},
                {"content-type": "application/json"},
            )
        payload = json.loads(response.body)
        self.assertEqual(payload["choices"], [])
        self.assertEqual(payload["model"], "demo")

    async def test_proxy_json_post_maps_timeout_to_gateway_error(self) -> None:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ReadTimeout("slow upstream"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        with patch(
            "orchestrator.gateway.services.http_proxy.get_http_client",
            return_value=mock_client,
        ):
            response = await proxy_json_post(
                TARGET,
                "http://127.0.0.1:11400/v1/chat/completions",
                {"model": "default_model"},
                {},
            )
        payload = json.loads(response.body)
        self.assertEqual(payload["error"]["code"], "gateway_timeout")
        self.assertEqual(response.status_code, 504)

    async def test_proxy_json_post_maps_queue_timeout(self) -> None:
        @asynccontextmanager
        async def _queue_timeout(_target: GatewayTarget):
            raise UpstreamQueueTimeoutError("queue full")
            yield  # pragma: no cover

        with patch(
            "orchestrator.gateway.services.http_proxy.upstream_concurrency_slot",
            _queue_timeout,
        ):
            response = await proxy_json_post(TARGET, "http://example.invalid", {}, {})
        payload = json.loads(response.body)
        self.assertEqual(payload["error"]["code"], "upstream_queue_timeout")
        self.assertEqual(response.status_code, 503)


class HttpProxyUtilityTests(TestCase):
    def test_forward_request_headers_adds_json_content_type(self) -> None:
        headers = forward_request_headers({"Accept": "application/json"})
        self.assertEqual(headers["content-type"], "application/json")

    def test_forward_request_headers_injects_inference_api_key(self) -> None:
        with patch.dict(os.environ, {"NADIR_INFERENCE_API_KEY": "worker-secret"}, clear=False):
            headers = forward_request_headers({"Accept": "application/json"})
        self.assertEqual(headers["authorization"], "Bearer worker-secret")

    def test_passthrough_response_headers_skips_hop_by_hop(self) -> None:
        upstream = httpx.Headers(
            [("content-type", "application/json"), ("transfer-encoding", "chunked")],
        )
        passthrough = passthrough_response_headers(upstream)
        self.assertIn("content-type", passthrough)
        self.assertNotIn("transfer-encoding", passthrough)

    def test_prepare_upstream_body_rewrites_model(self) -> None:
        body = prepare_upstream_body({"model": "alias"}, TARGET)
        self.assertEqual(body["model"], "default_model")

    def test_rewrite_sse_line_model_restores_alias(self) -> None:
        line = b'data: {"model":"default_model","choices":[]}'
        rewritten = rewrite_sse_line_model(line, "demo")
        self.assertEqual(json.loads(rewritten[5:])["model"], "demo")

    def test_rewrite_sse_line_model_keeps_done_marker(self) -> None:
        self.assertEqual(rewrite_sse_line_model(b"data: [DONE]", "demo"), b"data: [DONE]")

    def test_upstream_url_for_path_joins_base(self) -> None:
        url = upstream_url_for_path(TARGET, "/v1/chat/completions")
        self.assertTrue(url.endswith("/v1/chat/completions"))

    def test_validate_target_launch_mode_rejects_mismatch(self) -> None:
        image_target = GatewayTarget(
            alias="img",
            instance_id=2,
            launch_mode="IMAGE",
            host="127.0.0.1",
            port=11401,
            upstream_model="flux",
            api_path="/v1/images",
            max_concurrent_upstream=1,
        )
        with self.assertRaises(GatewayRouteError):
            validate_target_launch_mode(image_target, frozenset({"TEXT"}), "chat completions")


class HttpProxyBinaryPostTests(IsolatedAsyncioTestCase):
    async def test_proxy_binary_post_streams_success_body(self) -> None:
        async def _byte_stream():
            yield b"chunk"

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.headers = httpx.Headers({"content-type": "audio/wav"})
        mock_response.aiter_bytes = MagicMock(return_value=_byte_stream())
        mock_response.aclose = AsyncMock()

        mock_client = AsyncMock()
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.send = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        with patch(
            "orchestrator.gateway.services.http_proxy.get_http_client",
            return_value=mock_client,
        ):
            response = await proxy_binary_post(
                TARGET,
                "http://127.0.0.1:11400/v1/audio",
                {"model": "default_model"},
                {"content-type": "application/json"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.media_type, "audio/wav")

    async def test_proxy_sse_post_restores_model_in_chunks(self) -> None:
        async def _byte_stream():
            yield b'data: {"model":"default_model"}\n\n'

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_bytes = MagicMock(return_value=_byte_stream())
        mock_response.aclose = AsyncMock()

        mock_client = AsyncMock()
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.send = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        with patch(
            "orchestrator.gateway.services.http_proxy.get_http_client",
            return_value=mock_client,
        ), patch(
            "orchestrator.lifecycle_services.touch_instance_last_used_at",
        ):
            response = await proxy_sse_post(
                TARGET,
                "http://127.0.0.1:11400/v1/chat/completions",
                {"model": "default_model"},
                {},
            )
            chunks = [chunk async for chunk in response.body_iterator]
        joined = b"".join(chunks)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'"model":"demo"', joined)
        self.assertNotIn(b"default_model", joined)

    async def test_proxy_sse_post_propagates_upstream_http_error(self) -> None:
        mock_response = AsyncMock()
        mock_response.status_code = 400
        mock_response.headers = httpx.Headers({"content-type": "application/json"})
        mock_response.aread = AsyncMock(
            return_value=b'{"error":{"message":"bad request","type":"invalid_request_error"}}'
        )
        mock_response.aclose = AsyncMock()

        mock_client = AsyncMock()
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.send = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        with patch(
            "orchestrator.gateway.services.http_proxy.get_http_client",
            return_value=mock_client,
        ):
            response = await proxy_sse_post(
                TARGET,
                "http://127.0.0.1:11400/v1/chat/completions",
                {"model": "default_model", "stream": True},
                {},
            )
        self.assertEqual(response.status_code, 400)
        payload = json.loads(response.body)
        self.assertEqual(payload["error"]["message"], "bad request")

    async def test_proxy_sse_post_heartbeats_during_prefill_wait(self) -> None:
        """Background heartbeat refreshes activity before the first upstream chunk."""
        touch_times: list[float] = []
        first_chunk_at = {"value": None}

        async def delayed_stream():
            await asyncio.sleep(0.12)
            first_chunk_at["value"] = asyncio.get_running_loop().time()
            yield b'data: {"model":"default_model","choices":[{"delta":{"content":"hi"}}]}\n\n'
            yield b'data: {"model":"default_model","choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'

        async def fake_touch(target: GatewayTarget) -> None:
            touch_times.append(asyncio.get_running_loop().time())

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_bytes = MagicMock(return_value=delayed_stream())
        mock_response.aclose = AsyncMock()
        mock_client = AsyncMock()
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.send = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        with patch(
            "orchestrator.gateway.services.http_proxy.get_http_client",
            return_value=mock_client,
        ), patch(
            "orchestrator.gateway.services.http_proxy.activity_heartbeat_seconds",
            return_value=0.04,
        ), patch(
            "orchestrator.gateway.services.http_proxy.touch_target_last_used_at",
            side_effect=fake_touch,
        ):
            response = await proxy_sse_post(
                TARGET,
                "http://127.0.0.1:11400/v1/chat/completions",
                {"model": "default_model", "stream": True},
                {},
            )
            chunks = [chunk async for chunk in response.body_iterator]

        self.assertTrue(chunks)
        self.assertIsNotNone(first_chunk_at["value"])
        prefill_touches = [t for t in touch_times if t < first_chunk_at["value"]]
        self.assertGreaterEqual(len(prefill_touches), 1)

    async def test_proxy_sse_post_survives_touch_failures(self) -> None:
        async def chunk_stream():
            yield b'data: {"model":"default_model","choices":[{"delta":{"content":"ok"}}]}\n\n'

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_bytes = MagicMock(return_value=chunk_stream())
        mock_response.aclose = AsyncMock()
        mock_client = AsyncMock()
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.send = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        with patch(
            "orchestrator.gateway.services.http_proxy.get_http_client",
            return_value=mock_client,
        ), patch(
            "orchestrator.gateway.services.http_proxy.activity_heartbeat_seconds",
            return_value=0.01,
        ), patch(
            "orchestrator.lifecycle_services.touch_instance_last_used_at",
            side_effect=RuntimeError("database is locked"),
        ):
            response = await proxy_sse_post(
                TARGET,
                "http://127.0.0.1:11400/v1/chat/completions",
                {"model": "default_model", "stream": True},
                {},
            )
            chunks = [chunk async for chunk in response.body_iterator]

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'"model":"demo"', b"".join(chunks))

    async def test_proxy_json_post_survives_touch_failures(self) -> None:
        mock_response = httpx.Response(
            200,
            json={"choices": [], "model": "default_model"},
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "orchestrator.gateway.services.http_proxy.get_http_client",
            return_value=mock_client,
        ), patch(
            "orchestrator.gateway.services.http_proxy.activity_heartbeat_seconds",
            return_value=0.01,
        ), patch(
            "orchestrator.lifecycle_services.touch_instance_last_used_at",
            side_effect=RuntimeError("database is locked"),
        ):
            await asyncio.sleep(0.03)
            response = await proxy_json_post(
                TARGET,
                "http://127.0.0.1:11400/v1/chat/completions",
                {"model": "default_model"},
                {"content-type": "application/json"},
            )

        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.body)
        self.assertEqual(payload["model"], "demo")

    async def test_proxy_sse_post_heartbeats_keep_instance_out_of_lru(self) -> None:
        """Background heartbeats keep last_used_at fresh across a stream longer than grace."""
        activity = {"last_used_at": timezone.now()}

        async def slow_stream():
            await asyncio.sleep(0.55)
            yield b'data: {"model":"default_model","choices":[{"delta":{"content":"a"}}]}\n\n'
            await asyncio.sleep(0.55)
            yield b'data: {"model":"default_model","choices":[{"delta":{"content":"b"}}]}\n\n'
            yield b'data: {"model":"default_model","choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'

        async def fake_touch(target: GatewayTarget) -> None:
            activity["last_used_at"] = timezone.now()

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_bytes = MagicMock(return_value=slow_stream())
        mock_response.aclose = AsyncMock()
        mock_client = AsyncMock()
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.send = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        row = MagicMock()
        row.pk = TARGET.instance_id
        row.model_name = "streaming"
        row.port = 11400
        row.status = "RUNNING"
        row.server_config = {"ops": {"lifecycle_mode": "on_demand", "idle_minutes": 30}}
        type(row).last_used_at = property(lambda self: activity["last_used_at"])

        observed_while_streaming: list[list[str]] = []

        with patch(
            "orchestrator.gateway.services.http_proxy.get_http_client",
            return_value=mock_client,
        ), patch(
            "orchestrator.gateway.services.http_proxy.activity_heartbeat_seconds",
            return_value=0.25,
        ), patch(
            "orchestrator.gateway.services.http_proxy.touch_target_last_used_at",
            side_effect=fake_touch,
        ), patch(
            "orchestrator.instance_eviction.InferenceInstance.objects.filter",
        ) as mock_filter, patch(
            "orchestrator.instance_eviction.is_wake_in_progress",
            return_value=False,
        ), patch(
            "orchestrator.instance_eviction.instance_gateway_alias",
            side_effect=lambda i: i.model_name,
        ), patch.dict(
            "os.environ",
            {"NADIR_LRU_EVICTION_GRACE_SECONDS": "1"},
            clear=False,
        ):
            mock_filter.return_value.order_by.return_value = [row]
            response = await proxy_sse_post(
                TARGET,
                "http://127.0.0.1:11400/v1/chat/completions",
                {"model": "default_model", "stream": True},
                {},
            )
            from orchestrator.instance_eviction import list_lru_evictable_instances

            async for _chunk in response.body_iterator:
                observed_while_streaming.append(
                    [inst.model_name for inst in list_lru_evictable_instances()]
                )

        self.assertGreaterEqual(len(observed_while_streaming), 2)
        self.assertTrue(all(names == [] for names in observed_while_streaming))


class HttpProxyResolveTargetTests(IsolatedAsyncioTestCase):
    async def test_resolve_target_from_model_rejects_blank(self) -> None:
        with self.assertRaises(GatewayRouteError):
            await resolve_target_from_model("")

    @patch("orchestrator.gateway.services.http_proxy.sync_to_async")
    async def test_resolve_target_from_body_delegates_to_model(self, mock_sync: MagicMock) -> None:
        def fake_sync_to_async(fn, thread_sensitive=False):
            async def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

            return wrapper

        mock_sync.side_effect = fake_sync_to_async
        with patch(
            "orchestrator.lifecycle_services.ensure_instance_ready",
            return_value=TARGET,
        ):
            with patch("orchestrator.lifecycle_services.touch_instance_last_used_at"):
                target = await resolve_target_from_body({"model": "demo"})
        self.assertEqual(target.alias, "demo")
