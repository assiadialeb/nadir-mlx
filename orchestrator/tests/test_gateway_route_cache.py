"""Tests for gateway route cache."""

from __future__ import annotations

import os
from unittest.mock import patch

from django.test import TestCase, override_settings

from orchestrator.gateway.route_cache import (
    clear_gateway_route_cache,
    gateway_route_cache_ttl_seconds,
    get_route_snapshot,
)
from orchestrator.gateway.router import GatewayRouteError
from orchestrator.gateway.selectors import (
    build_route_snapshot_from_db,
    list_running_gateway_models,
    resolve_gateway_target,
)
from orchestrator.models import InferenceInstance


class GatewayRouteCacheTests(TestCase):
    def setUp(self) -> None:
        clear_gateway_route_cache()

    def tearDown(self) -> None:
        clear_gateway_route_cache()

    def test_gateway_route_cache_ttl_reads_settings(self) -> None:
        with patch.dict(os.environ, {"NADIR_GATEWAY_ROUTE_CACHE_TTL_SECONDS": "45"}, clear=False):
            clear_gateway_route_cache()
            self.assertEqual(gateway_route_cache_ttl_seconds(), 45.0)

    def test_resolve_and_list_models_share_single_snapshot_build(self) -> None:
        InferenceInstance.objects.create(
            model_name="Llama-3-8B",
            port=11400,
            launch_mode="TEXT",
            server_config={"model_id": "llama-chat"},
            status="RUNNING",
        )
        with patch(
            "orchestrator.gateway.selectors.build_route_snapshot_from_db",
            wraps=build_route_snapshot_from_db,
        ) as mock_build:
            resolve_gateway_target("llama-chat")
            list_running_gateway_models()
            resolve_gateway_target("llama-chat")
            self.assertEqual(mock_build.call_count, 1)

    def test_cache_expires_after_ttl(self) -> None:
        InferenceInstance.objects.create(
            model_name="Llama-3-8B",
            port=11400,
            launch_mode="TEXT",
            server_config={"model_id": "llama-chat"},
            status="RUNNING",
        )
        times = [0.0, 0.0, 5.0, 25.0, 25.0]
        with (
            override_settings(NADIR_GATEWAY_ROUTE_CACHE_TTL_SECONDS=20.0),
            patch(
                "orchestrator.gateway.route_cache.time.monotonic",
                side_effect=times,
            ),
            patch(
                "orchestrator.gateway.selectors.build_route_snapshot_from_db",
                wraps=build_route_snapshot_from_db,
            ) as mock_build,
        ):
            clear_gateway_route_cache()
            resolve_gateway_target("llama-chat")
            resolve_gateway_target("llama-chat")
            resolve_gateway_target("llama-chat")
            self.assertEqual(mock_build.call_count, 2)

    def test_force_refresh_rebuilds_snapshot(self) -> None:
        InferenceInstance.objects.create(
            model_name="Llama-3-8B",
            port=11400,
            launch_mode="TEXT",
            server_config={"model_id": "llama-chat"},
            status="RUNNING",
        )
        with patch(
            "orchestrator.gateway.selectors.build_route_snapshot_from_db",
            wraps=build_route_snapshot_from_db,
        ) as mock_build:
            get_route_snapshot()
            get_route_snapshot(force_refresh=True)
            self.assertEqual(mock_build.call_count, 2)

    def test_clear_cache_forces_next_resolve_to_rebuild(self) -> None:
        InferenceInstance.objects.create(
            model_name="Llama-3-8B",
            port=11400,
            launch_mode="TEXT",
            server_config={"model_id": "llama-chat"},
            status="RUNNING",
        )
        with patch(
            "orchestrator.gateway.selectors.build_route_snapshot_from_db",
            wraps=build_route_snapshot_from_db,
        ) as mock_build:
            resolve_gateway_target("llama-chat")
            clear_gateway_route_cache()
            resolve_gateway_target("llama-chat")
            self.assertEqual(mock_build.call_count, 2)

    def test_stopped_instance_still_resolves_via_alias_status(self) -> None:
        InferenceInstance.objects.create(
            model_name="offline-model",
            port=11402,
            launch_mode="TEXT",
            server_config={"model_id": "offline"},
            status="STOPPED",
        )
        with patch(
            "orchestrator.gateway.selectors.build_route_snapshot_from_db",
            wraps=build_route_snapshot_from_db,
        ) as mock_build:
            with self.assertRaises(GatewayRouteError) as ctx:
                resolve_gateway_target("offline")
            with self.assertRaises(GatewayRouteError):
                resolve_gateway_target("offline")
            self.assertEqual(ctx.exception.status_code, 503)
            self.assertEqual(mock_build.call_count, 1)
