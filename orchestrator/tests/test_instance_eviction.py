"""Tests for LRU eviction under memory pressure (MLX-112)."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase
from django.utils import timezone

from orchestrator.instance_eviction import (
    ensure_headroom_for_load,
    evict_one_lru_instance,
    list_lru_evictable_instances,
)


def _instance(
    *,
    pk: int,
    model_name: str,
    lifecycle: str,
    last_used_at,
    status: str = "RUNNING",
) -> MagicMock:
    row = MagicMock()
    row.pk = pk
    row.model_name = model_name
    row.port = 11400 + pk
    row.status = status
    row.last_used_at = last_used_at
    row.server_config = {"ops": {"lifecycle_mode": lifecycle, "idle_minutes": 30}}
    return row


class InstanceEvictionTests(SimpleTestCase):
    @patch("orchestrator.instance_eviction.is_wake_in_progress", return_value=False)
    @patch("orchestrator.instance_eviction.instance_gateway_alias", side_effect=lambda i: i.model_name)
    @patch("orchestrator.instance_eviction.InferenceInstance.objects.filter")
    def test_list_lru_orders_by_activity_and_skips_always_on(
        self,
        mock_filter: MagicMock,
        _mock_alias: MagicMock,
        _mock_wake: MagicMock,
    ) -> None:
        older = timezone.now() - timedelta(hours=2)
        newer = timezone.now() - timedelta(minutes=5)
        rows = [
            _instance(pk=1, model_name="old-model", lifecycle="on_demand", last_used_at=older),
            _instance(pk=2, model_name="new-model", lifecycle="on_demand", last_used_at=newer),
            _instance(pk=3, model_name="pinned", lifecycle="always_on", last_used_at=older),
        ]
        mock_filter.return_value.order_by.return_value = rows

        ordered = list_lru_evictable_instances()
        self.assertEqual([row.model_name for row in ordered], ["old-model", "new-model"])

    @patch("orchestrator.server_manager.is_manual_stop_in_progress", return_value=False)
    @patch("orchestrator.server_manager.stop_instance")
    @patch("orchestrator.instance_eviction.clear_gateway_route_cache")
    @patch("orchestrator.instance_eviction.list_lru_evictable_instances")
    def test_evict_one_stops_oldest(
        self,
        mock_list: MagicMock,
        _mock_cache: MagicMock,
        mock_stop: MagicMock,
        _mock_manual: MagicMock,
    ) -> None:
        victim = _instance(
            pk=1,
            model_name="victim",
            lifecycle="on_demand",
            last_used_at=timezone.now() - timedelta(hours=3),
        )
        mock_list.return_value = [victim]
        stopped = evict_one_lru_instance()
        self.assertIs(stopped, victim)
        mock_stop.assert_called_once_with(victim)

    @patch.dict("os.environ", {"NADIR_LRU_EVICTION_ENABLED": "true"}, clear=False)
    @patch("orchestrator.instance_eviction.host_available_bytes", return_value=100)
    @patch(
        "orchestrator.instance_eviction.estimate_instance_footprint_bytes",
        return_value=500,
    )
    @patch("orchestrator.instance_eviction.evict_one_lru_instance")
    def test_ensure_headroom_evicts_until_fit(
        self,
        mock_evict: MagicMock,
        _mock_need: MagicMock,
        mock_available: MagicMock,
    ) -> None:
        pending = _instance(
            pk=9,
            model_name="pending",
            lifecycle="on_demand",
            last_used_at=timezone.now(),
            status="LOADING",
        )
        mock_evict.side_effect = [MagicMock(model_name="a"), MagicMock(model_name="b")]
        mock_available.side_effect = [100, 100, 1000]

        stopped = ensure_headroom_for_load(pending)
        self.assertEqual(stopped, 2)
        self.assertEqual(mock_evict.call_count, 2)

    @patch.dict("os.environ", {"NADIR_LRU_EVICTION_ENABLED": "false"}, clear=False)
    def test_ensure_headroom_noop_when_disabled(self) -> None:
        pending = _instance(
            pk=10,
            model_name="pending",
            lifecycle="on_demand",
            last_used_at=timezone.now(),
            status="LOADING",
        )
        self.assertEqual(ensure_headroom_for_load(pending), 0)


class MemoryPressureHelperTests(SimpleTestCase):
    def test_estimate_footprint_uses_disk_factor(self) -> None:
        from orchestrator.memory_pressure import estimate_instance_footprint_bytes

        with patch(
            "orchestrator.memory_pressure.get_model_folder_size_bytes",
            return_value=10 * 1024**3,
        ):
            footprint = estimate_instance_footprint_bytes("big-model")
        self.assertGreaterEqual(footprint, int(10 * 1024**3 * 1.15))
