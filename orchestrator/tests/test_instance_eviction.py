"""Tests for LRU eviction under memory pressure (MLX-112)."""

from __future__ import annotations

import subprocess
from datetime import timedelta
from unittest.mock import MagicMock, mock_open, patch

from django.test import SimpleTestCase
from django.utils import timezone

from orchestrator.instance_eviction import (
    ensure_headroom_for_load,
    evict_one_lru_instance,
    list_lru_evictable_instances,
)
from orchestrator.memory_pressure import (
    estimate_instance_footprint_bytes,
    host_available_bytes,
    lru_eviction_enabled,
    memory_reserve_bytes,
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

    @patch("orchestrator.instance_eviction.is_wake_in_progress", side_effect=[True, False])
    @patch("orchestrator.instance_eviction.instance_gateway_alias", side_effect=lambda i: i.model_name)
    @patch("orchestrator.instance_eviction.InferenceInstance.objects.filter")
    def test_list_lru_skips_exclude_id_and_wake_in_progress(
        self,
        mock_filter: MagicMock,
        _mock_alias: MagicMock,
        _mock_wake: MagicMock,
    ) -> None:
        idle = timezone.now() - timedelta(hours=1)
        rows = [
            _instance(pk=1, model_name="excluded", lifecycle="on_demand", last_used_at=idle),
            _instance(pk=2, model_name="waking", lifecycle="on_demand", last_used_at=idle),
            _instance(pk=3, model_name="ready", lifecycle="on_demand", last_used_at=idle),
        ]
        mock_filter.return_value.order_by.return_value = rows
        ordered = list_lru_evictable_instances(exclude_id=1)
        self.assertEqual([row.model_name for row in ordered], ["ready"])

    @patch.dict("os.environ", {"NADIR_LRU_EVICTION_GRACE_SECONDS": "not-an-int"}, clear=False)
    def test_eviction_grace_falls_back_on_invalid_env(self) -> None:
        from orchestrator.instance_eviction import eviction_grace_seconds

        self.assertEqual(eviction_grace_seconds(), 60)

    @patch.dict("os.environ", {"NADIR_LRU_EVICTION_ENABLED": "true"}, clear=False)
    @patch("orchestrator.server_manager.is_manual_stop_in_progress", return_value=False)
    @patch("orchestrator.instance_eviction._stop_planned_victim")
    @patch("orchestrator.instance_eviction.list_lru_evictable_instances")
    @patch("orchestrator.instance_eviction.host_available_bytes", return_value=10)
    @patch(
        "orchestrator.instance_eviction.estimate_instance_footprint_bytes",
        return_value=500,
    )
    def test_plan_evictions_reserves_victims_until_stop_finishes(
        self,
        _mock_need: MagicMock,
        _mock_available: MagicMock,
        mock_list: MagicMock,
        mock_stop: MagicMock,
        _mock_manual: MagicMock,
    ) -> None:
        from orchestrator import instance_eviction as eviction_mod

        pending = _instance(
            pk=30,
            model_name="pending",
            lifecycle="on_demand",
            last_used_at=timezone.now(),
            status="LOADING",
        )
        victim = _instance(
            pk=7,
            model_name="victim",
            lifecycle="on_demand",
            last_used_at=timezone.now(),
        )
        mock_list.return_value = [victim]
        reserved_during_stop: set[int] = set()

        def _stop_and_capture(row: MagicMock) -> MagicMock:
            reserved_during_stop.update(eviction_mod._reserved_victim_ids)
            return row

        mock_stop.side_effect = _stop_and_capture
        eviction_mod._reserved_victim_ids.clear()
        self.assertEqual(ensure_headroom_for_load(pending), 1)
        self.assertIn(7, reserved_during_stop)
        self.assertNotIn(7, eviction_mod._reserved_victim_ids)

    @patch("orchestrator.instance_eviction._stop_planned_victim")
    @patch("orchestrator.server_manager.is_manual_stop_in_progress", side_effect=[True, False])
    @patch("orchestrator.instance_eviction.list_lru_evictable_instances")
    @patch("orchestrator.instance_eviction.host_available_bytes", return_value=10)
    @patch(
        "orchestrator.instance_eviction.estimate_instance_footprint_bytes",
        return_value=500,
    )
    @patch.dict("os.environ", {"NADIR_LRU_EVICTION_ENABLED": "true"}, clear=False)
    def test_plan_evictions_skips_manual_stop(
        self,
        _mock_need: MagicMock,
        _mock_available: MagicMock,
        mock_list: MagicMock,
        _mock_manual: MagicMock,
        mock_stop: MagicMock,
    ) -> None:
        pending = _instance(
            pk=20,
            model_name="pending",
            lifecycle="on_demand",
            last_used_at=timezone.now(),
            status="LOADING",
        )
        mock_list.return_value = [
            _instance(pk=1, model_name="manual", lifecycle="on_demand", last_used_at=timezone.now()),
            _instance(pk=2, model_name="ok", lifecycle="on_demand", last_used_at=timezone.now()),
        ]
        mock_stop.side_effect = lambda victim: victim
        self.assertEqual(ensure_headroom_for_load(pending), 1)
        self.assertEqual(mock_stop.call_args[0][0].model_name, "ok")

    @patch("orchestrator.server_manager.is_manual_stop_in_progress", return_value=True)
    def test_stop_planned_victim_skips_manual(self, _mock_manual: MagicMock) -> None:
        from orchestrator.instance_eviction import _stop_planned_victim

        victim = _instance(
            pk=1,
            model_name="manual",
            lifecycle="on_demand",
            last_used_at=timezone.now(),
        )
        self.assertIsNone(_stop_planned_victim(victim))

    @patch("orchestrator.server_manager.stop_instance", side_effect=RuntimeError("boom"))
    @patch("orchestrator.server_manager.is_manual_stop_in_progress", return_value=False)
    @patch("orchestrator.instance_eviction.instance_gateway_alias", return_value="x")
    def test_stop_planned_victim_handles_failure(
        self,
        _mock_alias: MagicMock,
        _mock_manual: MagicMock,
        _mock_stop: MagicMock,
    ) -> None:
        from orchestrator.instance_eviction import _stop_planned_victim

        victim = _instance(
            pk=1,
            model_name="fail",
            lifecycle="on_demand",
            last_used_at=timezone.now(),
        )
        self.assertIsNone(_stop_planned_victim(victim))

    @patch.dict("os.environ", {"NADIR_LRU_EVICTION_GRACE_SECONDS": "120"}, clear=False)
    @patch("orchestrator.instance_eviction.is_wake_in_progress", return_value=False)
    @patch("orchestrator.instance_eviction.instance_gateway_alias", side_effect=lambda i: i.model_name)
    @patch("orchestrator.instance_eviction.InferenceInstance.objects.filter")
    def test_list_lru_skips_instances_within_grace(
        self,
        mock_filter: MagicMock,
        _mock_alias: MagicMock,
        _mock_wake: MagicMock,
    ) -> None:
        recent = timezone.now() - timedelta(seconds=30)
        idle = timezone.now() - timedelta(hours=2)
        rows = [
            _instance(pk=1, model_name="hot", lifecycle="on_demand", last_used_at=recent),
            _instance(pk=2, model_name="cold", lifecycle="on_demand", last_used_at=idle),
        ]
        mock_filter.return_value.order_by.return_value = rows
        ordered = list_lru_evictable_instances()
        self.assertEqual([row.model_name for row in ordered], ["cold"])

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

    @patch(
        "orchestrator.server_manager.is_manual_stop_in_progress",
        side_effect=[True, False, False],
    )
    @patch("orchestrator.server_manager.stop_instance", side_effect=[RuntimeError("boom"), None])
    @patch("orchestrator.instance_eviction.clear_gateway_route_cache")
    @patch("orchestrator.instance_eviction.list_lru_evictable_instances")
    def test_evict_one_skips_manual_stop_and_failed_stop(
        self,
        mock_list: MagicMock,
        mock_cache: MagicMock,
        mock_stop: MagicMock,
        _mock_manual: MagicMock,
    ) -> None:
        manual = _instance(pk=1, model_name="manual", lifecycle="on_demand", last_used_at=timezone.now())
        failing = _instance(pk=2, model_name="fail", lifecycle="on_demand", last_used_at=timezone.now())
        ok = _instance(pk=3, model_name="ok", lifecycle="on_demand", last_used_at=timezone.now())
        mock_list.return_value = [manual, failing, ok]

        stopped = evict_one_lru_instance()
        self.assertIs(stopped, ok)
        self.assertEqual(mock_stop.call_count, 2)
        mock_cache.assert_called_once()

    @patch("orchestrator.server_manager.is_manual_stop_in_progress", return_value=False)
    @patch("orchestrator.server_manager.stop_instance", side_effect=RuntimeError("all failed"))
    @patch("orchestrator.instance_eviction.clear_gateway_route_cache")
    @patch("orchestrator.instance_eviction.list_lru_evictable_instances")
    def test_evict_one_returns_none_when_all_fail(
        self,
        mock_list: MagicMock,
        mock_cache: MagicMock,
        _mock_stop: MagicMock,
        _mock_manual: MagicMock,
    ) -> None:
        mock_list.return_value = [
            _instance(pk=1, model_name="a", lifecycle="on_demand", last_used_at=timezone.now())
        ]
        self.assertIsNone(evict_one_lru_instance())
        mock_cache.assert_not_called()

    @patch.dict("os.environ", {"NADIR_LRU_EVICTION_ENABLED": "true"}, clear=False)
    @patch("orchestrator.server_manager.is_manual_stop_in_progress", return_value=False)
    @patch("orchestrator.instance_eviction._stop_planned_victim")
    @patch("orchestrator.instance_eviction.list_lru_evictable_instances")
    @patch("orchestrator.instance_eviction.host_available_bytes")
    @patch("orchestrator.instance_eviction.estimate_instance_footprint_bytes")
    def test_ensure_headroom_evicts_until_fit(
        self,
        mock_footprint: MagicMock,
        mock_available: MagicMock,
        mock_list: MagicMock,
        mock_stop: MagicMock,
        _mock_manual: MagicMock,
    ) -> None:
        pending = _instance(
            pk=9,
            model_name="pending",
            lifecycle="on_demand",
            last_used_at=timezone.now(),
            status="LOADING",
        )
        victims = [
            _instance(pk=1, model_name="a", lifecycle="on_demand", last_used_at=timezone.now()),
            _instance(pk=2, model_name="b", lifecycle="on_demand", last_used_at=timezone.now()),
        ]
        mock_list.return_value = victims
        mock_stop.side_effect = lambda victim: victim
        mock_footprint.side_effect = lambda model_name, *_a, **_k: (
            1200 if model_name == "pending" else 600
        )
        mock_available.side_effect = [100, 1300]

        stopped = ensure_headroom_for_load(pending)
        self.assertEqual(stopped, 2)
        self.assertEqual(mock_stop.call_count, 2)

    @patch.dict("os.environ", {"NADIR_LRU_EVICTION_ENABLED": "true"}, clear=False)
    @patch("orchestrator.instance_eviction.host_available_bytes", return_value=None)
    @patch(
        "orchestrator.instance_eviction.estimate_instance_footprint_bytes",
        return_value=500,
    )
    def test_ensure_headroom_skips_when_available_unknown(
        self,
        _mock_need: MagicMock,
        _mock_available: MagicMock,
    ) -> None:
        pending = _instance(
            pk=11,
            model_name="pending",
            lifecycle="on_demand",
            last_used_at=timezone.now(),
            status="LOADING",
        )
        self.assertEqual(ensure_headroom_for_load(pending), 0)

    @patch.dict("os.environ", {"NADIR_LRU_EVICTION_ENABLED": "true"}, clear=False)
    @patch("orchestrator.server_manager.is_manual_stop_in_progress", return_value=False)
    @patch("orchestrator.instance_eviction._stop_planned_victim")
    @patch("orchestrator.instance_eviction.list_lru_evictable_instances", return_value=[])
    @patch("orchestrator.instance_eviction.host_available_bytes", return_value=10)
    @patch(
        "orchestrator.instance_eviction.estimate_instance_footprint_bytes",
        return_value=500,
    )
    def test_ensure_headroom_stops_when_no_victim(
        self,
        _mock_need: MagicMock,
        _mock_available: MagicMock,
        mock_list: MagicMock,
        mock_stop: MagicMock,
        _mock_manual: MagicMock,
    ) -> None:
        pending = _instance(
            pk=12,
            model_name="pending",
            lifecycle="on_demand",
            last_used_at=timezone.now(),
            status="LOADING",
        )
        self.assertEqual(ensure_headroom_for_load(pending), 0)
        mock_list.assert_called_once_with(exclude_id=12)
        mock_stop.assert_not_called()

    @patch.dict("os.environ", {"NADIR_LRU_EVICTION_ENABLED": "true"}, clear=False)
    @patch("orchestrator.server_manager.is_manual_stop_in_progress", return_value=False)
    @patch("orchestrator.instance_eviction._stop_planned_victim")
    @patch("orchestrator.instance_eviction.list_lru_evictable_instances")
    @patch("orchestrator.instance_eviction.host_available_bytes")
    @patch(
        "orchestrator.instance_eviction.estimate_instance_footprint_bytes",
        return_value=500,
    )
    def test_ensure_headroom_warns_when_still_short(
        self,
        _mock_need: MagicMock,
        mock_available: MagicMock,
        mock_list: MagicMock,
        mock_stop: MagicMock,
        _mock_manual: MagicMock,
    ) -> None:
        pending = _instance(
            pk=13,
            model_name="pending",
            lifecycle="on_demand",
            last_used_at=timezone.now(),
            status="LOADING",
        )
        victim = _instance(
            pk=1,
            model_name="victim",
            lifecycle="on_demand",
            last_used_at=timezone.now(),
        )
        mock_list.return_value = [victim]
        mock_stop.return_value = victim
        mock_available.side_effect = [10, 50]
        self.assertEqual(ensure_headroom_for_load(pending), 1)

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
        with patch(
            "orchestrator.memory_pressure.get_model_folder_size_bytes",
            return_value=10 * 1024**3,
        ):
            footprint = estimate_instance_footprint_bytes("big-model")
        self.assertGreaterEqual(footprint, int(10 * 1024**3 * 1.15))

    def test_estimate_footprint_defaults_on_error_or_empty(self) -> None:
        with patch(
            "orchestrator.memory_pressure.get_model_folder_size_bytes",
            side_effect=OSError("missing"),
        ):
            self.assertEqual(estimate_instance_footprint_bytes("gone"), 2 * 1024**3)
        with patch(
            "orchestrator.memory_pressure.get_model_folder_size_bytes",
            return_value=0,
        ):
            self.assertEqual(estimate_instance_footprint_bytes("tiny"), 2 * 1024**3)

    @patch.dict("os.environ", {"NADIR_LRU_EVICTION_ENABLED": "yes"}, clear=False)
    def test_lru_eviction_enabled_from_env(self) -> None:
        self.assertTrue(lru_eviction_enabled())

    @patch.dict("os.environ", {"NADIR_MEMORY_RESERVE_GB": "2.5"}, clear=False)
    def test_memory_reserve_from_env(self) -> None:
        self.assertEqual(memory_reserve_bytes(), int(2.5 * 1024**3))

    @patch.dict("os.environ", {"NADIR_MEMORY_RESERVE_GB": "not-a-number"}, clear=False)
    def test_memory_reserve_falls_back_on_invalid_env(self) -> None:
        with patch("orchestrator.memory_pressure.settings") as mock_settings:
            mock_settings.NADIR_MEMORY_RESERVE_GB = 4.0
            self.assertEqual(memory_reserve_bytes(), int(4.0 * 1024**3))

    @patch("orchestrator.memory_pressure.platform.system", return_value="Darwin")
    @patch("orchestrator.memory_pressure.memory_reserve_bytes", return_value=0)
    @patch("orchestrator.memory_pressure.subprocess.check_output")
    def test_darwin_available_bytes(
        self,
        mock_check_output: MagicMock,
        _mock_reserve: MagicMock,
        _mock_system: MagicMock,
    ) -> None:
        mock_check_output.side_effect = [
            "4096\n",
            (
                "Pages free:            100.\n"
                "Pages inactive:         50.\n"
                "Pages speculative:      25.\n"
                "Pages wired down:     9999.\n"
            ),
        ]
        available = host_available_bytes()
        self.assertEqual(available, (100 + 50 + 25) * 4096)

    @patch("orchestrator.memory_pressure.platform.system", return_value="Darwin")
    @patch(
        "orchestrator.memory_pressure.subprocess.check_output",
        side_effect=subprocess.CalledProcessError(1, "vm_stat"),
    )
    def test_darwin_available_bytes_returns_none_on_error(
        self,
        _mock_check_output: MagicMock,
        _mock_system: MagicMock,
    ) -> None:
        self.assertIsNone(host_available_bytes())

    @patch("orchestrator.memory_pressure.platform.system", return_value="Darwin")
    @patch("orchestrator.memory_pressure.memory_reserve_bytes", return_value=0)
    @patch("orchestrator.memory_pressure.subprocess.check_output")
    def test_darwin_available_bytes_none_when_no_free_pages(
        self,
        mock_check_output: MagicMock,
        _mock_reserve: MagicMock,
        _mock_system: MagicMock,
    ) -> None:
        mock_check_output.side_effect = ["4096\n", "Pages wired down: 10.\n"]
        self.assertIsNone(host_available_bytes())

    @patch("orchestrator.memory_pressure.platform.system", return_value="Linux")
    @patch("orchestrator.memory_pressure.memory_reserve_bytes", return_value=1024)
    def test_linux_available_bytes(self, _mock_reserve: MagicMock, _mock_system: MagicMock) -> None:
        meminfo = "MemTotal:       8000000 kB\nMemAvailable:   2000 kB\n"
        with patch("builtins.open", mock_open(read_data=meminfo)):
            available = host_available_bytes()
        self.assertEqual(available, 2000 * 1024 - 1024)

    @patch("orchestrator.memory_pressure.platform.system", return_value="Linux")
    def test_linux_available_bytes_oserror(self, _mock_system: MagicMock) -> None:
        with patch("builtins.open", side_effect=OSError("no /proc")):
            self.assertIsNone(host_available_bytes())

    @patch("orchestrator.memory_pressure.platform.system", return_value="Linux")
    @patch("orchestrator.memory_pressure.memory_reserve_bytes", return_value=0)
    def test_linux_available_bytes_missing_keys(
        self,
        _mock_reserve: MagicMock,
        _mock_system: MagicMock,
    ) -> None:
        with patch("builtins.open", mock_open(read_data="MemTotal: 1000 kB\n")):
            self.assertIsNone(host_available_bytes())
