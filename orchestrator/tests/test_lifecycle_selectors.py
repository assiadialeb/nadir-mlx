"""Tests for lifecycle policy selectors (MLX-39)."""

from unittest import TestCase

from orchestrator.lifecycle_selectors import (
    DEFAULT_IDLE_MINUTES,
    LIFECYCLE_MODE_ALWAYS_ON,
    LIFECYCLE_MODE_ON_DEMAND,
    get_idle_minutes,
    get_lifecycle_mode,
    is_on_demand_lifecycle,
)


class LifecycleSelectorTests(TestCase):
    def test_defaults_when_ops_missing(self) -> None:
        self.assertEqual(get_lifecycle_mode({}), LIFECYCLE_MODE_ALWAYS_ON)
        self.assertEqual(get_idle_minutes({}), DEFAULT_IDLE_MINUTES)
        self.assertFalse(is_on_demand_lifecycle({}))

    def test_reads_on_demand_and_idle_minutes(self) -> None:
        config = {
            "ops": {
                "lifecycle_mode": LIFECYCLE_MODE_ON_DEMAND,
                "idle_minutes": 45,
            }
        }
        self.assertEqual(get_lifecycle_mode(config), LIFECYCLE_MODE_ON_DEMAND)
        self.assertEqual(get_idle_minutes(config), 45)
        self.assertTrue(is_on_demand_lifecycle(config))

    def test_invalid_lifecycle_mode_falls_back_to_always_on(self) -> None:
        config = {"ops": {"lifecycle_mode": "invalid"}}
        self.assertEqual(get_lifecycle_mode(config), LIFECYCLE_MODE_ALWAYS_ON)

    def test_idle_minutes_clamped_to_bounds(self) -> None:
        low = {"ops": {"idle_minutes": 1}}
        high = {"ops": {"idle_minutes": 9999}}
        self.assertEqual(get_idle_minutes(low), 5)
        self.assertEqual(get_idle_minutes(high), 1440)
