"""Tests for lifecycle policy selectors (MLX-39 / MLX-44)."""

from unittest import TestCase

from django.test import TestCase as DjangoTestCase

from orchestrator.lifecycle_selectors import (
    DEFAULT_IDLE_MINUTES,
    LIFECYCLE_MODE_ALWAYS_ON,
    LIFECYCLE_MODE_ON_DEMAND,
    enrich_instance_lifecycle_ui,
    get_idle_minutes,
    get_lifecycle_mode,
    instance_status_badge,
    is_on_demand_lifecycle,
    lifecycle_policy_summary,
)
from orchestrator.models import InferenceInstance


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


class LifecycleUiTests(DjangoTestCase):
    def test_on_demand_status_badges(self) -> None:
        config = {"ops": {"lifecycle_mode": LIFECYCLE_MODE_ON_DEMAND, "idle_minutes": 15}}
        instance = InferenceInstance(
            model_name="demo",
            port=11400,
            launch_mode="TEXT",
            server_config=config,
            status="STOPPED",
        )
        self.assertEqual(instance_status_badge(instance), ("Sleeping", "sleeping"))
        instance.status = "LOADING"
        self.assertEqual(instance_status_badge(instance), ("Waking", "waking"))
        instance.status = "RUNNING"
        self.assertEqual(instance_status_badge(instance), ("Ready", "ready"))

    def test_always_on_keeps_default_status_labels(self) -> None:
        instance = InferenceInstance(
            model_name="demo",
            port=11400,
            launch_mode="TEXT",
            server_config={"ops": {"lifecycle_mode": LIFECYCLE_MODE_ALWAYS_ON}},
            status="RUNNING",
        )
        self.assertEqual(instance_status_badge(instance), ("Running", "running"))

    def test_lifecycle_policy_summary(self) -> None:
        self.assertEqual(lifecycle_policy_summary({}), "Always on")
        on_demand = {"ops": {"lifecycle_mode": LIFECYCLE_MODE_ON_DEMAND, "idle_minutes": 20}}
        self.assertEqual(lifecycle_policy_summary(on_demand), "On demand · idle 20 min")

    def test_enrich_instance_lifecycle_ui_sets_template_attrs(self) -> None:
        instance = InferenceInstance.objects.create(
            model_name="ui-demo",
            port=11401,
            launch_mode="TEXT",
            server_config={"ops": {"lifecycle_mode": LIFECYCLE_MODE_ON_DEMAND, "idle_minutes": 10}},
            status="STOPPED",
        )
        enrich_instance_lifecycle_ui(instance)
        self.assertTrue(instance.lifecycle_on_demand)
        self.assertEqual(instance.status_badge_label, "Sleeping")
        self.assertEqual(instance.status_badge_variant, "sleeping")
        self.assertEqual(instance.lifecycle_policy_label, "On demand · idle 10 min")
