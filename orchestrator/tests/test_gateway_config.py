"""Tests for gateway configuration."""

from django.test import TestCase, override_settings

from orchestrator.gateway.config import load_gateway_config


class GatewayConfigTests(TestCase):
    @override_settings(
        NADIR_GATEWAY_HOST="127.0.0.1",
        NADIR_GATEWAY_PORT=11380,
        INSTANCE_PORT_RANGE_START=11400,
        INSTANCE_PORT_RANGE_END=11500,
    )
    def test_load_gateway_config_uses_settings_defaults(self) -> None:
        config = load_gateway_config()
        self.assertEqual(config.host, "127.0.0.1")
        self.assertEqual(config.port, 11380)

    @override_settings(
        NADIR_GATEWAY_PORT=11450,
        INSTANCE_PORT_RANGE_START=11400,
        INSTANCE_PORT_RANGE_END=11500,
    )
    def test_load_gateway_config_rejects_instance_port_collision(self) -> None:
        with self.assertRaises(ValueError):
            load_gateway_config()
