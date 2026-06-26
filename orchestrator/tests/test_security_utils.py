"""Tests for shared security helpers."""

from pathlib import Path

from django.test import SimpleTestCase, override_settings

from orchestrator.security_utils import (
    benchmark_endpoint_enabled,
    public_error_message,
    safe_path_under_root,
    sanitize_hf_search_query,
    validate_benchmark_endpoint_host,
    validate_huggingface_api_url,
    validate_outbound_http_host,
    validate_server_bind_host,
)


class SecurityUtilsTests(SimpleTestCase):
    def test_validate_outbound_http_host_allows_loopback(self) -> None:
        self.assertEqual(validate_outbound_http_host("127.0.0.1"), "127.0.0.1")
        self.assertEqual(validate_outbound_http_host("localhost"), "localhost")

    def test_validate_outbound_http_host_rejects_metadata_host(self) -> None:
        with self.assertRaises(ValueError):
            validate_outbound_http_host("metadata.google.internal")

    def test_validate_outbound_http_host_rejects_public_ip(self) -> None:
        with self.assertRaises(ValueError):
            validate_outbound_http_host("8.8.8.8")

    def test_validate_server_bind_host_maps_unbound_listen_address(self) -> None:
        self.assertEqual(validate_server_bind_host("192.168.1.10"), "192.168.1.10")

    def test_sanitize_hf_search_query_rejects_url_like_input(self) -> None:
        with self.assertRaises(ValueError):
            sanitize_hf_search_query("http://evil.example")

    def test_validate_huggingface_api_url_rejects_foreign_origin(self) -> None:
        with self.assertRaises(ValueError):
            validate_huggingface_api_url("https://evil.example/api/models")

    def test_validate_huggingface_api_url_accepts_official_api(self) -> None:
        url = validate_huggingface_api_url("https://huggingface.co/api/models")
        self.assertEqual(url, "https://huggingface.co/api/models")

    def test_safe_path_under_root_rejects_traversal(self) -> None:
        root = Path("/tmp/nadir-benchmarks")
        with self.assertRaises(ValueError):
            safe_path_under_root(root, "../secret.json")

    @override_settings(MODELS_DIR="/tmp/mlx-models")
    def test_safe_path_under_root_stays_inside_root(self) -> None:
        root = Path("/tmp/nadir-benchmarks")
        resolved = safe_path_under_root(root, "bench_12.json")
        self.assertEqual(resolved, root.resolve() / "bench_12.json")

    def test_public_error_message_hides_unexpected_exceptions(self) -> None:
        message = public_error_message(RuntimeError("traceback details"), fallback="Failed.")
        self.assertEqual(message, "Failed.")

    def test_public_error_message_keeps_value_error_text(self) -> None:
        message = public_error_message(ValueError("Invalid host."))
        self.assertEqual(message, "Invalid host.")

    @override_settings(
        DEBUG=False,
        NADIR_BENCHMARK_ENDPOINT_ENABLED=True,
    )
    def test_validate_benchmark_endpoint_host_allows_loopback_in_prod(self) -> None:
        self.assertEqual(validate_benchmark_endpoint_host("127.0.0.1"), "127.0.0.1")
        self.assertEqual(validate_benchmark_endpoint_host("localhost"), "localhost")

    @override_settings(
        DEBUG=False,
        NADIR_BENCHMARK_ENDPOINT_ENABLED=True,
    )
    def test_validate_benchmark_endpoint_host_rejects_private_network_in_prod(self) -> None:
        with self.assertRaises(ValueError):
            validate_benchmark_endpoint_host("192.168.1.10")

    @override_settings(
        DEBUG=False,
        NADIR_BENCHMARK_ENDPOINT_ENABLED=True,
    )
    def test_validate_benchmark_endpoint_host_rejects_link_local(self) -> None:
        with self.assertRaises(ValueError):
            validate_benchmark_endpoint_host("169.254.169.254")

    @override_settings(DEBUG=False, NADIR_BENCHMARK_ENDPOINT_ENABLED=False)
    def test_validate_benchmark_endpoint_host_disabled_outside_debug(self) -> None:
        with self.assertRaises(ValueError):
            validate_benchmark_endpoint_host("localhost")

    @override_settings(DEBUG=True, NADIR_BENCHMARK_ENDPOINT_ENABLED=True)
    def test_validate_benchmark_endpoint_host_allows_private_network_in_debug(self) -> None:
        self.assertEqual(validate_benchmark_endpoint_host("192.168.1.10"), "192.168.1.10")

    @override_settings(DEBUG=False, NADIR_BENCHMARK_ENDPOINT_ENABLED=False)
    def test_benchmark_endpoint_enabled_follows_setting(self) -> None:
        self.assertFalse(benchmark_endpoint_enabled())
