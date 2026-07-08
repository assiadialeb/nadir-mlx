"""Tests for shared security helpers."""

import os
from pathlib import Path
from unittest.mock import patch

from django.test import SimpleTestCase, override_settings

from orchestrator.security_utils import (
    assert_path_under_directory,
    benchmark_compare_export_filename,
    benchmark_endpoint_enabled,
    build_models_redirect_url,
    build_validated_http_url,
    extract_bearer_token,
    public_error_message,
    safe_path_under_root,
    safe_positive_int,
    sanitize_hf_search_query,
    validated_launch_port,
    validated_subprocess_model_reference,
    validated_sqlite_migration_path,
    validate_benchmark_endpoint_host,
    validate_huggingface_api_url,
    validate_outbound_http_host,
    validate_server_bind_host,
)


class SecurityUtilsTests(SimpleTestCase):
    def test_extract_bearer_token_parses_authorization_header(self) -> None:
        self.assertEqual(extract_bearer_token("Bearer secret-token"), "secret-token")
        self.assertEqual(extract_bearer_token("secret-token"), "secret-token")
        self.assertEqual(extract_bearer_token(None), "")

    def test_sanitize_hf_search_query_truncates_long_input(self) -> None:
        long_query = "a" * 250
        self.assertEqual(len(sanitize_hf_search_query(long_query)), 200)

    def test_safe_positive_int_rejects_zero(self) -> None:
        with self.assertRaises(ValueError):
            safe_positive_int(0, field_name="run id")

    def test_assert_path_under_directory_rejects_escape(self) -> None:
        root = Path("/tmp/nadir-root")
        with self.assertRaises(ValueError):
            assert_path_under_directory(Path("/etc/passwd"), root)

    @override_settings(DEBUG=False, NADIR_BENCHMARK_ENDPOINT_ENABLED=True)
    @patch.dict(os.environ, {"NADIR_BENCHMARK_ENDPOINT_ALLOWED_HOSTS": "10.0.0.5"}, clear=False)
    def test_validate_benchmark_endpoint_host_honors_env_allowlist(self) -> None:
        self.assertEqual(validate_benchmark_endpoint_host("10.0.0.5"), "10.0.0.5")
        with self.assertRaises(ValueError):
            validate_benchmark_endpoint_host("192.168.1.10")

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
    @patch.dict(os.environ, {"NADIR_BENCHMARK_ENDPOINT_ALLOWED_HOSTS": ""}, clear=False)
    def test_validate_benchmark_endpoint_host_allows_private_network_in_debug(self) -> None:
        self.assertEqual(validate_benchmark_endpoint_host("192.168.1.10"), "192.168.1.10")

    @override_settings(DEBUG=False, NADIR_BENCHMARK_ENDPOINT_ENABLED=False)
    def test_benchmark_endpoint_enabled_follows_setting(self) -> None:
        self.assertFalse(benchmark_endpoint_enabled())

    def test_build_models_redirect_url_whitelists_query_keys(self) -> None:
        url = build_models_redirect_url({"tab": "hub", "evil": "x", "q": "llama"})
        self.assertIn("/models/?", url)
        self.assertIn("tab=hub", url)
        self.assertIn("q=llama", url)
        self.assertNotIn("evil=", url)

    def test_build_validated_http_url_rejects_public_host(self) -> None:
        with self.assertRaises(ValueError):
            build_validated_http_url("8.8.8.8", 8080, "/v1/models")

    def test_benchmark_compare_export_filename_uses_validated_ids(self) -> None:
        self.assertEqual(
            benchmark_compare_export_filename(3, 7),
            "bench_compare_3_vs_7.json",
        )

    def test_validated_sqlite_migration_path_rejects_traversal(self) -> None:
        with self.assertRaises(ValueError):
            validated_sqlite_migration_path("../etc/passwd")

    def test_validated_launch_port_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            validated_launch_port(0)
        with self.assertRaises(ValueError):
            validated_launch_port(70000)

    def test_validated_subprocess_model_reference_rejects_shell_chars(self) -> None:
        with self.assertRaises(ValueError):
            validated_subprocess_model_reference("model;rm -rf /")
        self.assertEqual(validated_subprocess_model_reference("llama-chat"), "llama-chat")
