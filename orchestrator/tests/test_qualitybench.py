"""Tests for platform qualitybench scorers and suites."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase

from orchestrator.vendor.qualitybench import (
    load_suite,
    run_platform_suites,
    run_suite_case,
    score_response,
)


class QualitybenchScorerTests(SimpleTestCase):
    def test_score_response_regex(self) -> None:
        self.assertTrue(score_response("YES", {"type": "regex", "pattern": "^YES$"}))
        self.assertFalse(score_response("no", {"type": "regex", "pattern": "^YES$"}))

    def test_score_response_contains(self) -> None:
        self.assertTrue(score_response("The sky is blue today", {"type": "contains", "value": "blue"}))

    def test_score_response_json_schema_valid(self) -> None:
        scorer = {
            "type": "json_schema_valid",
            "required_keys": ["name", "age"],
            "types": {"name": "string", "age": "integer"},
        }
        self.assertTrue(score_response('{"name": "Ada", "age": 30}', scorer))
        self.assertFalse(score_response('{"name": "Ada"}', scorer))

    def test_load_suite_text_platform_has_cases(self) -> None:
        suite = load_suite("text_platform")
        self.assertGreaterEqual(len(suite.get("cases", [])), 10)


class QualitybenchRunTests(SimpleTestCase):
    @patch("orchestrator.vendor.qualitybench.httpx.post")
    def test_run_platform_suites_aggregates_pass_rate(self, mock_post: MagicMock) -> None:
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": "YES"}}],
        }
        mock_post.return_value = response

        suite = load_suite("text_platform")
        with patch("orchestrator.vendor.qualitybench.DEFAULT_SUITE_NAMES", ("text_platform",)):
            with patch("orchestrator.vendor.qualitybench.load_suite", return_value=suite):
                result = run_platform_suites("127.0.0.1", 11380, "alias")

        payload = result["suites"]["text_platform"]
        self.assertEqual(payload["total"], len(suite["cases"]))
        self.assertGreaterEqual(payload["pass_rate"], 0.0)

    @patch("orchestrator.vendor.qualitybench.httpx.post")
    def test_run_suite_case_sends_gateway_api_key(self, mock_post: MagicMock) -> None:
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": "YES"}}],
        }
        mock_post.return_value = response
        case = {
            "id": "auth",
            "prompt": "Say YES",
            "max_tokens": 8,
            "scorer": {"type": "regex", "pattern": "^YES$"},
        }
        with patch.dict(os.environ, {"NADIR_GATEWAY_API_KEY": "bench-secret"}, clear=False):
            run_suite_case("127.0.0.1", 11380, "alias", case)
        headers = mock_post.call_args.kwargs.get("headers") or {}
        self.assertEqual(headers.get("Authorization"), "Bearer bench-secret")
