"""Tests for platform qualitybench scorers and suites."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase

from orchestrator.vendor.qualitybench import (
    load_suite,
    run_platform_suites,
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
