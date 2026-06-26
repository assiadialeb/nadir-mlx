"""Tests for lm_eval_runner helpers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.test import SimpleTestCase

from orchestrator.vendor.lm_eval_runner import (
    build_lm_eval_command,
    missing_industry_dependencies,
    normalize_lm_eval_results,
    parse_lm_eval_output,
    run_lm_eval,
)


class LmEvalRunnerTests(SimpleTestCase):
    def test_build_lm_eval_command_targets_gateway(self) -> None:
        command = build_lm_eval_command(
            "127.0.0.1",
            11380,
            "my-alias",
            Path("/tmp/lm-eval-out"),
            preset="industry_lite",
        )
        joined = " ".join(command)
        self.assertIn("lm_eval", joined)
        self.assertIn("local-chat-completions", joined)
        self.assertIn("http://127.0.0.1:11380/v1/chat/completions", joined)
        self.assertIn("ifeval,gsm8k", joined)
        self.assertIn("--apply_chat_template", command)

    def test_normalize_lm_eval_results_extracts_task_metrics(self) -> None:
        raw = {
            "results": {
                "gsm8k": {"exact_match,strict-match": 0.74},
                "ifeval": {"prompt_level_strict_acc,none": 0.72},
            }
        }
        normalized = normalize_lm_eval_results(raw)
        self.assertAlmostEqual(normalized["tasks"]["gsm8k"]["exact_match"], 0.74)
        self.assertAlmostEqual(normalized["tasks"]["ifeval"]["prompt_level_strict_acc"], 0.72)

    def test_parse_lm_eval_output_reads_newest_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            older = tmp_path / "results_old.json"
            newer = tmp_path / "results_new.json"
            older.write_text(json.dumps({"results": {"mmlu": {"acc,none": 0.5}}}), encoding="utf-8")
            newer.write_text(json.dumps({"results": {"mmlu": {"acc,none": 0.61}}}), encoding="utf-8")
            parsed = parse_lm_eval_output(tmp_path)
        self.assertAlmostEqual(parsed["tasks"]["mmlu"]["acc"], 0.61)

    @patch("orchestrator.vendor.lm_eval_runner.is_lm_eval_available", return_value=False)
    def test_run_lm_eval_skips_when_package_missing(self, _mock_available: object) -> None:
        result = run_lm_eval("127.0.0.1", 11380, "alias", Path("/tmp/out"))
        self.assertTrue(result["skipped"])

    @patch("orchestrator.vendor.lm_eval_runner.missing_industry_dependencies", return_value=["langdetect"])
    @patch("orchestrator.vendor.lm_eval_runner.is_lm_eval_available", return_value=True)
    def test_run_lm_eval_skips_when_task_dependencies_missing(
        self,
        _mock_available: object,
        _mock_deps: object,
    ) -> None:
        result = run_lm_eval("127.0.0.1", 11380, "alias", Path("/tmp/out"))
        self.assertTrue(result["skipped"])
        self.assertIn("langdetect", result["reason"])
