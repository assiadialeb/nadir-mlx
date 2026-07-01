"""Tests for one-click performance profiles."""

from django.test import SimpleTestCase

from orchestrator.perf_profiles import (
    resolve_perf_profiles_for_model,
    suggest_mtp_assistant_folder,
)


class PerfProfileTests(SimpleTestCase):
    def test_suggest_mtp_assistant_folder_for_e4b(self) -> None:
        self.assertEqual(
            suggest_mtp_assistant_folder("gemma-4-E4B-it-qat-4bit"),
            "gemma-4-E4B-it-assistant-bf16",
        )

    def test_suggest_mtp_assistant_folder_for_e2b(self) -> None:
        self.assertEqual(
            suggest_mtp_assistant_folder("gemma-4-E2B-it-qat-4bit"),
            "gemma-4-E2B-it-assistant-bf16",
        )

    def test_resolve_perf_profile_for_gemma4_e4b_qat(self) -> None:
        profiles = resolve_perf_profiles_for_model(
            "gemma-4-E4B-it-qat-4bit",
            "MULTIMODAL",
        )
        self.assertEqual(len(profiles), 1)
        advanced = profiles[0]["server_config"]["advanced"]
        self.assertEqual(advanced["draft_kind"], "mtp")
        self.assertEqual(advanced["draft_model"], "gemma-4-E4B-it-assistant-bf16")

    def test_resolve_perf_profile_skips_non_multimodal(self) -> None:
        profiles = resolve_perf_profiles_for_model(
            "gemma-4-E4B-it-qat-4bit",
            "TEXT",
        )
        self.assertEqual(profiles, [])
