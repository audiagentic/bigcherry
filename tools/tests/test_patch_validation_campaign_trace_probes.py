"""HI82 item 4: RD12/RD13 two-probe activation evidence (design: GPT,
req_cc5af49494fe457a). Also a regression guard for the real false-positive
bug found in RD13's own marker: the marker must be conditioned on
has_view, or it fires for the pre-existing, unrelated fusion path too."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import patch_validation_campaign  # noqa: E402


class TraceProbeClassificationTests(unittest.TestCase):
    def _run(self, monkeypatch, outputs, patch_name="1205_rd12_paired_mmvq_dual_output"):
        iterator = iter(outputs)
        monkeypatch.setattr(
            patch_validation_campaign, "_run_one_trace_probe",
            lambda **kwargs: next(iterator),
        )
        return patch_validation_campaign.run_trace_activation_probes(
            marker_regex=r"BIGCHERRY_PATCH_HIT patch=1205_rd12 path=dual_output_mmvq_fusion",
            description="RD12 trace", binary=Path("unused"), model=Path("unused"),
            hip_path=Path("unused"), workdir=Path("unused"),
            bench_prompt=512, bench_gen=128,
        )

    def test_unknown_patch_returns_none(self):
        result = patch_validation_campaign.run_trace_activation_probes(
            marker_regex=None, description=None, binary=Path("unused"), model=Path("unused"),
            hip_path=Path("unused"), workdir=Path("unused"),
            bench_prompt=512, bench_gen=128,
        )
        self.assertIsNone(result)

    def test_positive_hit_and_negative_absence_is_executed(self, monkeypatch=None):
        import unittest.mock as mock
        with mock.patch.object(
            patch_validation_campaign, "_run_one_trace_probe",
            side_effect=[
                "...\nBIGCHERRY_PATCH_HIT patch=1205_rd12 path=dual_output_mmvq_fusion\n...",
                "... no marker ...",
            ],
        ):
            result = patch_validation_campaign.run_trace_activation_probes(
                marker_regex=r"BIGCHERRY_PATCH_HIT patch=1205_rd12 path=dual_output_mmvq_fusion",
                description="RD12 trace", binary=Path("unused"), model=Path("unused"), hip_path=Path("unused"),
                workdir=Path("unused"), bench_prompt=512, bench_gen=128,
            )
        self.assertIsNotNone(result)
        evidence, detail = result
        self.assertEqual(evidence.status, "executed")
        self.assertTrue(detail["positive"]["marker_observed"])
        self.assertFalse(detail["negative_control"]["marker_observed"])

    def test_missing_positive_is_not_executed(self):
        import unittest.mock as mock
        with mock.patch.object(
            patch_validation_campaign, "_run_one_trace_probe", return_value="no marker",
        ):
            evidence, _ = patch_validation_campaign.run_trace_activation_probes(
                marker_regex=r"BIGCHERRY_PATCH_HIT patch=1205_rd12 path=dual_output_mmvq_fusion",
                description="RD12 trace", binary=Path("unused"), model=Path("unused"), hip_path=Path("unused"),
                workdir=Path("unused"), bench_prompt=512, bench_gen=128,
            )
        self.assertEqual(evidence.status, "not_executed")

    def test_negative_control_hit_is_unobservable(self):
        import unittest.mock as mock
        marker = "BIGCHERRY_PATCH_HIT patch=1206_rd13 path=mul_mat_add_view_fusion_q"
        with mock.patch.object(
            patch_validation_campaign, "_run_one_trace_probe", return_value=marker,
        ):
            evidence, _ = patch_validation_campaign.run_trace_activation_probes(
                marker_regex=r"BIGCHERRY_PATCH_HIT patch=1206_rd13 path=mul_mat_add_view_fusion_(?:f|q)",
                description="RD13 trace", binary=Path("unused"), model=Path("unused"), hip_path=Path("unused"),
                workdir=Path("unused"), bench_prompt=512, bench_gen=128,
            )
        self.assertEqual(evidence.status, "unobservable")


class Rd13MarkerHasViewGuardTests(unittest.TestCase):
    def test_rd13_trace_sites_are_conditioned_on_has_view(self):
        # Regression guard for the real false-positive bug: both trace
        # sites must test has_view, not merely getenv(...), or the marker
        # fires on the pre-existing (non-RD13) fusion path too.
        text = (
            Path(__file__).resolve().parents[2]
            / "patches" / "rd" / "1206_rd13_mul_mat_add_view_fusion" / "patch.py"
        ).read_text(encoding="utf-8")
        occurrences = text.count('if (has_view && getenv("BIGCHERRY_PATCH_TRACE") != nullptr)')
        self.assertEqual(occurrences, 2, "expected both RD13 trace sites to check has_view")
        self.assertNotIn(
            'if (getenv("BIGCHERRY_PATCH_TRACE") != nullptr)', text,
            "an RD13 trace site is missing the has_view guard",
        )


if __name__ == "__main__":
    unittest.main()
