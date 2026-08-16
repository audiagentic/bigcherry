"""RE14 cutover parity gate."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry.parity import CampaignArm, ParityError, check_parity  # noqa: E402


def _arm(name: str, *, source_revision="rev1", variant_set="replay-slim",
         manifest_hash="m1", descriptor_hash="d1",
         candidates=("mmq:native:v1", "mmvq:native:v1"), binary_hash="b1",
         toolchain=None) -> CampaignArm:
    return CampaignArm(
        name=name,
        manifest={"source_revision": source_revision, "variant_set": variant_set,
                  "manifest_hash": manifest_hash},
        descriptor={"descriptor_hash": descriptor_hash},
        candidate_names=frozenset(candidates),
        binary_hash=binary_hash,
        toolchain=toolchain if toolchain is not None else {},
    )


class ParityTests(unittest.TestCase):
    def test_identical_arms_pass_with_empty_allowed_differences(self):
        report = check_parity(_arm("legacy"), _arm("new"), label="tune",
                               allowed_differences=frozenset())
        self.assertEqual(report.missing_candidates, frozenset())
        self.assertEqual(report.extra_candidates, frozenset())

    def test_undeclared_manifest_difference_rejects(self):
        with self.assertRaises(ParityError):
            check_parity(_arm("legacy"), _arm("new", manifest_hash="m2"), label="tune")

    def test_declared_difference_is_allowed(self):
        report = check_parity(
            _arm("legacy"), _arm("new", manifest_hash="m2"), label="tune",
            allowed_differences=frozenset({"manifest.manifest_hash"}))
        self.assertEqual(report.label, "tune")

    def test_candidate_set_equal_size_but_different_membership_rejects(self):
        # The exact scenario RE09's replay-full investigation found: same
        # count, different membership. A count-only check would miss this.
        left = _arm("legacy", candidates=("mmq:a:v1", "mmq:b:v1"))
        right = _arm("new", candidates=("mmq:a:v1", "mmq:c:v1"))
        with self.assertRaises(ParityError) as ctx:
            check_parity(left, right, label="candidates")
        message = str(ctx.exception)
        self.assertIn("mmq:b:v1", message)
        self.assertIn("mmq:c:v1", message)

    def test_candidate_set_difference_can_be_explicitly_allowed(self):
        left = _arm("legacy", candidates=("mmq:a:v1", "mmq:b:v1"))
        right = _arm("new", candidates=("mmq:a:v1",))
        report = check_parity(left, right, label="candidates",
                               allowed_differences=frozenset({"candidate_names"}))
        self.assertEqual(report.missing_candidates, frozenset({"mmq:b:v1"}))
        self.assertEqual(report.extra_candidates, frozenset())

    def test_binary_hash_difference_rejects(self):
        with self.assertRaises(ParityError):
            check_parity(_arm("legacy"), _arm("new", binary_hash="b2"), label="tune")

    def test_toolchain_difference_rejects_even_with_binary_hash_allowed(self):
        # The actual gap this closes: a caller that always allows
        # binary_hash to differ (expected -- embedded build-tree paths)
        # must still catch a REAL divergence, like a different compiler or
        # ROCm version, that binary_hash alone can't distinguish from the
        # benign case.
        left = _arm("legacy", binary_hash="b1", toolchain={"rocm_version": "7.2.4"})
        right = _arm("new", binary_hash="b2", toolchain={"rocm_version": "7.1.0"})
        with self.assertRaises(ParityError) as ctx:
            check_parity(left, right, label="tune", allowed_differences=frozenset({"binary_hash"}))
        self.assertIn("toolchain", str(ctx.exception))

    def test_matching_toolchain_passes_alongside_allowed_binary_difference(self):
        left = _arm("legacy", binary_hash="b1", toolchain={"rocm_version": "7.2.4"})
        right = _arm("new", binary_hash="b2", toolchain={"rocm_version": "7.2.4"})
        report = check_parity(left, right, label="tune", allowed_differences=frozenset({"binary_hash"}))
        self.assertEqual(report.label, "tune")

    def test_toolchain_difference_can_be_explicitly_allowed(self):
        # The deliberate cross-version comparison use case: two BuildPlans
        # run against different ROCm installs on purpose.
        left = _arm("legacy", toolchain={"rocm_version": "7.2.4"})
        right = _arm("new", toolchain={"rocm_version": "7.1.0"})
        report = check_parity(left, right, label="tune", allowed_differences=frozenset({"toolchain"}))
        self.assertEqual(report.label, "tune")


if __name__ == "__main__":
    unittest.main()
