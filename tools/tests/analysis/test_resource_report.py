import json
import tempfile
import unittest
from pathlib import Path

from bigcherry.analysis import resource_report


def _remark(body: str, line: int) -> str:
    return f"kernel.cu:{line}:1: remark: {body} [-Rpass-analysis=kernel-resource-usage]\n"


def _raw(symbol="_Z6kernelv", scratch=0, occupancy=5, lds=0, sgpr_spill=0, omit_lds=False):
    values = [
        f"Function Name: {symbol}",
        "TotalSGPRs: 24", "VGPRs: 32", "AGPRs: 0",
        f"ScratchSize [bytes/lane]: {scratch}", "Dynamic Stack: False",
        f"Occupancy [waves/SIMD]: {occupancy}",
        f"SGPRs Spill: {sgpr_spill}", "VGPRs Spill: 0",
    ]
    if not omit_lds:
        values.append(f"LDS Size [bytes/block]: {lds}")
    return "".join(_remark(value, index + 1) for index, value in enumerate(values))


class ResourceReportTests(unittest.TestCase):
    def test_supported_clang_report_resolves_and_excludes_spill(self):
        policy = resource_report.ResourcePolicyV1("gfx1100-spill-v1", "gfx1100")
        report = resource_report.build_report(
            _raw(scratch=16).encode(), compiler_family="clang", compiler_major=21,
            compiler_version="21.0.0git", architecture="gfx1100",
            source_revision="a" * 40, manifest_hash="b" * 32,
            symbol_map={"_Z6kernelv": ["mmq:q8_0:j64:fb0:v1"]}, policy=policy)
        self.assertTrue(report["recognized_schema"])
        self.assertEqual(report["resolution_counts"]["resolved"], 1)
        self.assertEqual(report["exclusions"][0]["reasons"], ["scratch"])

    def test_low_occupancy_is_advisory_not_exclusion(self):
        policy = resource_report.ResourcePolicyV1(
            "gfx1100-spill-v1", "gfx1100", warn_occupancy_lt=2)
        report = resource_report.build_report(
            _raw(occupancy=1).encode(), compiler_family="clang", compiler_major=21,
            compiler_version="21", architecture="gfx1100",
            source_revision="a" * 40, manifest_hash="b" * 32,
            symbol_map={"_Z6kernelv": ["mmq:q8_0:j64:fb0:v1"]}, policy=policy)
        self.assertEqual(report["exclusions"], [])
        self.assertEqual(report["advisories"][0]["reason"], "low_occupancy")

    def test_unknown_compiler_or_shape_never_yields_empty_success(self):
        policy = resource_report.ResourcePolicyV1("p", "gfx1100")
        for major, raw in ((20, _raw().encode()), (21, b"ordinary compiler output")):
            report = resource_report.build_report(
                raw, compiler_family="clang", compiler_major=major,
                compiler_version=str(major), architecture="gfx1100",
                source_revision="a" * 40, manifest_hash="b" * 32,
                symbol_map={}, policy=policy)
            self.assertFalse(report["recognized_schema"])
            self.assertEqual(report["exclusions"], [])

    def test_ambiguous_or_missing_symbols_cannot_exclude(self):
        policy = resource_report.ResourcePolicyV1("p", "gfx1100")
        for symbol_map, status in (({}, "missing"),
                                   ({"_Z6kernelv": ["a", "b"]}, "ambiguous")):
            report = resource_report.build_report(
                _raw(scratch=64).encode(), compiler_family="clang", compiler_major=21,
                compiler_version="21", architecture="gfx1100",
                source_revision="a" * 40, manifest_hash="b" * 32,
                symbol_map=symbol_map, policy=policy)
            self.assertTrue(report["recognized_schema"])
            self.assertEqual(report["resolution_counts"][status], 1)
            self.assertEqual(report["exclusions"], [])

    def test_missing_lds_fails_closed_under_an_active_lds_threshold(self):
        """gpt-dev-agent review, 2026-08-31: LDS is not in the parser's
        required-field set, so a block that omits the LDS remark line
        entirely still parses as "resolved". --reject-lds-gt is an ACTIVE
        threshold the caller explicitly asked for; a missing field must not
        silently skip it (the previous behavior: recognized_schema=true,
        no exclusion, exit 0 -- looked like a clean pass)."""
        policy = resource_report.ResourcePolicyV1(
            "gfx1100-lds-v1", "gfx1100", reject_lds_gt=0)
        report = resource_report.build_report(
            _raw(omit_lds=True).encode(), compiler_family="clang", compiler_major=21,
            compiler_version="21.0.0git", architecture="gfx1100",
            source_revision="a" * 40, manifest_hash="b" * 32,
            symbol_map={"_Z6kernelv": ["mmq:q8_0:j64:fb0:v1"]}, policy=policy)
        self.assertTrue(report["recognized_schema"])  # still a valid LLVM-21 block
        self.assertEqual(report["exclusions"][0]["reasons"], ["lds_unknown"])

    def test_missing_lds_is_fine_when_no_lds_threshold_is_active(self):
        policy = resource_report.ResourcePolicyV1("gfx1100-no-lds-v1", "gfx1100")
        report = resource_report.build_report(
            _raw(omit_lds=True).encode(), compiler_family="clang", compiler_major=21,
            compiler_version="21.0.0git", architecture="gfx1100",
            source_revision="a" * 40, manifest_hash="b" * 32,
            symbol_map={"_Z6kernelv": ["mmq:q8_0:j64:fb0:v1"]}, policy=policy)
        self.assertEqual(report["exclusions"], [])

    def test_report_is_deterministic_and_raw_hash_changes(self):
        policy = resource_report.ResourcePolicyV1("p", "gfx1100")
        kwargs = dict(
            compiler_family="clang", compiler_major=21, compiler_version="21",
            architecture="gfx1100", source_revision="a" * 40,
            manifest_hash="b" * 32,
            symbol_map={"_Z6kernelv": ["candidate"]}, policy=policy)
        one = resource_report.build_report(_raw().encode(), **kwargs)
        two = resource_report.build_report(_raw().encode(), **kwargs)
        changed = resource_report.build_report((_raw() + "\n").encode(), **kwargs)
        self.assertEqual(one, two)
        self.assertNotEqual(one["raw_report_hash"], changed["raw_report_hash"])

    def test_blacklist_loader_requires_recognized_report_and_preserves_reason(self):
        policy = resource_report.ResourcePolicyV1("p", "gfx1100")
        report = resource_report.build_report(
            _raw(scratch=16).encode(), compiler_family="clang", compiler_major=21,
            compiler_version="21", architecture="gfx1100",
            source_revision="a" * 40, manifest_hash="b" * 32,
            symbol_map={"_Z6kernelv": ["mmvq:q8_0:w1:nw1:rpb1:sk0:v1"]},
            policy=policy)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resource-report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(
                resource_report.load_blacklist(path),
                {("mmvq:q8_0:w1:nw1:rpb1:sk0:v1", "gfx1100"): ("scratch",)},
            )
            report["recognized_schema"] = False
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(resource_report.ResourceError):
                resource_report.load_blacklist(path)


if __name__ == "__main__":
    unittest.main()
