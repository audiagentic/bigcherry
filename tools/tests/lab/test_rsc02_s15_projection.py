"""RSC02 S15: fail-closed semantic launch comparison, without GPU hardware.

These tests operate on the actual strict-v2 contract's helper fixture.
The integration dry-run CTest invokes both real launch owners with fake engines.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools/lab/radiance"))
from rsc02_s15_projection import (
    V2_EXAMPLE,
    ProjectionError,
    _fixture,
    compare,
    projection,
)


class SemanticProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original = json.loads(V2_EXAMPLE.read_text(encoding="utf-8"))
        self.a = _fixture(copy.deepcopy(self.original), "/fake/a/manifest.json")
        self.b = _fixture(copy.deepcopy(self.original), "/fake/b/manifest.json")

    def assert_rejects_change(self, updater) -> None:
        different = copy.deepcopy(self.b)
        updater(different)
        report = compare(self.a, different, verify_bytes=False)
        self.assertFalse(report["equal"], report)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(report["different_paths"])

    def test_dry_run_interprets_deliberately_missing_helper_as_rejection(self) -> None:
        # The real CTest still invokes the owner; only the OS subprocess is
        # faked here to prove that deliberate failure injection can succeed.
        from unittest import mock
        from rsc02_s15_projection import dry_run

        with mock.patch("rsc02_s15_projection.subprocess.run") as run:
            run.return_value.returncode = 0
            self.assertEqual(dry_run(Path("/fake-native-owner")), 0)
            run.assert_called_once()

    def test_fixture_observes_exact_manifest_helper_roles(self) -> None:
        expected = ["engine"] + [
            helper["role"] for helper in self.original["spec"]["processes"]["helpers"]
        ]
        actual = [row["role"] for row in self.a["processes"]["roles"]]
        self.assertEqual(actual, expected)
        projection(self.a, verify_bytes=False)

    def test_ignores_only_explicit_per_session_bookkeeping(self) -> None:
        other = self.b["manifest"]
        other["metadata"]["instance_id"] = "nonce"
        other["metadata"]["content_digest"] = "untrusted unit test bytes"
        other["provenance"] = {}
        other["spec"]["facts"]["host"]["memory"]["available_mib"] = 2
        for card in other["spec"]["facts"]["gpus"]:
            card["vram"]["available_mib"] = 1
        other["spec"]["processes"]["engine"]["server"]["port"] += 1
        other["spec"]["handoff"]["transport"]["shm"] = "/different-session"
        other["spec"]["handoff"]["telemetry"]["histogram_file"] = "/new-histogram"
        report = compare(self.a, self.b, verify_bytes=False)
        self.assertTrue(report["equal"], report)
        self.assertEqual(report["legacy_projection_sha256"], report["native_projection_sha256"])

    def test_live_memory_readings_are_observations_but_inputs_are_not(self) -> None:
        # S15 dense pair 1 (hardware): the three readings below differed between two
        # starts of the same config; every derived value and the greedy output agreed.
        inputs = self.b["manifest"]["spec"]["plan"]["resource_inputs"]
        for key in ("cgroup_headroom_bytes", "mem_available_bytes", "pin_available_bytes"):
            inputs[key] += 4096
        report = compare(self.a, self.b, verify_bytes=False)
        self.assertTrue(report["equal"], report)
        # A real input (what the plan must hold) still fails the comparison.
        self.assert_rejects_change(
            lambda snap: snap["manifest"]["spec"]["plan"]["resource_inputs"].__setitem__(
                "checkpoint_bytes", snap["manifest"]["spec"]["plan"]["resource_inputs"]["checkpoint_bytes"] + 1))

    def test_pinned_manifest_descriptor_snapshot_is_verified_at_capture(self) -> None:
        # A stored-manifest replay gives children `--manifest /proc/self/fd/N`: the path only
        # exists inside the child, so the capture verifies the bytes live and records that.
        pinned = _fixture(copy.deepcopy(self.original), "/proc/self/fd/5")
        pinned["manifest_bytes_verified_live"] = True
        pinned["manifest_read_path"] = "/proc/0/fd/5"      # the child is gone by compare time
        pinned["manifest_file_sha256"] = "0" * 64
        projection(pinned)                                  # must not re-read a vanished descriptor
        pinned.pop("manifest_bytes_verified_live")          # without the live proof it must read the path
        with self.assertRaises(Exception):
            projection(pinned)

    def test_refuses_semantic_changes_and_missing_roles(self) -> None:
        self.assert_rejects_change(
            lambda s: s["manifest"]["spec"]["processes"]["engine"]["execution"].update(
                num_speculative_tokens=s["manifest"]["spec"]["processes"]["engine"]["execution"][
                    "num_speculative_tokens"
                ] + 1
            )
        )
        self.assert_rejects_change(
            lambda s: s["manifest"]["spec"]["resources"]["host"].update(pool_mib=1)
        )
        self.assert_rejects_change(
            lambda s: s["manifest"]["spec"]["handoff"]["telemetry"].update(period_s=17)
        )
        self.assert_rejects_change(
            lambda s: s["manifest"]["spec"]["intent"].update(model="wrong.rad")
        )
        self.assert_rejects_change(
            lambda s: s["manifest"]["spec"]["intent"]["power"].update(profile="unsafe")
        )
        missing_manifest_helper = copy.deepcopy(self.b)
        missing_manifest_helper["manifest"]["spec"]["processes"]["helpers"].pop()
        with self.assertRaisesRegex(ProjectionError, "unexpected observed roles"):
            compare(self.a, missing_manifest_helper, verify_bytes=False)
        self.assert_rejects_change(
            lambda s: s["manifest"]["spec"]["plan"]["owner_map"].update(owner_map_sha256="changed")
        )
        self.assert_rejects_change(
            lambda s: s["manifest"]["spec"]["handoff"]["placement"].update(heat_file="/wrong")
        )
        self.assert_rejects_change(
            lambda s: s["processes"]["roles"][0]["argv"].append("--unexpected-argument")
        )
        self.assert_rejects_change(
            lambda s: s["processes"]["roles"][0]["env"].update(HIP_VISIBLE_DEVICES="1")
        )
        missing_observation = copy.deepcopy(self.b)
        missing_observation["processes"]["roles"].pop()
        with self.assertRaisesRegex(ProjectionError, "missing observed roles"):
            compare(self.a, missing_observation, verify_bytes=False)

    def test_refuses_unsupported_argv_and_duplicate_roles(self) -> None:
        bad = copy.deepcopy(self.b)
        bad["processes"]["roles"][0]["argv"] += ["--manifest", "/another"]
        with self.assertRaisesRegex(ProjectionError, "exactly one"):
            projection(bad, verify_bytes=False)
        bad = copy.deepcopy(self.b)
        bad["processes"]["roles"].append(copy.deepcopy(bad["processes"]["roles"][0]))
        with self.assertRaisesRegex(ProjectionError, "duplicate process role"):
            projection(bad, verify_bytes=False)
        bad = copy.deepcopy(self.b)
        bad["manifest"]["spec"]["facts"]["gpus"][1]["bdf"] = (
            bad["manifest"]["spec"]["facts"]["gpus"][0]["bdf"]
        )
        with self.assertRaisesRegex(ProjectionError, "duplicate physical card"):
            projection(bad, verify_bytes=False)

    def test_refuses_matching_but_incomplete_observed_roles(self) -> None:
        # Two arms agreeing is not enough: both must show every process
        # declared by the manifest, with its actual --role argument.
        missing = copy.deepcopy(self.a)
        missing["processes"]["roles"].pop()
        with self.assertRaisesRegex(ProjectionError, "missing observed roles"):
            compare(missing, copy.deepcopy(missing), verify_bytes=False)

        extra = copy.deepcopy(self.a)
        extra["processes"]["roles"].append(
            {"role": "ghost", "argv": ["r3_moe_worker", "--manifest",
                                     extra["manifest_path"], "--role", "ghost"],
             "env": {"ROCR_VISIBLE_DEVICES": "3"}}
        )
        with self.assertRaisesRegex(ProjectionError, "unexpected observed roles"):
            compare(extra, copy.deepcopy(extra), verify_bytes=False)

        forged = copy.deepcopy(self.a)
        forged["processes"]["roles"][0]["argv"][-1] = "helper0"
        with self.assertRaisesRegex(ProjectionError, "observed --role"):
            compare(forged, copy.deepcopy(forged), verify_bytes=False)

        omitted = copy.deepcopy(self.a)
        omitted["processes"]["roles"][0]["argv"] = (
            omitted["processes"]["roles"][0]["argv"][:-2]
        )
        with self.assertRaisesRegex(ProjectionError, "exactly one --role"):
            compare(omitted, copy.deepcopy(omitted), verify_bytes=False)

    def test_requires_manifest_bytes_and_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "captured.json"
            encoded = json.dumps(self.original, sort_keys=True).encode()
            path.write_bytes(encoded)
            a = _fixture(copy.deepcopy(self.original), str(path))
            a["manifest_file_sha256"] = hashlib.sha256(encoded).hexdigest()
            projection(a)
            a["manifest"]["spec"]["intent"]["model"] = "tampered"
            with self.assertRaisesRegex(ProjectionError, "disagree"):
                projection(a)
            a = _fixture(copy.deepcopy(self.original), str(path))
            a["manifest_file_sha256"] = "0" * 64
            with self.assertRaisesRegex(ProjectionError, "disagree"):
                projection(a)


if __name__ == "__main__":
    unittest.main()
