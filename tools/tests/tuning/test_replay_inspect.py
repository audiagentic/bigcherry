"""HI15/HI16 replay-inspect wrapper tests.

The C++ hip-autotune-inspect binary links the real ggml-hip loader, so its
judgements cannot be faked in-process; these tests run the same parse and
reconcile path against a scripted stand-in (a python script that emits the
tool's --json contract with a chosen exit code), keeping the contract
verifiable on a machine with no ROCm build. The C++ side of the contract is
pinned by the real build on the campaign host.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry import __main__ as bc_main  # noqa: E402
from bigcherry.tuning import replay_inspect # noqa: E402
from bigcherry.tuning.catalog import build_descriptor  # noqa: E402


def _python_executable() -> str:
    """Return a launchable interpreter even when sys.executable is an alias."""
    if Path(sys.executable).is_file():
        return sys.executable
    return shutil.which("python") or sys.executable


def _inventory_manifest() -> dict:
    """A minimal inventory-profile manifest: one native wrapper per family."""
    families = ("mmvq", "mmq", "mmvf", "mmf", "blas")
    manifest = {
        "source_revision": "b" * 40,
        "variant_set": "inventory",
        "manifest_hash": "a" * 32,
        "producer_capabilities": "0" * 32,
        "architectures": ["gfx1100"],
        "summary": {
            "total": len(families),
            "by_family": dict.fromkeys(families, 1),
            "by_source_class": {"native_wrapper": len(families)},
        },
        "candidates": [
            {
                "stable_name": f"{f}:native:v1",
                "family": f,
                "source_class": "native_wrapper",
                "implementation_version": 1,
                "config": {},
            }
            for f in families
        ],
    }
    descriptor = build_descriptor(manifest)
    manifest["build_descriptor"] = descriptor
    return manifest


def _build_block(manifest: dict) -> dict:
    descriptor = build_descriptor(manifest)
    return {
        "manifest_hash": descriptor["manifest_hash"],
        "source_revision": descriptor["source_revision"],
        "descriptor_hash": descriptor["descriptor_hash"],
        "variant_set": descriptor["variant_set"],
        "artifact_version": 1,
        "candidate_count": descriptor["candidate_count"],
    }


def _registry_block(manifest: dict) -> dict:
    descriptor = build_descriptor(manifest)
    return {
        "count": descriptor["candidate_count"],
        "by_family": dict(descriptor["by_family"]),
        "by_source_class": dict(descriptor["by_source_class"]),
        "anomalies": [],
    }


FAKE_TOOL = """\
import json
import sys
from pathlib import Path

# The scenario file carries both the exit code and the --json report, so one
# scripted stand-in exercises every branch of the real tool's contract.
# The cache argument is optional (registry-only mode); the scenario name is
# derived from it with the same with_suffix rule the tests use to write it.
args = [a for a in sys.argv[1:] if a != "--json"]
if args:
    scenario_path = Path(args[-1]).with_suffix(".scenario")
else:
    scenario_path = Path("scenario.json")
with open(scenario_path, encoding="utf-8") as handle:
    scenario = json.load(handle)
print(json.dumps(scenario["report"]))
sys.exit(scenario.get("exit", 0))
"""


class ReplayInspectToolTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.fake = self.root / "fake-inspect"
        self.fake.write_text(FAKE_TOOL, encoding="utf-8")
        self.cache = self.root / "cache.bin"
        self.cache.write_bytes(b"\x42\x43\x48\x59")

    def _scenario(self, report: dict, exit_code: int = 0) -> None:
        self.cache.with_suffix(".scenario").write_text(
            json.dumps({"exit": exit_code, "report": report}), encoding="utf-8"
        )

    def _run(self, report: dict, exit_code: int = 0):
        self._scenario(report, exit_code)
        return replay_inspect.run_tool(
            self.fake, cache=self.cache, interpreter=[_python_executable()]
        )

    def test_happy_path_returns_report_with_exit(self):
        manifest = _inventory_manifest()
        report = {
            "build": _build_block(manifest),
            "registry": _registry_block(manifest),
            "cache": {
                "path": str(self.cache),
                "outcome": "loaded",
                "stale": False,
                "winner_slots": 1,
                "usable": 1,
                "entries": [
                    {
                        "dispatch": "c" * 32,
                        "winner": "mmq:native:v1",
                        "registered": True,
                        "fresh": True,
                        "stale_impl": False,
                        "unrecognized_match": False,
                        "generation": 0,
                        "transform_id": 0,
                        "match_kind": 0,
                    }
                ],
            },
        }
        parsed = self._run(report)
        self.assertEqual(parsed["_exit"], 0)
        self.assertEqual(parsed["cache"]["outcome"], "loaded")
        self.assertEqual(parsed["cache"]["entries"][0]["winner"], "mmq:native:v1")

    def test_cache_rejected_is_reportable(self):
        manifest = _inventory_manifest()
        report = {
            "build": _build_block(manifest),
            "registry": _registry_block(manifest),
            "cache": {
                "path": str(self.cache),
                "outcome": "rerun_required",
                "stale": False,
                "winner_slots": 0,
                "usable": 0,
                "entries": [],
            },
        }
        parsed = self._run(report, exit_code=replay_inspect.EXIT_CACHE_REJECTED)
        self.assertEqual(parsed["_exit"], replay_inspect.EXIT_CACHE_REJECTED)
        self.assertEqual(parsed["cache"]["outcome"], "rerun_required")

    def test_registry_only_report_has_no_cache_key(self):
        manifest = _inventory_manifest()
        report = {
            "build": _build_block(manifest),
            "registry": _registry_block(manifest),
        }
        parsed = self._run(report)
        self.assertNotIn("cache", parsed)
        self.assertEqual(parsed["_exit"], 0)

    def test_usage_error_raises(self):
        self._scenario({}, exit_code=replay_inspect.EXIT_USAGE)
        with self.assertRaises(SystemExit):
            replay_inspect.run_tool(
                self.fake, cache=self.cache, interpreter=[_python_executable()]
            )

    def test_unparseable_output_raises(self):
        broken = self.root / "broken"
        broken.write_text(
            "import sys\nprint('not json')\nsys.exit(0)\n", encoding="utf-8"
        )
        with self.assertRaises(SystemExit):
            replay_inspect.run_tool(
                broken, cache=self.cache, interpreter=[_python_executable()]
            )

    def test_find_tool_explicit_path_wins(self):
        self.assertEqual(replay_inspect.find_tool(str(self.fake)), self.fake)

    def test_find_tool_env_variable(self):
        import os

        os.environ["BIGCHERRY_INSPECT_TOOL"] = str(self.fake)
        try:
            self.assertEqual(replay_inspect.find_tool(), self.fake)
        finally:
            del os.environ["BIGCHERRY_INSPECT_TOOL"]

    def test_find_tool_missing_raises_with_candidates(self):
        with self.assertRaises(SystemExit) as ctx:
            replay_inspect.find_tool(str(self.root / "nope"))
        self.assertIn("nope", str(ctx.exception))
        self.assertIn("BIGCHERRY_INSPECT_TOOL", str(ctx.exception))


class ManifestAgreementTests(unittest.TestCase):
    def setUp(self):
        self.manifest = _inventory_manifest()
        self.manifest_path = Path(tempfile.mkdtemp()) / "manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def _report(self, **overrides):
        report = {
            "build": _build_block(self.manifest),
            "registry": _registry_block(self.manifest),
        }
        report.update(overrides)
        return report

    def test_agreement_when_binary_matches_catalog(self):
        agreement = replay_inspect.manifest_agreement(
            self._report(), self.manifest_path
        )
        self.assertTrue(agreement["agrees"])
        self.assertTrue(agreement["by_family_agrees"])
        self.assertEqual(agreement["diffs"], [])
        self.assertTrue(agreement["manifest_embedded_descriptor_agrees"])

    def test_manifest_hash_drift_is_reported(self):
        build = _build_block(self.manifest)
        build["manifest_hash"] = "f" * 32
        agreement = replay_inspect.manifest_agreement(
            self._report(build=build), self.manifest_path
        )
        self.assertFalse(agreement["agrees"])
        fields = [d["field"] for d in agreement["diffs"]]
        self.assertIn("manifest_hash", fields)

    def test_descriptor_hash_drift_catches_registry_catalog_split(self):
        build = _build_block(self.manifest)
        build["descriptor_hash"] = "e" * 32
        agreement = replay_inspect.manifest_agreement(
            self._report(build=build), self.manifest_path
        )
        self.assertFalse(agreement["agrees"])
        self.assertIn("descriptor_hash", [d["field"] for d in agreement["diffs"]])

    def test_by_family_drift_is_reported(self):
        registry = _registry_block(self.manifest)
        registry["by_family"] = {"mmq": 2, "mmvq": 1, "mmvf": 1, "mmf": 1, "blas": 1}
        agreement = replay_inspect.manifest_agreement(
            self._report(registry=registry), self.manifest_path
        )
        self.assertFalse(agreement["by_family_agrees"])

    def test_embedded_descriptor_disagreement_is_flagged(self):
        import copy

        manifest = copy.deepcopy(self.manifest)
        manifest["build_descriptor"]["descriptor_hash"] = "0" * 32
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        agreement = replay_inspect.manifest_agreement(
            self._report(), self.manifest_path
        )
        self.assertTrue(
            agreement["agrees"], "the binary still matches the recomputed descriptor"
        )
        self.assertFalse(agreement["manifest_embedded_descriptor_agrees"])


class FormatAndCliTests(unittest.TestCase):
    def test_format_report_renders_all_sections(self):
        manifest = _inventory_manifest()
        report = {
            "build": _build_block(manifest),
            "registry": _registry_block(manifest),
            "manifest": replay_inspect.manifest_agreement(
                {
                    "build": _build_block(manifest),
                    "registry": _registry_block(manifest),
                },
                None,
            )
            if False
            else None,
        }
        text = replay_inspect.format_report(report)
        self.assertIn("hip-autotune-inspect", text)
        self.assertIn("candidate(s)", text)

    def test_format_report_renders_manifest_agreement(self):
        manifest = _inventory_manifest()
        manifest_path = Path(tempfile.mkdtemp()) / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        base = {"build": _build_block(manifest), "registry": _registry_block(manifest)}
        base["manifest"] = replay_inspect.manifest_agreement(base, manifest_path)
        text = replay_inspect.format_report(base)
        self.assertIn("agrees with the compiled registry", text)

    def test_cli_subcommand_is_registered(self):
        parser = bc_main.build_parser()
        # argparse stores subcommands privately; invoke parsing directly.
        namespace = parser.parse_args(
            ["replay-inspect", "x.cache", "--manifest", "m.json"]
        )
        self.assertEqual(namespace.cache, "x.cache")
        self.assertIs(namespace.json, False)
        self.assertIs(namespace.tool_interpreter, None)

    def test_cli_end_to_end_with_scripted_tool(self):
        """The full CLI path: parser -> find_tool -> run_tool -> exit code."""
        root = Path(tempfile.mkdtemp())
        fake = root / "fake-inspect"
        fake.write_text(FAKE_TOOL, encoding="utf-8")
        cache = root / "cache.bin"
        cache.write_bytes(b"stub")
        manifest = _inventory_manifest()
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        scenario = {
            "exit": replay_inspect.EXIT_CACHE_REJECTED,
            "report": {
                "build": _build_block(manifest),
                "registry": _registry_block(manifest),
                "cache": {
                    "path": str(cache),
                    "outcome": "rerun_required",
                    "stale": False,
                    "winner_slots": 0,
                    "usable": 0,
                    "entries": [],
                },
            },
        }
        cache.with_suffix(".scenario").write_text(
            json.dumps(scenario), encoding="utf-8"
        )
        completed = subprocess.run(
            [
                _python_executable(),
                "-m",
                "bigcherry",
                "replay-inspect",
                str(cache),
                "--manifest",
                str(manifest_path),
                "--tool",
                str(fake),
                "--tool-interpreter",
                _python_executable(),
                "--json",
            ],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode, replay_inspect.EXIT_CACHE_REJECTED, completed.stderr
        )
        report = json.loads(completed.stdout)
        self.assertEqual(report["_exit"], replay_inspect.EXIT_CACHE_REJECTED)
        self.assertEqual(report["cache"]["outcome"], "rerun_required")
        self.assertFalse(report["manifest"]["agrees"] is None)


if __name__ == "__main__":
    unittest.main()
