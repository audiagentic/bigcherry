from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bigcherry import release_validate  # noqa: E402


class SafeNameTests(unittest.TestCase):
    def test_safe_name_cannot_escape_staging_root(self):
        self.assertEqual(release_validate.safe_name("../../etc/passwd"), "etc-passwd")
        self.assertEqual(release_validate.safe_name("b10362"), "b10362")
        self.assertEqual(release_validate.safe_name("..."), "upstream")


class ReleaseGateTests(unittest.TestCase):
    def _evidence(self) -> dict:
        stages = {
            name: {"state": "validated", "ok": True,
                   "evidence": [f"{name}.json"]}
            for name in release_validate.PRODUCTION_GATE_STAGES
        }
        return {
            "claim": "validated",
            "architectures": ["gfx1201"],
            "required_architectures": ["gfx1201"],
            "production_gate": {
                "state": "validated",
                "required_architectures": ["gfx1201"],
                "stages": stages,
            },
            "architecture_coverage": {
                "required": ["gfx1201"],
                "observed": ["gfx1201"],
                "validated": ["gfx1201"],
                "by_architecture": {
                    "gfx1201": {
                        "observed": True, "validated": True,
                        "candidate_coverage": True,
                    },
                },
            },
            "candidate_coverage": {
                "variant_set": "workload-max",
                "observed_types": ["q8_0"],
                "by_type": {
                    "q8_0": {"observed": True, "candidate_count": 2,
                              "alternative_count": 1},
                },
            },
        }

    def test_compatibility_record_does_not_need_hardware_evidence(self):
        release_validate.validate_release_claim({"outcome": "compatible"})

    def test_validated_claim_requires_and_accepts_consistent_evidence(self):
        release_validate.validate_release_claim(self._evidence())

    def test_validated_claim_rejects_missing_architecture_evidence(self):
        record = self._evidence()
        del record["architecture_coverage"]
        with self.assertRaisesRegex(ValueError, "architecture_coverage"):
            release_validate.validate_release_claim(record)

    def test_validated_claim_rejects_mismatched_candidate_coverage(self):
        record = self._evidence()
        record["candidate_coverage"]["observed_types"].append("q6_k")
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            release_validate.validate_release_claim(record)

    def test_validated_claim_rejects_unvalidated_architecture(self):
        record = self._evidence()
        record["architecture_coverage"]["validated"] = []
        record["architecture_coverage"]["by_architecture"]["gfx1201"]["validated"] = False
        with self.assertRaisesRegex(ValueError, "gfx1201"):
            release_validate.validate_release_claim(record)

    def test_optimized_claim_rejects_missing_required_architecture(self):
        record = self._evidence()
        record["architectures"] = ["gfx1100", "gfx1201"]
        record["required_architectures"] = ["gfx1100", "gfx1201"]
        record["production_gate"]["required_architectures"] = ["gfx1100", "gfx1201"]
        record["architecture_coverage"]["required"] = ["gfx1100", "gfx1201"]
        with self.assertRaisesRegex(ValueError, "missing"):
            release_validate.validate_release_claim(record)

    def test_inventory_legacy_flat_architecture_map_remains_diagnostic(self):
        record = self._evidence()
        record["candidate_coverage"]["variant_set"] = "inventory"
        record["architecture_coverage"] = {
            "gfx1201": {"status": "observed", "candidate_coverage": True},
        }
        del record["required_architectures"]
        release_validate.validate_release_claim(record)

    def test_optimized_claim_rejects_legacy_flat_architecture_map(self):
        record = self._evidence()
        record["architecture_coverage"] = {
            "gfx1201": {"status": "validated", "candidate_coverage": True},
        }
        with self.assertRaisesRegex(ValueError, "explicit"):
            release_validate.validate_release_claim(record)

    def test_optimized_claim_rejects_zero_alternatives(self):
        record = self._evidence()
        coverage = record["candidate_coverage"]
        coverage["by_type"]["q8_0"]["alternative_count"] = 0
        coverage["by_type"]["q8_0"]["zero_alternative_reason"] = (
            "no supported alternative was generated for this type")
        with self.assertRaisesRegex(ValueError, "zero alternatives"):
            release_validate.validate_release_claim(record)

    def test_inventory_native_only_profile_remains_diagnostic(self):
        record = self._evidence()
        coverage = record["candidate_coverage"]
        coverage["variant_set"] = "inventory"
        coverage["by_type"]["q8_0"]["alternative_count"] = 0
        coverage["by_type"]["q8_0"]["native_count"] = 1
        coverage["by_type"]["q8_0"]["zero_alternative_reason"] = (
            "inventory profile contains native wrappers only")
        release_validate.validate_release_claim(record)

    def test_validated_claim_rejects_missing_variant_set(self):
        record = self._evidence()
        del record["candidate_coverage"]["variant_set"]
        with self.assertRaisesRegex(ValueError, "variant_set"):
            release_validate.validate_release_claim(record)

    def test_validated_claim_rejects_malformed_supported_coverage(self):
        record = self._evidence()
        record["supported_coverage"] = {"schema_version": 2}
        with self.assertRaisesRegex(ValueError, "supported_coverage"):
            release_validate.validate_release_claim(record)

    def test_validated_claim_rejects_incomplete_production_stage(self):
        record = self._evidence()
        record["production_gate"]["stages"]["tune"] = {
            "state": "prepared", "ok": False, "evidence": ["tune.json"]}
        with self.assertRaisesRegex(ValueError, "incomplete stage"):
            release_validate.validate_release_claim(record)

    def test_validated_claim_rejects_missing_production_stage(self):
        record = self._evidence()
        del record["production_gate"]["stages"]["replay"]
        with self.assertRaisesRegex(ValueError, "every required production stage"):
            release_validate.validate_release_claim(record)


def _git(root: Path, *args: str) -> str:
    import subprocess
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _init_bare_origin(directory: Path) -> tuple[Path, str]:
    repo = directory / "origin"
    _git(directory, "init", str(repo))
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    (repo / "source.txt").write_text("one\n", encoding="utf-8")
    _git(repo, "add", "source.txt")
    _git(repo, "commit", "-m", "initial")
    revision = _git(repo, "rev-parse", "HEAD")
    return repo, revision


def _init_project(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _git(directory, "init", str(directory))
    _git(directory, "config", "user.email", "test@example.invalid")
    _git(directory, "config", "user.name", "Test")
    (directory / ".gitkeep").write_text("", encoding="utf-8")
    _git(directory, "add", ".gitkeep")
    _git(directory, "commit", "-m", "initial")


_RECIPES_TOML = """\
version = 2
pinned = "unused"

[source.test-source]
ref = "pinned"
overlay = false
patch-sets = []

[source.test-source-full]
ref = "pinned"
overlay = false
patch-sets = []

[build.stock]
options = {}
needs = []

[platform.linux-multi]
targets = ["gfx1100"]
options = {}

[build.needs-inventory]
options = {}
needs = ["inventory"]

[build.needs-replay]
options = {}
needs = ["inventory", "promoted-winners"]

[compat.recipe.test-source]
ref = "pinned"
builds = ["stock"]
platform = "linux-multi"

[compat.recipe.test-source-full]
ref = "pinned"
builds = ["stock", "needs-inventory", "needs-replay"]
platform = "linux-multi"
"""


class _ProbeHarness:
    """RE13: probe() now runs execute_campaign_lane() for real (materialize,
    BuildPlan identity, ArtifactStore) against an isolated ProjectContext --
    only campaign_workers.subprocess.run (the actual compiler) and
    ProjectContext.resolve (so the probe's shared-upstream-mirror lookup
    lands in this fixture, never the real host) are faked, matching
    test_campaign_cutover_audit.py's harness pattern.
    """

    def __init__(self, root: Path):
        self.root = root
        self.origin, self.origin_revision = _init_bare_origin(root)
        self.project = root / "project"
        _init_project(self.project)
        (self.project / "recipes.toml").write_text(_RECIPES_TOML, encoding="utf-8")
        (self.project / "patches").mkdir()
        (self.project / "patches" / ".gitkeep").write_text("", encoding="utf-8")
        _git(self.project, "add", "recipes.toml", "patches/.gitkeep")
        _git(self.project, "commit", "-m", "recipes")
        self.mirror = root / "mirror"
        _git(root, "clone", str(self.origin), str(self.mirror))

    def patches(self, calls: list | None = None):
        from bigcherry.context import ProjectContext

        calls = calls if calls is not None else []
        real_resolve = ProjectContext.resolve.__func__

        def fake_resolve(cls, *, project_root=None, config_path=None,
                         artifacts_root=None, work_root=None, upstream_repo=None):
            return real_resolve(
                cls, project_root=project_root or self.project,
                config_path=config_path, artifacts_root=artifacts_root,
                work_root=work_root or (self.root / "default-work"),
                upstream_repo=upstream_repo or self.mirror,
            )

        return (
            mock.patch.object(ProjectContext, "resolve", classmethod(fake_resolve)),
            mock.patch("bigcherry.campaign_workers.subprocess.run", _fake_compiler(calls)),
        )


def _fake_compiler(calls: list):
    import subprocess as _subprocess
    real_run = _subprocess.run

    def run(args, cwd=None, check=None, **kwargs):
        argv = args if isinstance(args, (list, tuple)) else [args]
        if not argv or Path(str(argv[0])).name.split(".")[0] not in ("cmake",):
            return real_run(args, cwd=cwd, check=check, **kwargs)
        calls.append(args)
        if "--build" in argv:
            build_dir = Path(argv[2])
            (build_dir / "bin").mkdir(parents=True, exist_ok=True)
            (build_dir / "bin" / "llama-bench").write_bytes(b"launcher-bytes")
        else:
            build_dir = Path(argv[argv.index("-B") + 1]) if "-B" in argv else None
            if build_dir is not None:
                build_dir.mkdir(parents=True, exist_ok=True)
                (build_dir / "CMakeCache.txt").write_text(
                    "CMAKE_C_COMPILER:FILEPATH=/opt/rocm/llvm/bin/clang\n"
                    "AMDGPU_TARGETS:STRING=gfx1100\nGGML_HIP:BOOL=ON\n", encoding="utf-8")
        return _subprocess.CompletedProcess(args, 0, stdout="", stderr="")
    return run


class ProbeTests(unittest.TestCase):
    def test_run_already_exists_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            (staging / "dup").mkdir()
            with self.assertRaises(FileExistsError):
                release_validate.probe("dup", staging, "master", "test-source")

    def test_unknown_recipe_is_a_config_error_not_a_build_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness = _ProbeHarness(root)
            patches = harness.patches()
            with patches[0], patches[1]:
                code, path = release_validate.probe(
                    "r0", root / "staging", "HEAD", "no-such-recipe")
            self.assertEqual(code, 1)
            record = path.read_text(encoding="utf-8")
            self.assertIn('"outcome": "config-error"', record)

    def test_fetch_failure_short_circuits_before_any_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness = _ProbeHarness(root)
            patches = harness.patches()
            with patches[0], patches[1]:
                code, path = release_validate.probe(
                    "r1", root / "staging", "no-such-ref", "test-source")
            self.assertEqual(code, 1)
            record = path.read_text(encoding="utf-8")
            self.assertIn('"outcome": "pull-failed"', record)
            self.assertIn('"failure_class": "pull-failed"', record)
            self.assertIn('"stage": "pull"', record)

    def test_clean_probe_reports_compatible_and_runs_the_real_lane(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness = _ProbeHarness(root)
            calls: list = []
            patches = harness.patches(calls)
            with patches[0], patches[1]:
                code, path = release_validate.probe(
                    "r3", root / "staging", "HEAD", "test-source")
            self.assertEqual(code, 0)
            self.assertTrue(calls, "expected the fake cmake/compiler to have been invoked")
            record = path.read_text(encoding="utf-8")
            self.assertIn('"outcome": "compatible"', record)
            self.assertIn(harness.origin_revision, record)

    def test_build_failure_is_reported_with_its_own_detail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness = _ProbeHarness(root)
            patches = harness.patches()

            import subprocess as _subprocess
            real_run = _subprocess.run

            def failing_run(args, cwd=None, check=None, **kwargs):
                argv = args if isinstance(args, (list, tuple)) else [args]
                if argv and Path(str(argv[0])).name.split(".")[0] == "cmake":
                    raise OSError("compiler not found")
                return real_run(args, cwd=cwd, check=check, **kwargs)

            with patches[0], mock.patch(
                "bigcherry.campaign_workers.subprocess.run", side_effect=failing_run):
                code, path = release_validate.probe(
                    "r2", root / "staging", "HEAD", "test-source")
            self.assertEqual(code, 1)
            record = path.read_text(encoding="utf-8")
            self.assertIn('"outcome": "patch-drift-or-build-failed"', record)
            self.assertIn('"failure_class": "patch-drift"', record)
            self.assertIn('"build": "stock"', record)

    def test_builds_needing_an_unavailable_input_are_skipped_not_reported_as_drift(self):
        # GPT-auto-agent review (RE13 follow-up, 2026-08-17): the shipped
        # default recipe includes tune/replay-shaped builds needing
        # inventory/promoted-winners, neither of which is meaningful to
        # synthesize for an arbitrary probed ref. Those builds must be
        # skipped, not attempted-and-misreported as patch-drift.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness = _ProbeHarness(root)
            patches = harness.patches()
            with patches[0], patches[1]:
                code, path = release_validate.probe(
                    "r5", root / "staging", "HEAD", "test-source-full")
            self.assertEqual(code, 0)
            record = path.read_text(encoding="utf-8")
            # GPT-auto-agent review (RE13 follow-up): "compatible-partial",
            # not unqualified "compatible" -- a skipped build's compile
            # path was never actually exercised.
            self.assertIn('"outcome": "compatible-partial"', record)
            self.assertIn('"skipped": true', record)
            self.assertIn("missing required input", record)
            self.assertNotIn("patch-drift-or-build-failed", record)

    def test_supplying_inventory_lets_the_dependent_build_run_for_real(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness = _ProbeHarness(root)
            inventory = root / "inventory.json"
            inventory.write_text("{}", encoding="utf-8")
            calls: list = []
            patches = harness.patches(calls)
            with patches[0], patches[1]:
                code, path = release_validate.probe(
                    "r6", root / "staging", "HEAD", "test-source-full",
                    inventory=inventory)
            self.assertEqual(code, 0)
            record = path.read_text(encoding="utf-8")
            # needs-replay is still skipped (no promoted-winners supplied),
            # so this remains "compatible-partial", not unqualified
            # "compatible".
            self.assertIn('"outcome": "compatible-partial"', record)
            # needs-inventory actually ran (has a build_plan_id, not skipped);
            # needs-replay still skipped (no promoted-winners supplied).
            self.assertIn('"needs-inventory"', record)
            self.assertIn('"build_plan_id"', record)
            self.assertIn('"skipped": true', record)


if __name__ == "__main__":
    unittest.main()
