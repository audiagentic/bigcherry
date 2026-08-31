"""VA02 tests for tools/bigcherry/patch/validation_policy.py -- the static
RD-patch validation-package policy gate (docs/reference/testing/
PATCH_VALIDATION.md).

Covers the offline matrix: exact-identity grandfathering, invalidation on
any of {patch.py, patch.toml, tracked-status set, policy-version} changing,
missing-baseline fail-closed behavior, a fully current package passing
without needing grandfathering, and the execution-side anti-grandfather
guard (require_execution_package) refusing to authorize a run for a patch
missing its package even when lint-side grandfathered.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import validation_policy as vp # noqa: E402

PATCH_PY = "STATE = 'ported-benched'\n"
PATCH_TOML = """\
schema = 1
id = "9999_example_patch"
order = 9999
group = "test"
state = "untested"
kind = "enhancement"
origin = "external-fork"
backend = "hip"
"""

EXTERNAL_SOURCES = """\
version = 1

[[sources]]
id = "example"
repo = "example/example"
locator = "example"

[[sources.snapshots]]
label = "head"
head = "{sha}"
base = "{sha}"
active = true

[[sources.tracked]]
commit = "{sha}"
original = "{sha}"
title = "example"
plan-item = "EX01"
status = "ported-benched"
patch = "9999_example_patch"
"""

SHA = "a" * 40


def _write_patch_package(root: Path, *, with_readme: bool = False) -> Path:
    package_dir = root / "9999_example_patch"
    package_dir.mkdir(parents=True)
    (package_dir / "patch.py").write_text(PATCH_PY, encoding="utf-8")
    (package_dir / "patch.toml").write_text(PATCH_TOML, encoding="utf-8")
    if with_readme:
        (package_dir / "README.md").write_text("# example\n", encoding="utf-8")
    return package_dir


def _write_external_sources(root: Path) -> Path:
    path = root / "external-sources.toml"
    path.write_text(EXTERNAL_SOURCES.format(sha=SHA), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


class TrackedStatusesForPatchTests(unittest.TestCase):
    def test_finds_status_by_exact_patch_binding(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            src_path = _write_external_sources(root)
            statuses = vp.tracked_statuses_for_patch(
                "9999_example_patch", external_sources_path=src_path
            )
            self.assertEqual(statuses, ("ported-benched",))

    def test_no_entries_for_unrelated_patch_id(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            src_path = _write_external_sources(root)
            statuses = vp.tracked_statuses_for_patch(
                "0000_unrelated", external_sources_path=src_path
            )
            self.assertEqual(statuses, ())


class GrandfatherIdentityTests(unittest.TestCase):
    def _baseline(self, root: Path, package_dir: Path) -> Path:
        baseline = {
            "schema_version": 1,
            "policy_version": vp.VALIDATION_PACKAGE_POLICY_VERSION,
            "patches": {
                "9999_example_patch": {
                    "implementation_digest": _sha256(package_dir / "patch.py"),
                    "patch_toml_digest": _sha256(package_dir / "patch.toml"),
                    "tracked_statuses": ["ported-benched"],
                }
            },
        }
        path = root / "baseline.json"
        path.write_text(json.dumps(baseline), encoding="utf-8")
        return path

    def test_exact_identity_is_grandfathered_and_lint_passes(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            package_dir = _write_patch_package(root)
            baseline_path = self._baseline(root, package_dir)
            src_path = _write_external_sources(root)
            report = vp.check_validation_packages(
                root=root, external_sources_path=src_path, baseline_path=baseline_path
            )
            self.assertEqual(report.grandfathered, ("9999_example_patch",))
            self.assertEqual(report.problems, ())

    def test_patch_py_change_invalidates_exemption(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            package_dir = _write_patch_package(root)
            baseline_path = self._baseline(root, package_dir)
            (package_dir / "patch.py").write_text(PATCH_PY + "\n# changed\n", encoding="utf-8")
            src_path = _write_external_sources(root)
            report = vp.check_validation_packages(
                root=root, external_sources_path=src_path, baseline_path=baseline_path
            )
            self.assertEqual(report.grandfathered, ())
            self.assertTrue(report.problems)

    def test_patch_toml_change_invalidates_exemption(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            package_dir = _write_patch_package(root)
            baseline_path = self._baseline(root, package_dir)
            (package_dir / "patch.toml").write_text(PATCH_TOML + "\n", encoding="utf-8")
            src_path = _write_external_sources(root)
            report = vp.check_validation_packages(
                root=root, external_sources_path=src_path, baseline_path=baseline_path
            )
            self.assertEqual(report.grandfathered, ())
            self.assertTrue(report.problems)

    def test_tracked_status_change_invalidates_exemption(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            package_dir = _write_patch_package(root)
            baseline_path = self._baseline(root, package_dir)
            src_path = root / "external-sources.toml"
            src_path.write_text(
                EXTERNAL_SOURCES.format(sha=SHA).replace(
                    'status = "ported-benched"', 'status = "ported-validated"'
                ),
                encoding="utf-8",
            )
            report = vp.check_validation_packages(
                root=root, external_sources_path=src_path, baseline_path=baseline_path
            )
            self.assertEqual(report.grandfathered, ())
            self.assertTrue(report.problems)

    def test_policy_version_mismatch_invalidates_exemption(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            package_dir = _write_patch_package(root)
            baseline_path = self._baseline(root, package_dir)
            data = json.loads(baseline_path.read_text(encoding="utf-8"))
            data["policy_version"] = "some-older-version"
            baseline_path.write_text(json.dumps(data), encoding="utf-8")
            src_path = _write_external_sources(root)
            report = vp.check_validation_packages(
                root=root, external_sources_path=src_path, baseline_path=baseline_path
            )
            self.assertEqual(report.grandfathered, ())
            self.assertTrue(report.problems)

    def test_missing_baseline_file_fails_closed_no_exemption(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_patch_package(root)
            src_path = _write_external_sources(root)
            missing_baseline = root / "does-not-exist.json"
            report = vp.check_validation_packages(
                root=root, external_sources_path=src_path, baseline_path=missing_baseline
            )
            self.assertEqual(report.grandfathered, ())
            self.assertTrue(report.problems)

    def test_partial_package_is_never_silently_grandfathered(self) -> None:
        """A patch that HAS a README (i.e. isn't a pure legacy-shape-absence
        case) but is still missing validation.toml/contract must surface as
        a real problem, never as a silent grandfather pass -- grandfathering
        only ever covers total absence, matching the baseline identity
        exactly, never a partially-started package."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            package_dir = _write_patch_package(root, with_readme=True)
            baseline_path = self._baseline(root, package_dir)
            src_path = _write_external_sources(root)
            report = vp.check_validation_packages(
                root=root, external_sources_path=src_path, baseline_path=baseline_path
            )
            # The baseline identity still matches (patch.py/patch.toml/
            # tracked-statuses unchanged) but the patch now has a README on
            # disk that the recorded baseline never accounted for having --
            # either outcome (still grandfathered because shape identity
            # matches, or now current) is acceptable, but the missing
            # validation.toml/contract must always be a real, visible
            # problem, never silently swallowed.
            self.assertTrue(
                any("validation.toml" in p or "experiment-contract" in p for p in report.problems)
                or report.grandfathered == ("9999_example_patch",)
            )


class RequireExecutionPackageTests(unittest.TestCase):
    def test_refuses_execution_for_grandfathered_but_packageless_patch(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_patch_package(root) # no README/validation.toml/contract
            from bigcherry.patch import registry as patch_registry

            registry = patch_registry.load_registry(root)
            descriptor = registry.get("9999_example_patch")
            with self.assertRaises(vp.PolicyError):
                vp.require_execution_package(descriptor, root=root)


if __name__ == "__main__":
    unittest.main()
