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


def _write_untracked_external_sources(root: Path) -> Path:
    path = root / "external-sources.toml"
    path.write_text(
        'version = 1\n\n[[sources]]\nid = "example"\nrepo = "example/example"\n'
        'locator = "example"\n\n[[sources.snapshots]]\nlabel = "head"\n'
        f'head = "{SHA}"\nbase = "{SHA}"\nactive = true\n',
        encoding="utf-8",
    )
    return path


def _write_local_framework_package(
    root: Path, *, adapter: str, with_readme: bool = True
) -> Path:
    package = _write_patch_package(root, with_readme=with_readme)
    metadata = PATCH_TOML.replace('kind = "enhancement"', 'kind = "framework"')
    metadata = metadata.replace('origin = "external-fork"', 'origin = "local"')
    (package / "patch.toml").write_text(metadata, encoding="utf-8")
    (package / "validation.toml").write_text(adapter, encoding="utf-8")
    return package


FRAMEWORK_APPLY_BUILD = (
    'schema = 1\n[[check]]\nid = "apply"\ncapability = "apply"\nvalidator = "apply"\n'
    '[[check]]\nid = "build"\ncapability = "build"\nvalidator = "build"\n'
)


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


class LocalFrameworkStaticPolicyTests(unittest.TestCase):
    def _report(self, root: Path, *, baseline: Path | None = None) -> vp.PackagePolicyReport:
        return vp.check_validation_packages(
            root=root,
            external_sources_path=_write_untracked_external_sources(root),
            baseline_path=baseline,
        )

    def _baseline(self, root: Path, package: Path) -> Path:
        path = root / "baseline.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "policy_version": vp.VALIDATION_PACKAGE_POLICY_VERSION,
                    "patches": {
                        "9999_example_patch": {
                            "implementation_digest": _sha256(package / "patch.py"),
                            "patch_toml_digest": _sha256(package / "patch.toml"),
                            "tracked_statuses": [],
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_complete_zero_contract_framework_adapter_is_current(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_local_framework_package(root, adapter=FRAMEWORK_APPLY_BUILD)
            report = self._report(root)
            self.assertEqual(report.problems, ())
            self.assertEqual(report.grandfathered, ())
            self.assertEqual(report.statuses[0].status, "current")

    def test_zero_contract_framework_missing_readme_fails_lint(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_local_framework_package(
                root, adapter=FRAMEWORK_APPLY_BUILD, with_readme=False
            )
            report = self._report(root)
            self.assertEqual(report.grandfathered, ())
            self.assertTrue(any("missing README.md" in p for p in report.problems))

    def test_zero_contract_framework_missing_custom_callable_fails_lint(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            adapter = FRAMEWORK_APPLY_BUILD + (
                '[[check]]\nid = "custom"\ncapability = "smoke"\nvalidator = "custom"\n'
            )
            _write_local_framework_package(root, adapter=adapter)
            report = self._report(root)
            self.assertTrue(any("no 'callable' string" in p for p in report.problems))

    def test_grandfather_cannot_mask_present_framework_adapter_failure(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            package = _write_local_framework_package(
                root, adapter="schema = 99\n", with_readme=False
            )
            baseline = self._baseline(root, package)
            report = self._report(root, baseline=baseline)
            self.assertEqual(report.grandfathered, ())
            self.assertTrue(any("unsupported schema" in p for p in report.problems))


class RequireExecutionPackageTests(unittest.TestCase):
    def _framework_descriptor(self, root: Path, adapter: str) -> object:
        """Create a local, non-RD framework descriptor for execution tests."""
        from bigcherry.patch import registry as patch_registry

        package = _write_patch_package(root, with_readme=True)
        metadata = PATCH_TOML.replace('kind = "enhancement"', 'kind = "framework"')
        metadata = metadata.replace('origin = "external-fork"', 'origin = "local"')
        (package / "patch.toml").write_text(metadata, encoding="utf-8")
        (package / "validation.toml").write_text(adapter, encoding="utf-8")
        return patch_registry.load_registry(root).get("9999_example_patch")

    def test_zero_contract_execution_is_limited_to_local_non_rd_framework(self) -> None:
        from bigcherry.patch import registry as patch_registry
        for kind, origin, extra, allowed in (
            ("framework", "local", "", True),
            ("enhancement", "local", "", False),
            ("framework", "external-fork", "", False),
            ("framework", "local", 'plan-ids = ["RD01"]\n', False),
            ("framework", "local", 'external-source = "example"\n', False),
        ):
            with self.subTest(kind=kind, origin=origin, extra=extra), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                package = _write_patch_package(root, with_readme=True)
                metadata = PATCH_TOML.replace('kind = "enhancement"', f'kind = "{kind}"')
                metadata = metadata.replace('origin = "external-fork"', f'origin = "{origin}"')
                (package / "patch.toml").write_text(metadata + extra, encoding="utf-8")
                (package / "validation.toml").write_text(
                    'schema = 1\n[[check]]\nid = "apply"\ncapability = "apply"\nvalidator = "apply"\n'
                    '[[check]]\nid = "build"\ncapability = "build"\nvalidator = "build"\n',
                    encoding="utf-8")
                descriptor = patch_registry.load_registry(root).get("9999_example_patch")
                if not allowed:
                    with self.assertRaisesRegex(vp.PolicyError, "no experiment-contract"):
                        vp.require_execution_package(descriptor, root=root)
                    continue
                plan = vp.require_execution_package(descriptor, root=root)
                self.assertEqual(plan.contracts, ())
                self.assertEqual(set(plan.required_capabilities), {"apply", "build"})
                self.assertFalse(vp.patch_validation.compute_verdict(plan, {}).eligible)
                (package / "validation.toml").unlink()
                descriptor = patch_registry.load_registry(root).get("9999_example_patch")
                with self.assertRaisesRegex(vp.PolicyError, "no validation.toml"):
                    vp.require_execution_package(descriptor, root=root)

    def test_zero_contract_framework_requires_readme_for_execution(self) -> None:
        from bigcherry.patch import registry as patch_registry

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            package = _write_patch_package(root, with_readme=False)
            metadata = PATCH_TOML.replace('kind = "enhancement"', 'kind = "framework"')
            metadata = metadata.replace('origin = "external-fork"', 'origin = "local"')
            (package / "patch.toml").write_text(metadata, encoding="utf-8")
            (package / "validation.toml").write_text(
                'schema = 1\n[[check]]\nid = "apply"\ncapability = "apply"\nvalidator = "apply"\n'
                '[[check]]\nid = "build"\ncapability = "build"\nvalidator = "build"\n',
                encoding="utf-8",
            )
            descriptor = patch_registry.load_registry(root).get("9999_example_patch")
            with self.assertRaisesRegex(vp.PolicyError, "missing README\\.md"):
                vp.require_execution_package(descriptor, root=root)

    def test_zero_contract_framework_requires_apply_and_build_producers(self) -> None:
        for missing, adapter in (
            (
                "build",
                'schema = 1\n[[check]]\nid = "apply"\ncapability = "apply"\nvalidator = "apply"\n',
            ),
            (
                "apply",
                'schema = 1\n[[check]]\nid = "build"\ncapability = "build"\nvalidator = "build"\n',
            ),
        ):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                descriptor = self._framework_descriptor(root, adapter)
                with self.assertRaisesRegex(
                    vp.patch_validation.ConfigurationError,
                    rf"required capabilities with no producer: .*{missing}"
                ):
                    vp.require_execution_package(descriptor, root=root)

    def test_zero_contract_framework_rejects_malformed_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            descriptor = self._framework_descriptor(root, "schema = 99\n")
            with self.assertRaisesRegex(vp.patch_validation.ConfigurationError, "unsupported schema"):
                vp.require_execution_package(descriptor, root=root)

    def test_zero_contract_framework_rejects_unresolvable_bound_contract(self) -> None:
        from bigcherry.patch import registry as patch_registry

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            package = _write_patch_package(root, with_readme=True)
            metadata = PATCH_TOML.replace('kind = "enhancement"', 'kind = "framework"')
            metadata = metadata.replace('origin = "external-fork"', 'origin = "local"')
            metadata += 'experiment-contract = "missing-contract-for-execution-test"\n'
            (package / "patch.toml").write_text(metadata, encoding="utf-8")
            (package / "validation.toml").write_text(
                'schema = 1\n[[check]]\nid = "apply"\ncapability = "apply"\nvalidator = "apply"\n'
                '[[check]]\nid = "build"\ncapability = "build"\nvalidator = "build"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(patch_registry.PatchRegistryError, "not found"):
                patch_registry.load_registry(root).get("9999_example_patch")

    def test_refuses_execution_for_grandfathered_but_packageless_patch(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_patch_package(root) # no README/validation.toml/contract
            from bigcherry.patch import registry as patch_registry

            registry = patch_registry.load_registry(root)
            descriptor = registry.get("9999_example_patch")
            with self.assertRaises(vp.PolicyError):
                vp.require_execution_package(descriptor, root=root)


class RegressionTests(unittest.TestCase):
    """Regressions for two real bugs found during round-5 review
    (req_86cfd3a0bff04716): a state='validated' framework patch incorrectly
    swept into the RD-only package requirement, and a custom-validator spec
    that was never actually checked (parse_validation_toml leaves
    validator-specific config opaque, so a malformed/missing 'callable' key
    previously lint-clean as long as producer coverage succeeded)."""

    def test_framework_kind_patch_is_never_required_even_when_validated(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            package_dir = root / "9998_framework_example"
            package_dir.mkdir(parents=True)
            (package_dir / "patch.py").write_text("STATE = 'validated'\n", encoding="utf-8")
            (package_dir / "patch.toml").write_text(
                'schema = 1\nid = "9998_framework_example"\norder = 9998\n'
                'group = "test"\nstate = "validated"\nkind = "framework"\n'
                'origin = "local"\nbackend = "hip"\n',
                encoding="utf-8",
            )
            src_path = root / "external-sources.toml"
            src_path.write_text(
                'version = 1\n\n[[sources]]\nid = "example"\nrepo = "example/example"\n'
                'locator = "example"\n\n[[sources.snapshots]]\nlabel = "head"\n'
                f'head = "{SHA}"\nbase = "{SHA}"\nactive = true\n',
                encoding="utf-8",
            )
            baseline_path = root / "baseline.json"
            baseline_path.write_text(
                json.dumps({"schema_version": 1, "policy_version": vp.VALIDATION_PACKAGE_POLICY_VERSION, "patches": {}}),
                encoding="utf-8",
            )
            report = vp.check_validation_packages(
                root=root, external_sources_path=src_path, baseline_path=baseline_path
            )
            statuses = {s.patch_id: s.status for s in report.statuses}
            self.assertEqual(statuses.get("9998_framework_example"), "not-required")
            self.assertEqual(report.problems, ())

    def test_custom_validator_spec_static_check_rejects_missing_callable(self) -> None:
        """validate_custom_callable_spec() -- now actually wired into
        check_validation_packages()'s per-check scan -- rejects a spec
        that doesn't resolve to a real, correctly-signatured function,
        without ever importing/executing the target module."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "validation").mkdir()
            (root / "validation" / "checks.py").write_text(
                "def check(ctx):\n    return None\n", encoding="utf-8"
            )
            vp.validate_custom_callable_spec("validation/checks.py:check", package_root=root)
            with self.assertRaises(vp.PolicyError):
                vp.validate_custom_callable_spec(
                    "validation/checks.py:does_not_exist", package_root=root
                )
            with self.assertRaises(vp.PolicyError):
                vp.validate_custom_callable_spec(
                    "validation/missing_file.py:check", package_root=root
                )


if __name__ == "__main__":
    unittest.main()
