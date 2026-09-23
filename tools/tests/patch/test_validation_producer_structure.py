"""PA36-F step 6: structural tests that fail if patch/RD-specific execution
machinery creeps back into the generic shared campaign/validation modules,
or if the generic producer runtime/dispatcher regresses its own typed
contracts (GPT design req_8ec9b90c05f84a30, section 6).

Each AST-scan test here is baseline/shrink-only: the baseline set below is
the frozen, already-existing legacy inventory at the time this file was
added (2026-09-16) -- a real migration deletes names FROM these baselines
in the same commit that deletes the function/flag; it must never grow.
"""

from __future__ import annotations

import ast
import dataclasses
import re
import sys
import unittest
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))

from bigcherry.patch import validation as pv  # noqa: E402
from bigcherry.patch import validation_campaign as vc  # noqa: E402
from bigcherry.patch import validation_producer as vp  # noqa: E402

_CAMPAIGN_SRC_PATH = TOOLS_DIR / "bigcherry" / "patch" / "validation_campaign.py"
_PRODUCER_SRC_PATH = TOOLS_DIR / "bigcherry" / "patch" / "validation_producer.py"

# --------------------------------------------------------------- frozen baselines

# Function names already matching run_rd\d+_*/run_patch\d+_*/_load_rd\d+_*
# in validation_campaign.py as of the PA36-F step-6 pass. Migrating a
# producer deletes its entry from BOTH the source file and this baseline in
# the same commit -- the baseline only ever shrinks.
_BASELINE_LEGACY_FUNCTION_NAMES: frozenset[str] = frozenset(
    {
        "run_patch1000_backend_ops_correctness",
        "run_patch1000_backend_ops_perf",
        "run_patch1000_verification",
        "run_rd73_contract_qualification",
        "run_rd73_decode_control_lane",
        "run_rd73_mtp_server_lane",
        "run_rd73_resource_burst_session",
    }
)

# --run-rdNN-*/--run-patchNNNN-* string literals already passed to
# add_argument() in validation_campaign.py as of the PA36-F step-6 pass.
# Same shrink-only rule. The --run-rd73-contract flag remains for the
# legacy RD73 full-qualification path (PA36 close-out: policy-blocked
# sub-slice 3, state="rejected"; flag retained for future re-opening).
_BASELINE_LEGACY_CLI_FLAGS: frozenset[str] = frozenset(
    {"--run-rd73-contract"}
)


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _all_function_names(tree: ast.Module) -> set[str]:
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _all_add_argument_string_literals(tree: ast.Module) -> set[str]:
    literals: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    literals.add(arg.value)
    return literals


class NoNewPatchNamedFunctionsTests(unittest.TestCase):
    """test_shared_campaign_adds_no_new_patch_named_functions"""

    def test_shared_campaign_adds_no_new_patch_named_functions(self) -> None:
        tree = _parse(_CAMPAIGN_SRC_PATH)
        current = {
            name
            for name in _all_function_names(tree)
            if re.fullmatch(r"run_rd\d+.*", name)
            or re.fullmatch(r"run_patch\d+.*", name)
            or re.fullmatch(r"_load_rd\d+.*", name)
        }
        new = current - _BASELINE_LEGACY_FUNCTION_NAMES
        self.assertEqual(
            new,
            set(),
            f"new patch/RD-named function(s) added to shared validation_campaign.py: "
            f"{sorted(new)} -- a normal producer must not add a run_rdXX_*/"
            "run_patchXXXX_*/_load_rdXX_* function to shared execution code",
        )
        shrunk = _BASELINE_LEGACY_FUNCTION_NAMES - current
        if shrunk:
            self.skipTest(
                f"baseline has shrunk (migration landed): {sorted(shrunk)} -- "
                "update _BASELINE_LEGACY_FUNCTION_NAMES to match"
            )


class NoNewPatchNamedCliFlagsTests(unittest.TestCase):
    """test_shared_campaign_adds_no_new_patch_named_cli_flags"""

    def test_shared_campaign_adds_no_new_patch_named_cli_flags(self) -> None:
        tree = _parse(_CAMPAIGN_SRC_PATH)
        current = {
            literal
            for literal in _all_add_argument_string_literals(tree)
            if re.fullmatch(r"--run-rd\d+-.*", literal)
            or re.fullmatch(r"--run-patch\d+-.*", literal)
        }
        new = current - _BASELINE_LEGACY_CLI_FLAGS
        self.assertEqual(
            new,
            set(),
            f"new --run-rdXX-*/--run-patchXXXX-* CLI flag(s) added: {sorted(new)} -- "
            "a normal producer must use --validation-producer/--producer-input, "
            "never a new dedicated flag",
        )


class NoSingularContractAccessTests(unittest.TestCase):
    """test_generic_shared_path_never_reads_singular_validation_plan_contract"""

    def test_generic_shared_path_never_reads_singular_validation_plan_contract(
        self,
    ) -> None:
        # validation.py's own ValidationPlan.contract PROPERTY DEFINITION is
        # explicitly permitted (the 0/1-contract compatibility view GPT's
        # design allows to exist) -- what must never appear is a CONSUMER
        # (an attribute *access*, `x.contract`) anywhere in the shared
        # execution modules.
        for path in (_CAMPAIGN_SRC_PATH, _PRODUCER_SRC_PATH):
            tree = _parse(path)
            offenders = [
                node.lineno
                for node in ast.walk(tree)
                if isinstance(node, ast.Attribute) and node.attr == "contract"
            ]
            self.assertEqual(
                offenders,
                [],
                f"{path.name}: singular '.contract' attribute access at line(s) "
                f"{offenders} -- shared execution code must use plural "
                "'.contracts'/'contract_ids_for_check()', never the singular "
                "compatibility accessor",
            )


class ValidationContextPluralFieldsTests(unittest.TestCase):
    """test_validation_context_has_no_singular_contract_fields"""

    def test_validation_context_has_no_singular_contract_fields(self) -> None:
        field_names = {f.name for f in dataclasses.fields(pv.ValidationContext)}
        self.assertIn("contracts", field_names)
        self.assertIn("contract_hashes", field_names)
        self.assertNotIn("contract", field_names)
        self.assertNotIn("contract_hash", field_names)


class ProducerResultTypedCheckResultsTests(unittest.TestCase):
    """test_producer_result_has_only_typed_check_results"""

    def test_producer_result_has_only_typed_check_results(self) -> None:
        field_names = {f.name for f in dataclasses.fields(vp.ProducerResult)}
        self.assertNotIn("named_correctness_results", field_names)
        self.assertIn("check_results", field_names)
        check_results_field = next(
            f
            for f in dataclasses.fields(vp.ProducerResult)
            if f.name == "check_results"
        )
        self.assertEqual(check_results_field.type, "tuple[ProducerCheckResult, ...]")

    def _plan_and_context(self) -> tuple[pv.ValidationPlan, pv.ValidationContext]:
        check = pv.CheckSpec(
            check_id="c1",
            capability="correctness",
            validator="custom",
            required=True,
            config={"callable": "checks.py:check"},
        )
        plan = pv.ValidationPlan(
            patch_id="p", checks=(check,), universal_capabilities=()
        )
        context = pv.ValidationContext(
            descriptor=None,
            base_revision="a" * 40,
            control_source=None,
            subject_source=None,
        )
        return plan, context

    def _spec(self) -> vp.ProducerSpec:
        return vp.ProducerSpec(
            patch_id="p",
            producer_id="prod",
            entrypoint=Path("producer.py"),
            callable_name="run",
            trace_probe="skip",
            standard_campaign="skip",
            correctness_evidence_cli="forbid",
            performance_benchmark_cli="forbid",
            artifact_names=frozenset(),
            inputs={},
        )

    def _result(self, check_results: tuple) -> vp.ProducerResult:
        return vp.ProducerResult(
            correctness=None,
            validation_build_identities={"control": {}, "subject": {}},
            activation_evidence=None,
            performance_evidence=None,
            trace_evidence=None,
            check_results=check_results,
            lane_effects=(),
            emitted_artifacts=frozenset(),
        )

    def test_invalid_scope_fails_closed(self) -> None:
        plan, context = self._plan_and_context()
        bad = vp.ProducerCheckResult(
            check_id="c1",
            contract_ids=("wrong",),
            validation_result=pv.ValidationResult(
                check_id="c1",
                capability="correctness",
                status=pv.PASS,
                summary="ok",
            ),
        )
        with self.assertRaises(vp.ValidationProducerError):
            vp.validate_producer_result(
                self._spec(), self._result((bad,)), plan=plan, context=context
            )

    def test_invalid_check_id_fails_closed(self) -> None:
        plan, context = self._plan_and_context()
        bad = vp.ProducerCheckResult(
            check_id="unknown-check",
            contract_ids=(),
            validation_result=pv.ValidationResult(
                check_id="unknown-check",
                capability="correctness",
                status=pv.PASS,
                summary="ok",
            ),
        )
        with self.assertRaises(vp.ValidationProducerError):
            vp.validate_producer_result(
                self._spec(), self._result((bad,)), plan=plan, context=context
            )

    def test_invalid_capability_fails_closed(self) -> None:
        plan, context = self._plan_and_context()
        bad = vp.ProducerCheckResult(
            check_id="c1",
            contract_ids=(),
            validation_result=pv.ValidationResult(
                check_id="c1",
                capability="performance",
                status=pv.PASS,
                summary="ok",
            ),
        )
        with self.assertRaises(vp.ValidationProducerError):
            vp.validate_producer_result(
                self._spec(), self._result((bad,)), plan=plan, context=context
            )


class NoPatchIdentityBranchesTests(unittest.TestCase):
    """test_generic_dispatch_has_no_patch_identity_branches"""

    def test_generic_dispatch_has_no_patch_identity_branches(self) -> None:
        tree = _parse(_CAMPAIGN_SRC_PATH)
        target = None
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "execute_validation_producer"
            ):
                target = node
                break
        self.assertIsNotNone(target, "execute_validation_producer() not found")
        offenders = []
        for node in ast.walk(target):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if re.fullmatch(r"RD\d+", node.value) or re.fullmatch(
                    r"\d{4}_.*", node.value
                ):
                    offenders.append((node.lineno, node.value))
        self.assertEqual(
            offenders,
            [],
            f"execute_validation_producer() contains patch-identity string constant(s): "
            f"{offenders}",
        )


class NoProducerSpecificPayloadDecodingTests(unittest.TestCase):
    """test_generic_dispatch_does_not_decode_producer_specific_payload_keys"""

    _FORBIDDEN_ATTRS = {
        "correctness",
        "performance_evidence",
        "trace_evidence",
        "disposition",
    }

    def test_generic_dispatch_does_not_decode_producer_specific_payload_keys(
        self,
    ) -> None:
        tree = _parse(_CAMPAIGN_SRC_PATH)
        target = None
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "execute_validation_producer"
            ):
                target = node
                break
        self.assertIsNotNone(target)
        offenders = []
        for node in ast.walk(target):
            # Merely reading the DECLARED field itself (`record.disposition`,
            # `result.correctness`) is legitimate typed access -- what is
            # forbidden is reaching INTO one of these opaque per-producer
            # payloads for a patch-specific key, e.g. `result.correctness
            # ["some_rd_key"]` or `record.disposition.get("some_rd_key")`.
            if isinstance(node, ast.Subscript) and isinstance(
                node.value, ast.Attribute
            ):
                if node.value.attr in self._FORBIDDEN_ATTRS:
                    offenders.append((node.lineno, f"{node.value.attr}[...]"))
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("get", "__getitem__")
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr in self._FORBIDDEN_ATTRS
            ):
                offenders.append(
                    (node.lineno, f"{node.func.value.attr}.{node.func.attr}(...)")
                )
        self.assertEqual(
            offenders,
            [],
            f"execute_validation_producer() decodes forbidden producer-specific "
            f"payload key(s): {offenders} -- it may read the declared "
            "ProducerResult/ProducerCheckResult fields themselves "
            "(correctness/performance_evidence/trace_evidence/disposition), "
            "but must never subscript/`.get()` a patch-specific key out of "
            "one of those opaque payloads",
        )


class ProducerModulesCannotImportCampaignTests(unittest.TestCase):
    """test_producer_modules_cannot_import_validation_campaign"""

    def test_producer_modules_cannot_import_validation_campaign(self) -> None:
        patches_root = REPO_ROOT / "patches"
        offenders = []
        if patches_root.is_dir():
            for path in patches_root.glob("*/validation/*.py"):
                tree = _parse(path)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name == "bigcherry.patch.validation_campaign":
                                offenders.append(str(path))
                    if isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        if (
                            module == "bigcherry.patch.validation_campaign"
                            or module.endswith(".validation_campaign")
                        ):
                            offenders.append(str(path))
                        # `from bigcherry.patch import validation_campaign`:
                        # the forbidden module is one of the imported NAMES,
                        # not the dotted `module` prefix itself.
                        if module in ("bigcherry.patch", "patch") and any(
                            alias.name == "validation_campaign" for alias in node.names
                        ):
                            offenders.append(str(path))
        self.assertEqual(
            offenders,
            [],
            f"patch-local producer module(s) import validation_campaign.py: {offenders} "
            "-- this is the exact coupling PA36 exists to remove",
        )


class BuildPairBuildsExactlyTwoFatArmsTests(unittest.TestCase):
    """test_build_pair_builds_exactly_two_fat_arms"""

    def test_build_pair_builds_exactly_two_fat_arms(self) -> None:
        from unittest import mock

        from bigcherry.patch import source as real_psi

        fat_targets = vp.FatTargetPlan(targets=("gfx1100", "gfx1201"))
        runtime = vc.CampaignProducerRuntime(
            repo_root=Path("/repo"),
            patch_id="0000_fake",
            base_revision="a" * 40,
            workdir=Path("/work"),
            hip_path=Path("/hip"),
            fat_targets=fat_targets,
            run_dir=Path("/work/run"),
        )

        build_calls: list[dict[str, object]] = []

        def fake_build_tree(
            *,
            name,
            hip_path,
            amdgpu_targets,
            workdir,
            targets,
            source,
            extra_cmake_args,
        ):
            build_calls.append(
                {"name": name, "amdgpu_targets": amdgpu_targets, "source": source}
            )
            return Path(f"/builds/{name}")

        evidence_calls: list[dict[str, object]] = []

        class _FakeEvidence:
            def __init__(self, role: str) -> None:
                self._role = role

            def campaign_identity(self) -> dict[str, str]:
                return {"role": self._role}

        def fake_capture(
            build_dir,
            *,
            source_root,
            architecture,
            binary,
            requested_cmake_args,
            build_env,
        ):
            evidence_calls.append({"architecture": architecture, "binary": binary})
            return _FakeEvidence("control" if len(evidence_calls) == 1 else "subject")

        def run_once() -> object:
            # Patching the REAL bigcherry.patch.source module's own
            # attributes (not swapping sys.modules["bigcherry.patch.source"]
            # wholesale) -- build_pair()'s `from bigcherry.patch import
            # source as psi` resolves via the already-imported `bigcherry.
            # patch` package's cached `source` attribute whenever another
            # test in the same process already imported it first, which a
            # module-swap via mock.patch.dict(sys.modules, ...) does not
            # intercept (a real full-suite-run regression this fixed).
            with (
                mock.patch.object(
                    real_psi,
                    "resolve_source_composition",
                    side_effect=[
                        ("rev123", [("bigcherry", "rev123")]),
                        ("rev123", [("bigcherry", "rev123"), ("0000_fake", "1")]),
                    ],
                ),
                mock.patch.object(
                    real_psi,
                    "materialize_composition",
                    side_effect=[Path("/src/control"), Path("/src/subject")],
                ),
                mock.patch.object(
                    real_psi,
                    "REPO_ROOT",
                    Path("/repo"),
                ),
                mock.patch.object(
                    vc,
                    "build_tree",
                    side_effect=fake_build_tree,
                ),
                mock.patch.object(
                    vc,
                    "capture_completed_build_evidence",
                    side_effect=fake_capture,
                ),
            ):
                return runtime.build_pair(
                    targets=("gfx1100", "gfx1201"), primary_target="llama-bench"
                )

        pair = run_once()

        self.assertEqual(
            len(build_calls), 2, "build_pair() must build exactly two arms"
        )
        self.assertEqual(build_calls[0]["amdgpu_targets"], "gfx1100;gfx1201")
        self.assertEqual(build_calls[1]["amdgpu_targets"], "gfx1100;gfx1201")
        self.assertIn("control", build_calls[0]["name"])
        self.assertIn("subject", build_calls[1]["name"])
        self.assertEqual(len(evidence_calls), 2)
        self.assertEqual(set(pair.validation_build_identities), {"control", "subject"})

        # Device count must never affect the build call count: build_pair()'s
        # own signature has no device-count parameter at all, so calling it
        # again (as a producer with many devices still would, exactly once)
        # produces the same two builds, not more.
        build_calls.clear()
        evidence_calls.clear()
        run_once()
        self.assertEqual(len(build_calls), 2)


class DeviceContextsAreHipOnlyTests(unittest.TestCase):
    """test_device_contexts_are_hip_only"""

    def test_device_contexts_are_hip_only(self) -> None:
        from unittest import mock
        from bigcherry.core import environment as bc_environment

        fat_targets = vp.FatTargetPlan(targets=("gfx1100",))
        runtime = vc.CampaignProducerRuntime(
            repo_root=Path("/repo"),
            patch_id="0000_fake",
            base_revision="a" * 40,
            workdir=Path("/work"),
            hip_path=Path("/hip"),
            fat_targets=fat_targets,
            run_dir=Path("/work/run"),
        )
        devices = (
            bc_environment.Device(
                index=0, arch="gfx1100", model="m0", vram_mib=1, locator="0000:01:00.0"
            ),
            bc_environment.Device(
                index=1, arch="gfx1100", model="m1", vram_mib=1, locator="0000:02:00.0"
            ),
        )
        fake_host = mock.Mock(devices=devices)
        fake_env = mock.Mock()
        fake_env.host.return_value = fake_host

        with mock.patch.object(bc_environment, "load_default", return_value=fake_env):
            contexts = runtime.device_contexts(device_map={"gfx1100": (0, 1)})

        self.assertEqual(len(contexts), 2)
        for ctx in contexts:
            self.assertEqual(
                ctx.env_overrides, {"HIP_VISIBLE_DEVICES": str(ctx.device_index)}
            )
            self.assertNotIn("ROCR_VISIBLE_DEVICES", ctx.env_overrides)
            self.assertEqual(ctx.env_unset, ("ROCR_VISIBLE_DEVICES",))


class BuildPairOverrideParamsTests(unittest.TestCase):
    """test_build_pair_override_params

    PA36 RD12 pilot (dev-gpt-agent req_98777b7a51f84820 + ruling
    req_7efbe5cb5e434582): ``targets`` is authoritative when non-empty, and
    ``common_extra_patches`` resolves into BOTH arms (control + subject) so a
    producer whose correctness pair needs supplementary evidence patches in
    both arms passes them here instead of materializing its own pair below
    the build authority."""

    @staticmethod
    def _runtime(fat_targets):
        return vc.CampaignProducerRuntime(
            repo_root=Path("/repo"),
            patch_id="0000_fake",
            base_revision="a" * 40,
            workdir=Path("/work"),
            hip_path=Path("/hip"),
            fat_targets=vp.FatTargetPlan(targets=tuple(fat_targets)),
            run_dir=Path("/work/run"),
        )

    def _run(self, runtime, **kwargs):
        from unittest import mock

        from bigcherry.patch import source as real_psi

        resolve_calls = []
        build_calls = []
        evidence_calls = []

        def fake_resolve(
            source_name,
            *,
            focal=None,
            extra_patches=(),
            base_ref="HEAD",
            base_repo=None,
            recipes=None,
            patches_root=None,
        ):
            resolve_calls.append(
                {"focal": focal, "extra_patches": tuple(extra_patches)}
            )
            composition = [("bigcherry", "rev123")]
            for extra in extra_patches:
                composition.append((extra, "1"))
            if focal is not None:
                composition.append((focal, "1"))
            return "rev123", tuple(composition)

        def fake_build_tree(
            *,
            name,
            hip_path,
            amdgpu_targets,
            workdir,
            targets,
            source,
            extra_cmake_args,
        ):
            build_calls.append(
                {
                    "name": name,
                    "amdgpu_targets": amdgpu_targets,
                    "targets": list(targets),
                }
            )
            return Path("/builds/" + str(name))

        class _FakeEvidence:
            def campaign_identity(self):
                return {"role": "x"}

        def fake_capture(
            build_dir,
            *,
            source_root,
            architecture,
            binary,
            requested_cmake_args,
            build_env,
        ):
            evidence_calls.append(
                {"architecture": tuple(architecture), "binary": binary}
            )
            return _FakeEvidence()

        with (
            mock.patch.object(
                real_psi,
                "resolve_source_composition",
                side_effect=fake_resolve,
            ),
            mock.patch.object(
                real_psi,
                "materialize_composition",
                side_effect=[Path("/src/control"), Path("/src/subject")],
            ),
            mock.patch.object(real_psi, "REPO_ROOT", Path("/repo")),
            mock.patch.object(
                vc,
                "build_tree",
                side_effect=fake_build_tree,
            ),
            mock.patch.object(
                vc,
                "capture_completed_build_evidence",
                side_effect=fake_capture,
            ),
        ):
            runtime.build_pair(**kwargs)

        return {
            "resolve": resolve_calls,
            "build": build_calls,
            "evidence": evidence_calls,
        }

    def test_build_pair_common_extra_patches_apply_to_both_arms(self):
        runtime = self._runtime(("gfx1100",))
        got = self._run(
            runtime,
            targets=("gfx1100",),
            primary_target="test-backend-ops",
            common_extra_patches=("1222_e", "1223_e"),
        )
        resolve = got["resolve"]
        self.assertEqual(
            len(resolve), 2, "build_pair() must resolve exactly control + subject"
        )
        self.assertIsNone(resolve[0]["focal"], "control arm must stay focal-free")
        self.assertEqual(resolve[0]["extra_patches"], ("1222_e", "1223_e"))
        self.assertEqual(
            resolve[1]["focal"], "0000_fake", "subject arm must keep its focal"
        )
        self.assertEqual(
            resolve[1]["extra_patches"],
            ("1222_e", "1223_e"),
            "common_extra_patches must reach BOTH arms",
        )
        self.assertEqual(len(got["build"]), 2)

    def test_build_pair_targets_override_is_authoritative(self):
        # fat_targets says gfx1100, but an explicit targets=("gfx1030",) override
        # must drive the actual CMake AMDGPU_TARGETS value and the directory slug.
        runtime = self._runtime(("gfx1100",))
        got = self._run(
            runtime, targets=("gfx1030",), primary_target="test-backend-ops"
        )
        self.assertEqual(len(got["build"]), 2)
        for call in got["build"]:
            self.assertEqual(
                call["amdgpu_targets"],
                "gfx1030",
                "non-empty targets must override fat_targets",
            )
            self.assertIn("gfx1030", str(call["name"]))
            self.assertNotIn("gfx1100", str(call["name"]))

    def test_build_pair_targets_required_and_validated(self):
        # GPT review req_052817cb66d14bc1: no silent defaults -- targets and
        # primary_target are required, and targets must satisfy the
        # FatTargetPlan invariants (non-empty, no duplicates); there is no
        # accidental fallback to the runtime fat plan.
        runtime = self._runtime(("gfx1100", "gfx1201"))
        with self.assertRaises(TypeError):
            runtime.build_pair(primary_target="llama-bench")
        with self.assertRaises(vp.ValidationProducerError):
            self._run(runtime, targets=(), primary_target="llama-bench")
        with self.assertRaises(vp.ValidationProducerError):
            self._run(
                runtime, targets=("gfx1100", "gfx1100"), primary_target="llama-bench"
            )

    def test_build_pair_require_parity_fails_closed_on_mismatch(self) -> None:
        # GPT review req_7a72896b609a48b5 BLOCKER #3: with require_parity set,
        # build_pair() must run the REAL assert_validation_subject_parity()
        # and fail closed on a configure/build-id mismatch (and return the
        # pair when parity holds).
        from unittest import mock

        from bigcherry.patch import source as real_psi

        runtime = self._runtime(("gfx1100",))

        class _Ev:
            def __init__(self, configure: str, build_id: str) -> None:
                self.effective_configure = configure
                self.effective_build_id = build_id

            def campaign_identity(self) -> dict[str, object]:
                return {"role": "x"}

        def _run_pair(parity_mismatch: bool):
            evidences = [
                _Ev("cfg-control", "bid-1"),
                _Ev(
                    "cfg-subject-different" if parity_mismatch else "cfg-control",
                    "bid-1",
                ),
            ]
            with (
                mock.patch.object(
                    real_psi,
                    "resolve_source_composition",
                    side_effect=[
                        ("rev123", (("bigcherry", "rev123"),)),
                        (
                            "rev123",
                            (("bigcherry", "rev123"), (runtime.patch_id, "1")),
                        ),
                    ],
                ),
                mock.patch.object(
                    real_psi,
                    "materialize_composition",
                    side_effect=[Path("/src/control"), Path("/src/subject")],
                ),
                mock.patch.object(real_psi, "REPO_ROOT", Path("/repo")),
                mock.patch.object(
                    vc,
                    "build_tree",
                    side_effect=[
                        Path("/builds/control"),
                        Path("/builds/subject"),
                    ],
                ),
                mock.patch.object(
                    vc,
                    "capture_completed_build_evidence",
                    side_effect=evidences,
                ),
            ):
                if parity_mismatch:
                    with self.assertRaisesRegex(
                        vc.PatchCampaignError,
                        "parity",
                    ):
                        runtime.build_pair(
                            targets=("gfx1100",),
                            primary_target="test-backend-ops",
                            require_parity=True,
                        )
                else:
                    result = runtime.build_pair(
                        targets=("gfx1100",),
                        primary_target="test-backend-ops",
                        require_parity=True,
                    )
                    self.assertEqual(
                        result.validation_build_identities,
                        {"control": {"role": "x"}, "subject": {"role": "x"}},
                    )

        # Mismatched effective_configure -> fail closed before returning.
        _run_pair(True)
        # Matching configure + build id -> parity holds, pair returned.
        _run_pair(False)


if __name__ == "__main__":
    unittest.main()
