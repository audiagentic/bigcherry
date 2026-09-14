"""PA39 / GPT review (req_36cddfd518444cda, MUST 4): CLI wiring coverage for
--run-rd12-contract.

RD12 is a genuinely model/manifest-free contract -- the documented
invocation runs with neither --model nor --manifest, and the generic
e2e_smoke_campaign.Campaign Path()-converts both in __post_init__, so it
cannot be constructed on this path. The settled shape:

  * run() performs the five shared builds (tune/replay/stock campaign
    builds + control/validation-subject validation builds -- the record's
    build_identities MUST be exactly the {tune,replay,stock} campaign
    domain), then RETURNS EARLY to _run_rd12_contract() BEFORE the
    ``from bigcherry.e2e_smoke_campaign import Campaign`` line;
  * _run_rd12_contract() owns the full RD12 evidence pipeline itself and
    never touches args.model / args.manifest at all.

The source-inspection tests below pin that wiring (matching the
established pattern for the other specialized modes); the behavioral test
at the bottom drives run() end to end with the documented model/manifest-
free invocation, a Campaign-instantiation tripwire, and real make_record()
-- proving the documented command actually reaches a written record
without constructing the Campaign.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import validation as patch_validation  # noqa: E402
from bigcherry.patch import validation_campaign as vc  # noqa: E402
from bigcherry.experiment.contract import CorrectnessResult  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]
PATCH_ID = "1205_rd12_paired_mmvq_dual_output"
ARCH = "gfx1100"
BASE_REVISION = "ab" * 20  # 40-hex
PATCH_DIGEST = "cd" * 32  # 64-hex
CONTROL_TREE = "11" * 20  # 40-hex
SUBJECT_TREE = "22" * 20  # 40-hex
STOCK_TREE = "33" * 20  # 40-hex
COMPOSITION_DIGEST = "44" * 32  # 64-hex

_HEX_DIGITS = "0123456789abcdef"


def _build_identity(tag: int) -> dict:
    """One full-shape build identity (evidence.BUILD_IDENTITY_KEYS)."""
    def h(offset: int) -> str:
        return _HEX_DIGITS[(tag + offset) % 16] * 64

    return {
        "effective_build_id": h(0),
        "compile_verification_id": h(1),
        "compile_commands_digest": h(2),
        "hip_compile_commands_digest": h(3),
        "runtime_bundle_hash": h(4),
        "runtime_artifacts": {"llama-bench.exe": h(5)},
    }


def _build_evidence(tag: int) -> SimpleNamespace:
    ident = _build_identity(tag)
    return SimpleNamespace(
        effective_build_id=ident["effective_build_id"],
        runtime_bundle_hash=ident["runtime_bundle_hash"],
        compile_verification_id=ident["compile_verification_id"],
        effective_configure=["-DCMAKE_BUILD_TYPE=Release"],
        verification=SimpleNamespace(to_dict=lambda: {"status": "pass"}),
        runtime_artifacts=dict(ident["runtime_artifacts"]),
        campaign_identity=lambda: _build_identity(tag),
    )


def _rd12_qualification() -> dict:
    """Canned run_rd12_correctness_check() result (real producer shape)."""
    return {
        "results": {
            "bit_identical": CorrectnessResult(
                check="bit_identical", passed=True,
                detail="6/6 rows bit-identical (subject vs control)",
            ),
            "backend_reference": CorrectnessResult(
                check="backend_reference", passed=True,
            ),
            "activation": CorrectnessResult(
                check="activation", passed=True,
                detail="marker emitted on subject, absent on control",
            ),
        },
        "artifact": {"path": "rd12/artifact.json", "sha256": "55" * 32},
        "rows": 6,
        "subject_log_path": f"logs/activation-rd12-{ARCH}-subject.log",
        "control_log_path": f"logs/activation-rd12-{ARCH}-control.log",
        "subject_log_artifact": {
            "path": f"logs/activation-rd12-{ARCH}-subject.log", "sha256": "66" * 32,
        },
        "control_log_artifact": {
            "path": f"logs/activation-rd12-{ARCH}-control.log", "sha256": "77" * 32,
        },
        "validation_build_identities": {
            "control": _build_identity(3),
            "subject": _build_identity(4),
        },
    }


def _descriptor() -> SimpleNamespace:
    return SimpleNamespace(
        patch_id=PATCH_ID,
        experiment_contract="RD12-PAIRED-MMVQ-DUAL",
        # compute_persisted_validation_eligible() uses the PLURAL -- empty
        # here so the adapter verdict alone is the only claim.
        experiment_contracts=(),
        implementation_path="patch.py",
        package_root=None,
        implementation_digest=PATCH_DIGEST,
        validation_digest="88" * 32,
        representation="simple",
    )


def _args(tmp: Path) -> SimpleNamespace:
    """The documented invocation: --run-rd12-contract with NEITHER
    --model NOR --manifest (MUST 4)."""
    return SimpleNamespace(
        run_rd12_contract=True,
        model=None,
        manifest=None,
        patch=PATCH_ID,
        amdgpu_targets=ARCH,
        hip_path=tmp / "rocm",
        workdir=tmp / "workdir",
        build_root=tmp / "build",
        worktree_root=tmp / "worktree",
        baseline_source="bigcherry",
        trace_marker_regex=None,
        trace_description=None,
        correctness_evidence=None,
        run_rd08_lanes=False,
        run_rd08_contract=False,
        run_rd04_benchmark=False,
        run_rd58_state_restore=False,
        run_rd73_contract=False,
        run_rd04_contract=False,
        run_rd13_contract=False,
        run_rd26_contract=False,
        framework_configuration=False,
        run_performance_benchmark=False,
    )


class Rd12ContractCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fn_source = inspect.getsource(vc._run_rd12_contract)
        self.run_source = inspect.getsource(vc.run)
        self.main_source = inspect.getsource(vc.main)

    # -- early-return placement in run() ---------------------------------

    def test_early_return_precedes_campaign_import(self) -> None:
        # MUST 4: the specialized path must return before the generic
        # Campaign is even imported, so the documented model/manifest-free
        # invocation can never construct it.
        guard = self.run_source.index(
            'if getattr(args, "run_rd12_contract", False):'
        )
        self.assertLess(
            guard, self.run_source.index("from bigcherry.e2e_smoke_campaign import"),
        )

    def test_early_return_sits_after_all_five_shared_builds(self) -> None:
        # The record's build_identities MUST be exactly the {tune,replay,
        # stock} campaign domain and the declared build check needs the
        # real validation-domain control/subject identities -- so the five
        # shared builds run FIRST, and the last of them (the
        # validation-subject parity assert) precedes the early return.
        self.assertLess(
            self.run_source.index("assert_validation_subject_parity("),
            self.run_source.index('if getattr(args, "run_rd12_contract", False):'),
        )

    def test_no_inline_rd12_block_or_dead_flag_remains_in_run(self) -> None:
        # The old inline RD12 block lived between the shared builds and the
        # Campaign import; after the MUST 4 refactor it must be gone, and
        # no bare args.run_rd12_contract reference may linger (the live
        # guard is the getattr form above).
        self.assertNotIn("if args.run_rd12_contract:", self.run_source)
        self.assertNotIn("args.run_rd12_contract", self.run_source)

    def test_skip_lists_do_not_carry_dead_rd12_flag(self) -> None:
        # --run-rd12-contract returns before both standalone gates, so
        # neither skip list may mention it (dead flags would imply a path
        # that never exists). The NB comment records WHY it is absent.
        self.assertIn(
            "trace_result = None if (args.run_rd08_contract or "
            "args.run_rd04_benchmark or args.run_rd58_state_restore or "
            "args.run_rd73_contract or args.run_rd04_contract) else "
            "run_trace_activation_probes(",
            self.run_source,
        )
        self.assertIn(
            "if not (args.run_rd08_contract or args.run_rd04_benchmark or "
            "args.run_rd58_state_restore or args.run_rd73_contract or "
            "args.run_rd04_contract or args.run_rd13_contract or "
            "args.run_rd26_contract):",
            self.run_source,
        )
        self.assertIn(
            "# NB: --run-rd12-contract is not in these skip lists",
            self.run_source,
        )

    # -- _run_rd12_contract() invariants ----------------------------------

    def test_mutually_exclusive_with_other_specialized_modes(self) -> None:
        self.assertIn(
            "run-rd12-contract is mutually exclusive with the", self.fn_source,
        )
        for flag in (
            "args.run_rd08_lanes", "args.run_rd08_contract",
            "args.run_rd04_benchmark", "args.run_rd58_state_restore",
            "args.run_rd73_contract", "args.run_rd04_contract",
            "args.run_rd13_contract", "args.run_rd26_contract",
        ):
            self.assertIn(flag, self.fn_source)
        # The mutual-exclusion condition must never reference itself --
        # a real bug caught during authoring: including run_rd12_contract
        # in its own guard makes the condition always true inside the
        # branch, so the flag could never actually run.
        self.assertNotIn('args.run_rd12_contract\n', self.fn_source)

    def test_rd12_only_gating(self) -> None:
        self.assertIn(
            'descriptor.experiment_contract != "RD12-PAIRED-MMVQ-DUAL"',
            self.fn_source,
        )

    def test_correctness_evidence_ambiguity_guard(self) -> None:
        self.assertIn("args.correctness_evidence is not None", self.fn_source)
        self.assertIn(
            "--correctness-evidence and --run-rd12-contract are ambiguous",
            self.fn_source,
        )

    def test_never_reads_model_or_manifest(self) -> None:
        # MUST 4, the core invariant: the specialized path is
        # model/manifest-free end to end -- it may not read args.model or
        # args.manifest anywhere (the old code Path()-converted both via
        # the Campaign constructor before reaching this point).
        self.assertNotIn("args.model", self.fn_source)
        self.assertNotIn("args.manifest", self.fn_source)

    def test_campaign_identity_digest_is_model_free(self) -> None:
        # The record still needs a campaign_identity_digest: bind one from
        # the shared-build facts only (same canonical-JSON sha256
        # convention as e2e_smoke_campaign._stable_json_sha256), never
        # from model/manifest.
        self.assertIn('"schema": "rd12-contract-campaign-identity-v1"', self.fn_source)
        self.assertIn('"campaign_build_identities": campaign_build_identities', self.fn_source)

    def test_binds_correctness_summary_evidence_never_contract_promotions(self) -> None:
        # GPT review (req_5631b12dc3fb4a23): correctness_evidence must
        # point at the canonical correctness.json (real "disposition"
        # field), never rd12_qualification["artifact"] (no "disposition").
        self.assertNotIn(
            'correctness_evidence = {"artifact": rd12_qualification["artifact"]}',
            self.fn_source,
        )
        self.assertIn(
            '"path": correctness_path.relative_to(campaign_run_dir).as_posix()',
            self.fn_source,
        )
        self.assertIn(
            '"sha256": hashlib.sha256(correctness_path.read_bytes()).hexdigest()',
            self.fn_source,
        )
        self.assertNotIn("contract_promotions[", self.fn_source)

    def test_disposition_derives_from_bit_identical_not_backend_reference(self) -> None:
        self.assertIn(
            '"disposition": "passed" if bit_identical_result.passed else "failed"',
            self.fn_source,
        )

    def test_binds_trace_evidence_from_per_arm_activation_logs(self) -> None:
        # This path skips the generic trace probe, so _run_rd12_contract()
        # is the ONLY source of trace_evidence for the record -- otherwise
        # the declared trace-marker check can never leave BLOCKED no
        # matter how many real runs pass.
        self.assertIn("trace_evidence = {", self.fn_source)
        self.assertIn('"marker_regex": trace_marker_regex', self.fn_source)
        self.assertIn('rd12_qualification["subject_log_artifact"]', self.fn_source)
        self.assertIn('rd12_qualification["control_log_artifact"]', self.fn_source)
        # positive must be the subject (marker-present) arm and negative
        # the control (marker-absent) arm -- swapped arms would silently
        # invert the check's meaning.
        self.assertLess(
            self.fn_source.index('rd12_qualification["subject_log_artifact"]'),
            self.fn_source.index('"negative"'),
        )
        self.assertGreater(
            self.fn_source.index('rd12_qualification["control_log_artifact"]'),
            self.fn_source.index('"negative"'),
        )

    def test_contract_correctness_gate_threads_rd12_results(self) -> None:
        # Real bug caught on first real-hardware run (2026-09-13): RD12's
        # bit_identical PASS never reached compute_contract_correctness_
        # gate(), so the record reported "contract correctness gate:
        # blocked" despite genuinely passing evidence.
        self.assertIn(
            "contract_correctness_gate = compute_contract_correctness_gate(\n"
            "            full_contract, rd12_qualification[\"results\"],\n"
            "        )",
            self.fn_source,
        )

    def test_validation_build_identities_thread_rd12_qualification(self) -> None:
        # RD12's producer materializes and builds its OWN isolated
        # control/subject worktrees -- record ITS identities (the RD58
        # pattern), never the generic campaign's.
        self.assertIn(
            'validation_build_identities=rd12_qualification["validation_build_identities"],',
            self.fn_source,
        )

    def test_validation_context_runs_model_free(self) -> None:
        self.assertIn("model=None,", self.fn_source)

    # -- CLI prerequisite (main) ------------------------------------------

    def test_flag_exists_in_main(self) -> None:
        self.assertIn('"--run-rd12-contract"', self.main_source)

    def test_cli_prerequisite_is_model_free_for_rd12(self) -> None:
        # GPT review (req_243e3fcd3d684077): the documented
        # --run-rd12-contract command cannot execute unless the CLI
        # prerequisite stops demanding --model/--manifest for RD12 --
        # the producer consumes neither (its workload is the registered
        # 1258 test-backend-ops case). RD12 keeps its own precise
        # requirement (one contract architecture).
        main_source = self.main_source
        rd12_gate = main_source.index("if args.run_rd12_contract:")
        elif_pos = main_source.index("elif (", rd12_gate)
        prerequisite_block = main_source[rd12_gate:elif_pos]
        self.assertIn(
            'parser.error("--run-rd12-contract requires --amdgpu-targets")',
            prerequisite_block,
        )
        self.assertNotIn("args.model", prerequisite_block)
        self.assertNotIn("args.manifest", prerequisite_block)
        # The generic model/manifest requirement must remain for the other
        # modes, and must sit in the RD12 branch's else, not before it.
        generic = main_source.index(
            "runtime qualification requires --model, --manifest, and --amdgpu-targets"
        )
        self.assertGreater(generic, elif_pos)

    def test_run_performance_benchmark_exclusion_includes_rd12(self) -> None:
        self.assertIn(
            '"run_rd58_state_restore", "run_rd73_contract", "run_rd12_contract", '
            '"run_rd04_contract",',
            self.main_source,
        )

    # -- producer-level invariants (run_rd12_correctness_check) ------------

    def test_run_rd12_correctness_check_exposes_validation_build_identities(self) -> None:
        source = inspect.getsource(vc.run_rd12_correctness_check)
        self.assertIn('"validation_build_identities": {', source)
        self.assertIn('"control": artifact_doc["control_build_identity"]', source)
        self.assertIn('"subject": artifact_doc["subject_build_identity"]', source)

    def test_run_rd12_correctness_check_writes_per_arm_activation_logs(self) -> None:
        # The declared trace-marker check can only be satisfied from REAL
        # subprocess output: the producer must write one raw per-arm log
        # into the run dir and expose bound {path, sha256} refs for the
        # CLI to consume (RD08/RD73 precedent). Namespaced by architecture
        # under logs/ (GPT review req_243e3fcd3d684077): the standalone
        # lab driver shares one run_dir across all three contract
        # architectures, so an un-namespaced pair would be overwritten by
        # each later architecture.
        source = inspect.getsource(vc.run_rd12_correctness_check)
        self.assertIn('run_dir / "logs"', source)
        self.assertIn(
            'log_dir / f"activation-rd12-{architecture}-subject.log"', source,
        )
        self.assertIn(
            'log_dir / f"activation-rd12-{architecture}-control.log"', source,
        )
        self.assertNotIn('"activation-rd12-subject.log"', source)
        self.assertNotIn('"activation-rd12-control.log"', source)
        self.assertIn('"subject_log_artifact": subject_log_ref', source)
        self.assertIn('"control_log_artifact": control_log_ref', source)
        # Raw streams must be captured per invocation, not summarized away
        # (the old code discarded stdout/stderr entirely).
        self.assertIn('"stdout": stdout, "stderr": stderr', source)
        # The producer's own hit observation must search the same
        # stream space the validator does (both streams).
        self.assertIn(
            '"hit": trace_marker in stdout or trace_marker in stderr', source,
        )


class Rd12ContractBehavioralTests(unittest.TestCase):
    """Behavioral proof for MUST 4: drive run() with the documented
    model/manifest-free invocation and prove the full pipeline reaches a
    written record WITHOUT ever constructing e2e_smoke_campaign.Campaign
    (the old code crashed the documented command with Path(None) in
    Campaign.__post_init__)."""

    def _expected_identity_digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "schema": "rd12-contract-campaign-identity-v1",
                    "patch_identity": {"name": PATCH_ID, "digest": PATCH_DIGEST},
                    "patched_source_tree": SUBJECT_TREE,
                    "gpu_architecture": ARCH,
                    "campaign_build_identities": {
                        "tune": _build_identity(0),
                        "replay": _build_identity(1),
                        "stock": _build_identity(2),
                    },
                    "base_revision": BASE_REVISION,
                },
                sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()

    def test_documented_model_free_invocation_writes_record_without_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            patches_root = tmp_root / "patches"
            patches_root.mkdir()
            # Real file: make_record() and patch_validation_subject_digest()
            # hash the actual subject bytes.
            (patches_root / "patch.py").write_text(
                'STATE = "untested"\nPATCHES = []\n', encoding="utf-8",
            )
            registry_stub = SimpleNamespace(root=patches_root, get=lambda _: _descriptor())
            # Real 1205 adapter (6 checks) -- only evaluate_check itself is
            # stubbed, so compute_verdict() runs for real.
            checks = patch_validation.parse_validation_toml(
                ROOT / "patches" / PATCH_ID / "validation.toml", patch_id=PATCH_ID,
            )
            plan = SimpleNamespace(
                patch_id=PATCH_ID,
                checks=checks,
                universal_capabilities=(),
                contracts=(),
                contract=None,
                required_capabilities=tuple({spec.capability for spec in checks}),
                contract_requirements=(),
            )
            evidence_by_build = {
                "tune": _build_evidence(0),
                "replay": _build_evidence(1),
                "stock": _build_evidence(2),
                "control": _build_evidence(3),
                "validation-subject": _build_evidence(4),
            }

            def fake_capture(build_dir, *args, **kwargs):
                return evidence_by_build[Path(build_dir).name]

            def fake_materialize(**kwargs):
                return Path(kwargs["worktree_root"])

            def fake_tree(source_dir):
                # Names as the fakes above produce them: materialize_*
                # return their own worktree_root arg (.../control,
                # .../subject); the stock stub ends in src-stock.
                return {
                    "control": CONTROL_TREE,
                    "subject": SUBJECT_TREE,
                    "src-stock": STOCK_TREE,
                }[Path(source_dir).name]

            def fake_evaluate(spec, ctx):
                # apply/build/activation/correctness pass; performance and
                # controls stay honestly BLOCKED (no real benchmark
                # evidence on this path) -- the adapter verdict is
                # therefore ineligible by design.
                status = "blocked" if spec.check_id in {"performance", "controls"} else "pass"
                return patch_validation.ValidationResult(
                    check_id=spec.check_id, capability=spec.capability,
                    status=status, summary="stub",
                )

            # Started as a list (not one giant `with (...)` chain) -- the
            # compiler rejects a statically-nested with this deep.
            load_registry = mock.patch(
                "bigcherry.patch.registry.load_registry", return_value=registry_stub,
            ).start()
            load_cfg = mock.patch(
                "bigcherry.core.config.load",
                return_value=SimpleNamespace(pinned="b10705"),
            ).start()
            require_pkg = mock.patch(
                "bigcherry.patch.validation_policy.require_execution_package",
                return_value=plan,
            ).start()
            mock.patch(
                "bigcherry.patch.source.resolve_source_composition",
                return_value=(BASE_REVISION, (PATCH_ID,)),
            ).start()
            mock.patch(
                "bigcherry.patch.source.materialize_composition",
                side_effect=fake_materialize,
            ).start()
            mock.patch(
                "bigcherry.patch.source.verify_composition_idempotent",
                return_value=True,
            ).start()
            mock.patch(
                "bigcherry.patch.source.materialize_stock_source",
                return_value=tmp_root / "worktree" / "src-stock",
            ).start()
            mock.patch(
                "bigcherry.patch.source.git_worktree_tree", side_effect=fake_tree,
            ).start()
            mock.patch(
                "bigcherry.patch.source.composition_digest",
                return_value=COMPOSITION_DIGEST,
            ).start()
            mock.patch(
                "bigcherry.patch.source.patch_implementation_digest",
                return_value=PATCH_DIGEST,
            ).start()
            mock.patch(
                "bigcherry.patch.validation.evaluate_check", side_effect=fake_evaluate,
            ).start()
            mock.patch(
                "bigcherry.patch.validation.load_contract_for_descriptor",
                return_value=None,
            ).start()
            write_record = mock.patch(
                "bigcherry.patch.evidence.write_record",
                return_value=tmp_root / "evidence" / f"{PATCH_ID}.json",
            ).start()
            # THE tripwire: if anything on this path constructs the
            # generic Campaign (with model=None/manifest=None) the
            # documented command is broken again.
            campaign_cls = mock.patch(
                "bigcherry.e2e_smoke_campaign.Campaign",
                side_effect=AssertionError(
                    "Campaign must not be constructed for --run-rd12-contract"
                ),
            ).start()
            mock.patch.object(vc, "generate_registry").start()
            mock.patch.object(vc, "build_tree", return_value=tmp_root / "bin").start()
            mock.patch.object(
                vc, "_full_requested_cmake_args", return_value=[],
            ).start()
            mock.patch.object(
                vc, "ensure_stock_baseline", return_value=tmp_root / "bin",
            ).start()
            mock.patch.object(
                vc, "capture_completed_build_evidence", side_effect=fake_capture,
            ).start()
            mock.patch.object(
                vc, "assert_validation_subject_parity", return_value=None,
            ).start()
            mock.patch.object(
                vc, "run_rd12_correctness_check", return_value=_rd12_qualification(),
            ).start()
            try:
                exit_code = vc.run(_args(tmp_root))
            finally:
                mock.patch.stopall()

        self.assertEqual(exit_code, 0)
        load_registry.assert_called_once()
        load_cfg.assert_called_once()
        require_pkg.assert_called_once()
        # MUST 4's whole point: the generic Campaign was never constructed.
        campaign_cls.assert_not_called()
        write_record.assert_called_once()
        record = write_record.call_args[0][0]

        expected_digest = self._expected_identity_digest()
        self.assertRegex(record["campaign_identity_digest"], r"^[0-9a-f]{64}$")
        # Determinism: the digest must be exactly the canonical-JSON sha256
        # over the shared-build facts -- re-derivable from the record.
        self.assertEqual(record["campaign_identity_digest"], expected_digest)
        self.assertEqual(
            record["correctness"]["campaign_identity_digest"], expected_digest,
        )
        # The record carries the five real shared builds: exactly the
        # {tune,replay,stock} campaign domain...
        self.assertEqual(
            {role: ident["effective_build_id"]
             for role, ident in record["campaign_build_identities"].items()},
            {
                "tune": _build_identity(0)["effective_build_id"],
                "replay": _build_identity(1)["effective_build_id"],
                "stock": _build_identity(2)["effective_build_id"],
            },
        )
        # ...and the producer's own validation-domain control/subject
        # identities (never the generic campaign's).
        self.assertEqual(record["validation_build_identities"], {
            "control": _build_identity(3),
            "subject": _build_identity(4),
        })
        # Correctness/activation reflect the real (canned) producer result.
        self.assertEqual(record["correctness"]["disposition"], "passed")
        self.assertEqual(record["correctness"]["mechanism"], "rd12-paired-mmvq-bit-identical")
        self.assertEqual(record["activation"]["status"], "executed")
        # Real adapter verdict (compute_verdict ran for real on the 1205
        # plan): performance/controls are honestly BLOCKED, so the record
        # is NOT eligible -- never a fabricated PASS.
        self.assertEqual(record["check_results"]["performance"]["status"], "blocked")
        self.assertEqual(record["check_results"]["controls"]["status"], "blocked")
        self.assertEqual(record["check_results"]["activation"]["status"], "pass")
        self.assertIs(record["final_eligibility"], False)
        self.assertEqual(record["validation_disposition"], "incomplete")
        # RD12 is a correctness-evidence producer only: no contracts, no
        # promotions.
        self.assertEqual(record["contracts"], [])
        self.assertEqual(record["contract_verdicts"], {})


if __name__ == "__main__":
    unittest.main()
