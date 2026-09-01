"""VA18 slice 1: evidence schema v4 (plural contract identity/verdicts)
and the fail-closed plural contract-identity resolution fix.

GPT spec (session ses_d1759a9471d443d5, following up on the VA17 review
thread after ses_5bbee8ce5c9a4265 went idle-timeout stale): schema v4
replaces the singular contract_id/contract_hash fields with a canonical
sorted contracts=[{id,hash},...] array plus contract_verdicts keyed by
bound contract id; a validated-state record must have the EXACT current
contract set (missing/extra/stale-hash on even one contract is
nonqualification for the whole patch), a complete verdict set, and every
verdict passed==true; v1-v3 remain readable only for a CURRENT 0/1-
contract patch, never a multi-contract one; ported-benched stays
qualifiable with BLOCKED contract verdicts (contract PASS is required
only for validated promotion, never ported-benched); the broad
`except (OSError, ValueError, KeyError)` that used to silently swallow a
real multi-contract registry error is now narrowed to exclude
ConfigurationError.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import evidence as pve  # noqa: E402
from bigcherry.patch import patchset  # noqa: E402
from bigcherry.patch.activation import ActivationEvidence  # noqa: E402

_HEX64 = "a" * 64
_HEX40 = "b" * 40


def _build_identity(tag: str = "") -> dict:
    return {
        "effective_build_id": f"eff{tag}", "compile_verification_id": f"cv{tag}",
        "compile_commands_digest": f"ccd{tag}", "hip_compile_commands_digest": f"hccd{tag}",
        "runtime_bundle_hash": f"rbh{tag}", "runtime_artifacts": {"llama-server": "a" * 64},
    }


class MakeRecordV4ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.patch_path = self.root / "9999_example.py"
        self.patch_path.write_text('STATE = "untested"\n', encoding="utf-8")
        self.workdir = self.root / "campaign"
        self.workdir.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _make(self, **kwargs) -> dict:
        activation_evidence = ActivationEvidence(status="executed", mechanism="m", detail="d")
        correctness = {"schema_version": 1, "disposition": "passed", "mechanism": "m", "detail": "d"}
        return pve.make_record(
            patch_id="9999_example", patch_path=self.patch_path,
            patch_implementation_digest=_HEX64, base_ref="b10502", base_revision=_HEX40,
            framework_baseline_digest=_HEX64, patched_source_tree=_HEX40,
            gpu_architectures="gfx1100", activation_evidence=activation_evidence,
            activation_disposition="activation-verified", correctness=correctness,
            campaign_identity_digest=_HEX64,
            build_identities={"tune": _build_identity("1"), "replay": _build_identity("2"),
                               "stock": _build_identity("3")},
            validation_build_identities={"control": _build_identity("4"), "subject": _build_identity("5")},
            campaign_workdir=self.workdir, **kwargs,
        )

    def test_contract_ordering_is_canonical_regardless_of_input_order(self) -> None:
        record_a = self._make(contracts=[{"id": "RD07", "hash": "h7"}, {"id": "RD05", "hash": "h5"}])
        record_b = self._make(contracts=[{"id": "RD05", "hash": "h5"}, {"id": "RD07", "hash": "h7"}])
        self.assertEqual(record_a["contracts"], record_b["contracts"])
        self.assertEqual(record_a["contracts"], [{"id": "RD05", "hash": "h5"}, {"id": "RD07", "hash": "h7"}])

    def test_contract_id_and_contracts_are_mutually_exclusive(self) -> None:
        with self.assertRaises(pve.ValidationEvidenceError):
            self._make(contract_id="RD05", contracts=[{"id": "RD06", "hash": "h6"}])

    def test_contract_verdicts_rejects_unbound_id(self) -> None:
        with self.assertRaises(pve.ValidationEvidenceError):
            self._make(
                contracts=[{"id": "RD05", "hash": "h5"}],
                contract_verdicts={"RD06": {"passed": True}},
            )

    def test_contract_verdicts_requires_boolean_passed_field(self) -> None:
        with self.assertRaises(pve.ValidationEvidenceError):
            self._make(
                contracts=[{"id": "RD05", "hash": "h5"}],
                contract_verdicts={"RD05": {"passed": "yes"}},
            )

    def test_bound_contract_with_no_verdict_is_not_a_write_time_error(self) -> None:
        # A bound contract with no produced verdict yet is legal at write
        # time (the caller may not have run that contract's executor) --
        # it reads as incomplete/BLOCKED at qualification time, never an
        # inferred PASS.
        record = self._make(contracts=[{"id": "RD05", "hash": "h5"}, {"id": "RD06", "hash": "h6"}])
        self.assertEqual(record["contract_verdicts"], {})
        self.assertEqual(record["record_schema_version"], 4)


def _module(
    patch_path: Path, *, state: str = "validated", catalog_root: Path | None = None,
) -> patchset.PatchModule:
    return patchset.PatchModule(
        patch_id="9999_example", path=patch_path, order=0, group="g", state=state,
        upstream=None, content_hash="deadbeef" * 8, catalog_root=catalog_root,
    )


class RecordQualifiesV4MultiContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.patch_path = self.root / "9999_example.py"
        self.patch_path.write_text('STATE = "validated"\n', encoding="utf-8")
        self.subject_digest = pve.patch_validation_subject_digest(self.patch_path)
        self.module = _module(self.patch_path)
        self.contracts = (
            {"id": "RD05", "hash": "h5"}, {"id": "RD06", "hash": "h6"}, {"id": "RD07", "hash": "h7"},
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _record(self, *, contracts=None, contract_verdicts=None, record_schema_version=4) -> dict:
        base = {
            "record_schema_version": record_schema_version, "validation_contract_version": "hi83-v1",
            "patch_id": "9999_example", "patch_validation_subject_digest": self.subject_digest,
            "base_ref": "b10502", "validation_disposition": "validated",
            "eligible_for_validated_state": True,
            "representation": "packaged", "validation_implementation_digest": "d" * 64,
            "baseline_composition": {"x": 1}, "control_composition": {"x": 1},
            "subject_composition": {"x": 1}, "control_tree": "t", "subject_tree": "t",
            "stock_tree": "t", "check_results": {"x": 1}, "hardware": {"architectures": ["gfx1100"]},
            "artifact_hashes": {"a": "b"}, "blockers": [], "final_eligibility": True,
            "activation": {"status": "executed", "disposition": "activation-verified"},
            "correctness": {"disposition": "passed"},
            "patch_implementation_digest": _HEX64, "base_revision": _HEX40,
            "framework_baseline_digest": _HEX64, "patched_source_tree": _HEX40,
            "campaign_identity_digest": _HEX64,
            "campaign_build_identities": {"tune": _build_identity("1"), "replay": _build_identity("2"),
                                           "stock": _build_identity("3")},
            "validation_build_identities": {"control": _build_identity("4"), "subject": _build_identity("5")},
            "gpu_architectures": ["gfx1100"],
        }
        if contracts is not None:
            base["contracts"] = contracts
        if contract_verdicts is not None:
            base["contract_verdicts"] = contract_verdicts
        base["record_digest"] = pve._record_digest(base)
        return base

    def _all_pass_verdicts(self) -> dict:
        return {c["id"]: {"passed": True} for c in self.contracts}

    def test_exact_three_contract_set_with_all_pass_qualifies(self) -> None:
        record = self._record(contracts=list(self.contracts), contract_verdicts=self._all_pass_verdicts())
        ok, problems = pve._record_qualifies(
            record, module=self.module, pinned_ref="b10502", subject_digest=self.subject_digest,
            contracts=self.contracts,
        )
        self.assertTrue(ok, problems)

    def test_missing_contract_from_record_rejects(self) -> None:
        record = self._record(
            contracts=[self.contracts[0], self.contracts[1]],  # RD07 missing
            contract_verdicts={self.contracts[0]["id"]: {"passed": True}, self.contracts[1]["id"]: {"passed": True}},
        )
        ok, problems = pve._record_qualifies(
            record, module=self.module, pinned_ref="b10502", subject_digest=self.subject_digest,
            contracts=self.contracts,
        )
        self.assertFalse(ok)
        self.assertTrue(any("contract identity set is stale" in p for p in problems))

    def test_extra_contract_in_record_rejects(self) -> None:
        record = self._record(
            contracts=list(self.contracts) + [{"id": "RD99", "hash": "h9"}],
            contract_verdicts={**self._all_pass_verdicts(), "RD99": {"passed": True}},
        )
        ok, problems = pve._record_qualifies(
            record, module=self.module, pinned_ref="b10502", subject_digest=self.subject_digest,
            contracts=self.contracts,
        )
        self.assertFalse(ok)
        self.assertTrue(any("contract identity set is stale" in p for p in problems))

    def test_stale_hash_on_one_contract_rejects(self) -> None:
        stale = [dict(self.contracts[0]), self.contracts[1], self.contracts[2]]
        stale[0]["hash"] = "STALE"
        record = self._record(contracts=stale, contract_verdicts=self._all_pass_verdicts())
        ok, problems = pve._record_qualifies(
            record, module=self.module, pinned_ref="b10502", subject_digest=self.subject_digest,
            contracts=self.contracts,
        )
        self.assertFalse(ok)
        self.assertTrue(any("contract identity set is stale" in p for p in problems))

    def test_missing_verdict_for_one_bound_contract_rejects(self) -> None:
        verdicts = self._all_pass_verdicts()
        del verdicts[self.contracts[2]["id"]]  # RD07 has no verdict at all
        record = self._record(contracts=list(self.contracts), contract_verdicts=verdicts)
        ok, problems = pve._record_qualifies(
            record, module=self.module, pinned_ref="b10502", subject_digest=self.subject_digest,
            contracts=self.contracts,
        )
        self.assertFalse(ok)
        self.assertTrue(any("missing contract_verdicts" in p for p in problems))

    def test_blocked_verdict_on_one_contract_rejects_validated_qualification(self) -> None:
        verdicts = self._all_pass_verdicts()
        verdicts[self.contracts[1]["id"]] = {"passed": False, "status": "blocked"}
        record = self._record(contracts=list(self.contracts), contract_verdicts=verdicts)
        ok, problems = pve._record_qualifies(
            record, module=self.module, pinned_ref="b10502", subject_digest=self.subject_digest,
            contracts=self.contracts,
        )
        self.assertFalse(ok)
        self.assertTrue(any("contract_verdicts did not pass" in p for p in problems))

    def test_failing_verdict_on_one_contract_rejects_validated_qualification(self) -> None:
        verdicts = self._all_pass_verdicts()
        verdicts[self.contracts[2]["id"]] = {"passed": False, "status": "fail"}
        record = self._record(contracts=list(self.contracts), contract_verdicts=verdicts)
        ok, problems = pve._record_qualifies(
            record, module=self.module, pinned_ref="b10502", subject_digest=self.subject_digest,
            contracts=self.contracts,
        )
        self.assertFalse(ok)
        self.assertTrue(any("contract_verdicts did not pass" in p for p in problems))

    def test_v1_record_still_qualifies_a_current_zero_contract_patch(self) -> None:
        record = self._record(record_schema_version=1)
        # v1 has no contracts/contract_verdicts fields, and no build_identities
        # split concept -- add the legacy shape it needs.
        record["build_identities"] = record.pop("campaign_build_identities")
        del record["validation_build_identities"]
        record["record_digest"] = pve._record_digest(record)
        ok, problems = pve._record_qualifies(
            record, module=self.module, pinned_ref="b10502", subject_digest=self.subject_digest,
            contracts=(),
        )
        self.assertTrue(ok, problems)

    def test_v3_record_rejected_for_a_current_multi_contract_patch(self) -> None:
        record = self._record(record_schema_version=3)
        record["contract_id"] = self.contracts[0]["id"]
        record["contract_hash"] = self.contracts[0]["hash"]
        record["record_digest"] = pve._record_digest(record)
        ok, problems = pve._record_qualifies(
            record, module=self.module, pinned_ref="b10502", subject_digest=self.subject_digest,
            contracts=self.contracts,  # patch is CURRENTLY bound to 3 contracts
        )
        self.assertFalse(ok)
        self.assertTrue(any("cannot qualify a current" in p for p in problems))


class RecordQualifiesForBenchedV4Tests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.patch_path = self.root / "9999_example.py"
        self.patch_path.write_text('STATE = "untested"\n', encoding="utf-8")
        self.subject_digest = pve.patch_validation_subject_digest(self.patch_path)
        self.module = _module(self.patch_path, state="untested")
        self.contracts = ({"id": "RD05", "hash": "h5"}, {"id": "RD06", "hash": "h6"})

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_v4_record_with_blocked_verdicts_still_qualifies_ported_benched(self) -> None:
        # VA18: contract PASS is required only for validated promotion --
        # ported-benched only checks that the contract IDENTITY is fresh,
        # never that any bound contract's verdict actually passed.
        artifact_path = self.root / "performance.json"
        artifact_path.write_text("{}", encoding="utf-8")
        import hashlib
        sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        record = {
            "record_schema_version": 4, "patch_id": "9999_example",
            "patch_validation_subject_digest": self.subject_digest, "base_ref": "b10502",
            "validation_implementation_digest": "d" * 64,
            "contracts": list(self.contracts),
            "contract_verdicts": {"RD05": {"passed": False, "status": "blocked"}},
            "validation_build_identities": {"control": _build_identity("c"), "subject": _build_identity("s")},
            "hardware": {"architectures": ["gfx1100"]},
            "correctness": {"disposition": "unknown"},
            "artifact_hashes": {"performance.json": sha},
            "check_results": {
                "controls": {"capability": "controls", "status": "pass", "artifacts": [
                    {"path": "performance.json", "sha256": sha},
                ]},
            },
            "patch_implementation_digest": _HEX64, "base_revision": _HEX40,
            "campaign_identity_digest": _HEX64,
        }
        record["record_digest"] = pve._record_digest(record)
        ok, problems = pve._record_qualifies_for_benched(
            record, module=self.module, pinned_ref="b10502", subject_digest=self.subject_digest,
            contracts=self.contracts,
        )
        self.assertTrue(ok, problems)

    def test_v4_stale_contract_set_rejects_ported_benched(self) -> None:
        record = {
            "record_schema_version": 4, "patch_id": "9999_example",
            "patch_validation_subject_digest": self.subject_digest, "base_ref": "b10502",
            "validation_implementation_digest": "d" * 64,
            "contracts": [{"id": "RD05", "hash": "h5"}],  # RD06 missing vs. current bound set
            "contract_verdicts": {},
            "validation_build_identities": {"control": _build_identity("c"), "subject": _build_identity("s")},
            "hardware": {"architectures": ["gfx1100"]},
            "correctness": {"disposition": "unknown"},
            "artifact_hashes": {},
            "check_results": {},
            "patch_implementation_digest": _HEX64, "base_revision": _HEX40,
            "campaign_identity_digest": _HEX64,
        }
        record["record_digest"] = pve._record_digest(record)
        ok, problems = pve._record_qualifies_for_benched(
            record, module=self.module, pinned_ref="b10502", subject_digest=self.subject_digest,
            contracts=self.contracts,
        )
        self.assertFalse(ok)
        self.assertTrue(any("contract identity set is stale" in p for p in problems))


class ContractIdentityResolutionNotSwallowedTests(unittest.TestCase):
    """Regression: a real registry/contract-resolution failure for a
    multi-contract patch must propagate, never be silently swallowed into
    empty contract identity (the bug found while scoping this slice)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_package(self, contract_ids: list[str]) -> Path:
        package_dir = self.root / "9999_example"
        package_dir.mkdir(parents=True)
        (package_dir / "patch.py").write_text('STATE = "validated"\n', encoding="utf-8")
        contracts_toml = ", ".join(f'"{c}"' for c in contract_ids)
        (package_dir / "patch.toml").write_text(
            "schema = 1\n"
            'id = "9999_example"\n'
            "order = 9999\n"
            'group = "test"\n'
            'state = "validated"\n'
            'kind = "enhancement"\n'
            'origin = "external-fork"\n'
            'backend = "hip"\n'
            f"experiment-contracts = [{contracts_toml}]\n",
            encoding="utf-8",
        )
        return package_dir

    def test_unresolvable_multi_contract_reference_propagates_not_swallowed(self) -> None:
        # A patch bound to two REAL, registrable contract ids (the registry
        # load itself must succeed -- an unregistrable id is caught even
        # earlier, at patch_registry.load_registry() time, which is a
        # different, legitimate "legacy/synthetic caller" tolerance this
        # fix does not touch). The real bug this regression proves: once
        # the descriptor resolves, a genuine bind_contract()/
        # load_contracts_for_descriptor() failure (validation.py's
        # ConfigurationError, a ValueError subclass) must propagate out of
        # _resolve_contract_identities()/verify_validated_patch(), never
        # be silently swallowed into empty contract identity the way the
        # old broad `except (OSError, ValueError, KeyError)` did.
        self._write_package(["RD05-WMMA-FA-CORRECTNESS-BARRIERS", "RD06-RDNA4-WMMA-FA-CONFIG"])
        module = _module(self.root / "9999_example" / "patch.py", catalog_root=self.root)

        from bigcherry.patch import validation as patch_validation

        original = patch_validation.bind_contract

        def _boom(contract):
            raise patch_validation.ConfigurationError("synthetic bind_contract failure for regression test")

        patch_validation.bind_contract = _boom
        try:
            with self.assertRaises(patch_validation.ConfigurationError) as ctx:
                pve.verify_validated_patch(module, pinned_ref="b10502", root=self.root)
        finally:
            patch_validation.bind_contract = original
        self.assertIn("synthetic bind_contract failure", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
