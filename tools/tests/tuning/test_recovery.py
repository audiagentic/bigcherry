"""HTR01: offline tests for the bounded recovery search -- the pure
BoundedDeltaDebugStrategy outcome matrix (no GPU/server needed), plus
AssignmentExecutor/run_recovery with a fully mocked executor so the
orchestration/budget/circuit-breaker logic is exercised without real
hardware. Matches test_behavioral_gate.py's own offline-first discipline.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.tuning import behavioral_gate as bg  # noqa: E402
from bigcherry.tuning import recovery as rec  # noqa: E402
from bigcherry import hi80_generate_correctness_evidence as hi80  # noqa: E402


def _trace(ids, draft_n=0, draft_n_accepted=0):
    return bg.BehavioralTrace(generated_token_ids=tuple(ids), draft_n=draft_n, draft_n_accepted=draft_n_accepted)


def _vector(name):
    return bg.BehavioralVector(name=name, prompt="p", n_predict=8)


def _assignment(dispatch, alternatives=()):
    return rec.SignatureAssignment(
        dispatch=dispatch, signature=f"sig-{dispatch}", current_candidate=f"cand-{dispatch}",
        native_candidate=f"family-{dispatch}:native:v1", alternatives=alternatives,
    )


class _FakeExecutor:
    """A real (not MagicMock) scripted executor with GROUND TRUTH about
    which exact (dispatch, candidate) combination(s) cause a failure --
    lets these tests drive the REAL BoundedPairedBisectionStrategy through
    the REAL run_recovery orchestrator and verify it actually converges to
    the correct answer, not just that its internal bookkeeping looks
    plausible in isolation (the class of bug the v1 pairing bug slipped
    through under)."""

    def __init__(self, fail_fn, ineligible=frozenset()):
        self.fail_fn = fail_fn
        self.ineligible = ineligible
        self.evaluate_log: list[dict] = []
        self.native_traces_captured = False

    def capture_native_traces(self, vectors):
        self.native_traces_captured = True

    def evaluate(self, proposal, *, full_corpus):
        self.evaluate_log.append(dict(proposal.overrides))
        for dispatch, candidate in proposal.overrides.items():
            if (dispatch, candidate) in self.ineligible:
                raise rec.RecoveryError(f"{candidate!r} ineligible for {dispatch!r}")
        verdict = "hard_fail" if self.fail_fn(proposal.overrides) else "pass"
        return rec.Observation(proposal=proposal, verdict=verdict, report=None)

    def validate_full_corpus(self, overrides, *, full_corpus):
        obs = self.evaluate(rec.AssignmentProposal(label="final", overrides=overrides), full_corpus=full_corpus)
        return rec.Observation(proposal=obs.proposal, verdict=obs.verdict, report=None, full_corpus_validated=True)

    def build_candidate_cache(self, overrides):
        return Path("fake.cache")


def _report_with_failing(name="v1"):
    report = bg.BehavioralGateReport()
    report.verdicts.append(bg.compare_traces(name, _trace([1]), _trace([2])))
    return report


class BoundedPairedBisectionStrategyRealConvergenceTests(unittest.TestCase):
    """GPT's mandatory pre-third-hardware-run checklist (session
    ses_330ae3c055084f38, 2026-08-30, after the v1 pairing bug was found
    on real hardware): drives the REAL strategy through the REAL
    run_recovery orchestrator against a scripted ground-truth executor,
    not unit-level state pokes -- this is what the v1 tests failed to
    catch (they exercised propose()/record() individually and never
    caught that a SECOND sibling's outcome fell through un-paired)."""

    def _assignments(self, n, alternatives_by_dispatch=None):
        alternatives_by_dispatch = alternatives_by_dispatch or {}
        return {
            f"d{i}": _assignment(f"d{i}", alternatives=alternatives_by_dispatch.get(f"d{i}", ()))
            for i in range(n)
        }

    def test_single_culprit_isolated_and_alternative_committed(self):
        # d3's OWN original candidate is the sole cause of failure; its one
        # real alternative is clean. Recovery must end up using that
        # alternative for d3 -- NOT native, and NOT touching any other
        # signature at all.
        assignments = self._assignments(8, {"d3": ("alt-d3",)})
        fail_fn = lambda overrides: overrides.get("d3") == "cand-d3"
        executor = _FakeExecutor(fail_fn)
        strategy = rec.BoundedPairedBisectionStrategy()
        result = rec.run_recovery(
            executor=executor, strategy=strategy, initial_assignments=assignments,
            initial_report=_report_with_failing(), full_corpus=[_vector("v1")],
            dispatch_hits=frozenset(assignments), max_evaluations=24,
        )
        self.assertTrue(result.published)
        self.assertEqual(result.final_overrides.get("d3"), "alt-d3")
        # log2(8) = 3 splits x 2 siblings = 6 isolation probes, + 1
        # baseline + 1 alternative trial + 1 final validation = 9 max.
        self.assertLessEqual(len(executor.evaluate_log), 10)

    def test_a_only_failure_recurses_into_a_not_b(self):
        assignments = self._assignments(4, {"d0": ("alt-d0",)})
        fail_fn = lambda overrides: overrides.get("d0") == "cand-d0"
        executor = _FakeExecutor(fail_fn)
        strategy = rec.BoundedPairedBisectionStrategy()
        result = rec.run_recovery(
            executor=executor, strategy=strategy, initial_assignments=assignments,
            initial_report=_report_with_failing(), full_corpus=[_vector("v1")],
            dispatch_hits=frozenset(assignments), max_evaluations=24,
        )
        self.assertTrue(result.published)
        self.assertEqual(result.final_overrides.get("d0"), "alt-d0")
        self.assertNotIn("d1", result.final_overrides)
        self.assertNotIn("d2", result.final_overrides)
        self.assertNotIn("d3", result.final_overrides)

    def test_both_halves_independently_fail_both_get_isolated(self):
        # d0 AND d2 are each independently sufficient to cause failure.
        assignments = self._assignments(4)  # no alternatives -- must fall back to native for both
        fail_fn = lambda overrides: overrides.get("d0") == "cand-d0" or overrides.get("d2") == "cand-d2"
        executor = _FakeExecutor(fail_fn)
        strategy = rec.BoundedPairedBisectionStrategy()
        result = rec.run_recovery(
            executor=executor, strategy=strategy, initial_assignments=assignments,
            initial_report=_report_with_failing(), full_corpus=[_vector("v1")],
            dispatch_hits=frozenset(assignments), max_evaluations=24,
        )
        self.assertTrue(result.published)
        self.assertEqual(result.final_overrides.get("d0"), "family-d0:native:v1")
        self.assertEqual(result.final_overrides.get("d2"), "family-d2:native:v1")

    def test_cross_half_interaction_neither_half_blamed_alone(self):
        # Failure requires d0 AND d1 BOTH active simultaneously -- each
        # alone (the other forced native) must PASS, but their union fails.
        assignments = self._assignments(2)
        fail_fn = lambda overrides: overrides.get("d0") == "cand-d0" and overrides.get("d1") == "cand-d1"
        executor = _FakeExecutor(fail_fn)
        strategy = rec.BoundedPairedBisectionStrategy()
        result = rec.run_recovery(
            executor=executor, strategy=strategy, initial_assignments=assignments,
            initial_report=_report_with_failing(), full_corpus=[_vector("v1")],
            dispatch_hits=frozenset(assignments), max_evaluations=24,
        )
        self.assertTrue(result.published)
        # Both individually pass, so bisection reports the pair as an
        # INTERACTION group -- neither is individually blamed, but since
        # neither has an alternative, both still end up conservatively
        # forced to native (the safe, if not maximally efficient, outcome).
        self.assertEqual(result.final_overrides.get("d0"), "family-d0:native:v1")
        self.assertEqual(result.final_overrides.get("d1"), "family-d1:native:v1")

    def test_second_sibling_evaluated_against_identical_parent_baseline(self):
        assignments = self._assignments(4)
        fail_fn = lambda overrides: False  # never fails -- just inspect the probes themselves
        executor = _FakeExecutor(fail_fn)
        strategy = rec.BoundedPairedBisectionStrategy()
        rec.run_recovery(
            executor=executor, strategy=strategy, initial_assignments=assignments,
            initial_report=_report_with_failing(), full_corpus=[_vector("v1")],
            dispatch_hits=frozenset(assignments), max_evaluations=24,
        )
        # The very first action is a 2-proposal split -- both siblings must
        # cover the exact same universe of dispatches (H), differing only
        # in which half is "active" vs forced native.
        first_two = executor.evaluate_log[:2]
        self.assertEqual(set(first_two[0]), set(first_two[1]))

    def test_diagnostic_pass_never_commits_overrides_mid_isolation(self):
        # d0 is guilty; while isolating, several diagnostic PASSes occur
        # (every probe that excludes d0). None of those may leave any
        # trace in committed_overrides -- only the repair phase may.
        assignments = self._assignments(8, {"d0": ("alt-d0",)})
        fail_fn = lambda overrides: overrides.get("d0") == "cand-d0"
        executor = _FakeExecutor(fail_fn)
        strategy = rec.BoundedPairedBisectionStrategy()
        state = rec.RecoveryState(
            assignments=assignments, failing_vectors=(_vector("v1"),),
            dispatch_hits=frozenset(assignments), evaluated=(), remaining_budget=23,
        )
        # Manually drive one isolation split and confirm a PASS sibling
        # does not touch committed_overrides.
        action = strategy.propose(state)
        observations = tuple(
            executor.evaluate(p, full_corpus=[]) for p in action.proposals
        )
        state = strategy.record(state, rec.ActionObservation(action=action, observations=observations))
        self.assertEqual(state.committed_overrides, {})

    def test_alternative_pass_commits_only_through_record(self):
        assignments = {"d0": _assignment("d0", alternatives=("alt-d0",))}
        fail_fn = lambda overrides: overrides.get("d0") == "cand-d0"
        executor = _FakeExecutor(fail_fn)
        strategy = rec.BoundedPairedBisectionStrategy()
        result = rec.run_recovery(
            executor=executor, strategy=strategy, initial_assignments=assignments,
            initial_report=_report_with_failing(), full_corpus=[_vector("v1")],
            dispatch_hits=frozenset(assignments), max_evaluations=24,
        )
        self.assertEqual(result.final_overrides.get("d0"), "alt-d0")

    def test_budget_cannot_start_a_split_with_only_one_remaining(self):
        assignments = self._assignments(4)
        fail_fn = lambda overrides: True  # irrelevant -- budget should stop first
        executor = _FakeExecutor(fail_fn)
        strategy = rec.BoundedPairedBisectionStrategy()
        # max_evaluations=2 -> after RESERVE_FINAL_VALIDATION(1), only 1
        # evaluation of real budget -- cannot afford a 2-proposal split.
        result = rec.run_recovery(
            executor=executor, strategy=strategy, initial_assignments=assignments,
            initial_report=_report_with_failing(), full_corpus=[_vector("v1")],
            dispatch_hits=frozenset(assignments), max_evaluations=2,
        )
        # No 2-proposal SPLIT should ever have been attempted (unaffordable
        # at budget=1) -- but the cheap 1-proposal repair-baseline probe
        # IS affordable and correctly still runs, plus the final
        # validation: 2 evaluate() calls total, never a pair.
        self.assertEqual(len(executor.evaluate_log), 2)
        self.assertFalse(result.published)

    def test_unstable_sibling_aborts_recovery_entirely(self):
        assignments = self._assignments(4)

        class _UnstableExecutor(_FakeExecutor):
            def evaluate(self, proposal, *, full_corpus):
                return rec.Observation(proposal=proposal, verdict="unstable", report=None)

        executor = _UnstableExecutor(lambda o: False)
        strategy = rec.BoundedPairedBisectionStrategy()
        with self.assertRaises(rec.RecoveryError):
            rec.run_recovery(
                executor=executor, strategy=strategy, initial_assignments=assignments,
                initial_report=_report_with_failing(), full_corpus=[_vector("v1")],
                dispatch_hits=frozenset(assignments), max_evaluations=24,
            )

    def test_ineligible_native_probe_is_a_structural_error(self):
        assignments = self._assignments(4)
        executor = _FakeExecutor(lambda o: False, ineligible={(d, a.native_candidate) for d, a in assignments.items()})
        strategy = rec.BoundedPairedBisectionStrategy()
        with self.assertRaises(rec.RecoveryError):
            rec.run_recovery(
                executor=executor, strategy=strategy, initial_assignments=assignments,
                initial_report=_report_with_failing(), full_corpus=[_vector("v1")],
                dispatch_hits=frozenset(assignments), max_evaluations=24,
            )


class RunRecoveryOrchestrationTests(unittest.TestCase):
    """Fully mocked executor -- exercises budget accounting, the "collect
    ALL failing vectors up front" requirement, and the mandatory final
    full-corpus validation before anything is treated as publishable."""

    def _initial_report(self, failing_names):
        report = bg.BehavioralGateReport()
        for name in failing_names:
            report.verdicts.append(bg.compare_traces(name, _trace([1]), _trace([2])))
        return report

    def test_raises_if_no_failing_vectors_in_initial_report(self):
        executor = MagicMock()
        strategy = rec.BoundedPairedBisectionStrategy()
        report = bg.BehavioralGateReport(verdicts=[
            bg.compare_traces("v1", _trace([1]), _trace([1]))  # exact_pass
        ])
        with self.assertRaises(rec.RecoveryError):
            rec.run_recovery(
                executor=executor, strategy=strategy, initial_assignments={},
                initial_report=report, full_corpus=[_vector("v1")],
                dispatch_hits=frozenset(),
            )

    def test_falls_back_to_native_for_all_implicated_when_nothing_recovers(self):
        assignments = {"d0": _assignment("d0")}  # no alternatives -- must fall back to native
        executor = MagicMock()

        def fake_evaluate(proposal, *, full_corpus):
            return rec.Observation(proposal=proposal, verdict="hard_fail", report=None)

        native_name = assignments["d0"].native_candidate

        def fake_validate(overrides, *, full_corpus):
            # The final native-fallback assignment must be accepted.
            verdict = "pass" if overrides == {"d0": native_name} else "hard_fail"
            return rec.Observation(
                proposal=rec.AssignmentProposal(label="final", overrides=overrides),
                verdict=verdict, report=None, full_corpus_validated=True,
            )

        executor.evaluate.side_effect = fake_evaluate
        executor.validate_full_corpus.side_effect = fake_validate
        executor.build_candidate_cache.return_value = Path("fake.cache")

        strategy = rec.BoundedPairedBisectionStrategy()
        report = self._initial_report(["v1"])
        result = rec.run_recovery(
            executor=executor, strategy=strategy, initial_assignments=assignments,
            initial_report=report, full_corpus=[_vector("v1")], dispatch_hits=frozenset(["d0"]),
            max_evaluations=6,
        )
        self.assertTrue(result.published)
        self.assertEqual(result.final_overrides, {"d0": native_name})

    def test_final_validation_failure_means_not_published(self):
        assignments = {"d0": _assignment("d0")}
        executor = MagicMock()
        executor.evaluate.side_effect = lambda proposal, *, full_corpus: rec.Observation(
            proposal=proposal, verdict="hard_fail", report=None,
        )
        executor.validate_full_corpus.side_effect = lambda overrides, *, full_corpus: rec.Observation(
            proposal=rec.AssignmentProposal(label="final", overrides=overrides),
            verdict="hard_fail", report=None, full_corpus_validated=True,
        )
        strategy = rec.BoundedPairedBisectionStrategy()
        report = self._initial_report(["v1"])
        result = rec.run_recovery(
            executor=executor, strategy=strategy, initial_assignments=assignments,
            initial_report=report, full_corpus=[_vector("v1")], dispatch_hits=frozenset(["d0"]),
            max_evaluations=4,
        )
        self.assertFalse(result.published)
        self.assertIsNone(result.cache_path)

    def test_collects_all_failing_vectors_not_just_first(self):
        assignments = {"d0": _assignment("d0")}
        executor = MagicMock()
        seen_probe_vectors = []

        def fake_evaluate(proposal, *, full_corpus):
            seen_probe_vectors.append(proposal.probe_vectors)
            return rec.Observation(proposal=proposal, verdict="hard_fail", report=None)

        executor.evaluate.side_effect = fake_evaluate
        executor.validate_full_corpus.side_effect = lambda overrides, *, full_corpus: rec.Observation(
            proposal=rec.AssignmentProposal(label="final", overrides=overrides),
            verdict="pass", report=None, full_corpus_validated=True,
        )
        executor.build_candidate_cache.return_value = Path("fake.cache")
        strategy = rec.BoundedPairedBisectionStrategy()
        report = self._initial_report(["v1", "v2"])
        rec.run_recovery(
            executor=executor, strategy=strategy, initial_assignments=assignments,
            initial_report=report, full_corpus=[_vector("v1"), _vector("v2")],
            dispatch_hits=frozenset(["d0"]), max_evaluations=4,
        )
        # Every probe should have been offered both failing vectors, not
        # just the first one encountered.
        for probe in seen_probe_vectors:
            names = {v.name for v in probe}
            self.assertEqual(names, {"v1", "v2"})


class EnsureCorrectnessEvidenceTests(unittest.TestCase):
    """HTR01 (2026-08-30, locked design): AssignmentExecutor's lazy
    correctness-qualification budget/skip logic, with generate_for_candidate
    mocked (no real GPU/test-backend-ops needed to exercise the wiring)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        workdir = Path(self._tmp.name)
        measurements_path = workdir / "promoted.jsonl"
        row = {"dispatch": "d0", "signature": "sig0", "native": "family:native:v1",
               "winner": "family:alt1:v1", "promotion_status": "promoted"}
        measurements_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        dispatch_db = workdir / "tune.sqlite"
        dispatch_db.write_bytes(b"")
        self.executor = rec.AssignmentExecutor(
            binary_path=Path("llama-server"), model_path=Path("model.gguf"), devices="0",
            common_args=(), measurements_path=measurements_path,
            manifest_path=workdir / "manifest.json", ggml_h_path=workdir / "ggml.h",
            workdir=workdir, dispatch_db=dispatch_db,
            correctness_binary_path=Path("test-backend-ops"), vendor_root=workdir,
            max_new_correctness_candidates=2,
        )
        self.addCleanup(self._close_dispatch_db_conn)

    def _close_dispatch_db_conn(self):
        if self.executor._dispatch_db_conn is not None:
            self.executor._dispatch_db_conn.close()

    def test_native_candidate_never_calls_generate_for_candidate(self):
        with (
            patch.object(hi80, "generate_for_candidate") as mock_gen,
            patch.object(self.executor, "build_candidate_cache", return_value=Path("fake.cache")),
            patch("bigcherry.tuning.recovery.ServerRunner"),
        ):
            proposal = rec.AssignmentProposal(label="x", overrides={"d0": "family:native:v1"})
            try:
                self.executor.evaluate(proposal, full_corpus=[])
            except Exception:
                pass  # only the pre-cache correctness-qualification loop is under test here
        mock_gen.assert_not_called()

    def test_dispatchable_alternative_is_accepted_and_counts_against_budget(self):
        result = hi80.EvidenceGenerationResult(
            evidence_id=1, status="generated", dispatchable=True, subprocess_runs=6,
        )
        with patch.object(hi80, "generate_for_candidate", return_value=result):
            self.executor.ensure_correctness_evidence("d0", "family:alt2:v1")
        self.assertEqual(self.executor._correctness_candidates_used, 1)

    def test_non_dispatchable_alternative_raises_recovery_error(self):
        result = hi80.EvidenceGenerationResult(
            evidence_id=1, status="generated", dispatchable=False, subprocess_runs=6,
        )
        with patch.object(hi80, "generate_for_candidate", return_value=result):
            with self.assertRaises(rec.RecoveryError):
                self.executor.ensure_correctness_evidence("d0", "family:alt2:v1")

    def test_existing_evidence_does_not_consume_budget(self):
        result = hi80.EvidenceGenerationResult(
            evidence_id=1, status="existing", dispatchable=True, subprocess_runs=0,
        )
        with patch.object(hi80, "generate_for_candidate", return_value=result):
            self.executor.ensure_correctness_evidence("d0", "family:alt2:v1")
        self.assertEqual(self.executor._correctness_candidates_used, 0)

    def test_budget_exhaustion_raises_without_calling_generate_for_candidate_again(self):
        generated = hi80.EvidenceGenerationResult(
            evidence_id=1, status="generated", dispatchable=True, subprocess_runs=6,
        )
        with patch.object(hi80, "generate_for_candidate", return_value=generated) as mock_gen:
            self.executor.ensure_correctness_evidence("d0", "family:alt2:v1")
            self.executor.ensure_correctness_evidence("d0", "family:alt3:v1")
            self.assertEqual(mock_gen.call_count, 2)
            with self.assertRaises(rec.RecoveryError):
                self.executor.ensure_correctness_evidence("d0", "family:alt4:v1")
            # Budget was exhausted BEFORE a third real generation call.
            self.assertEqual(mock_gen.call_count, 2)


if __name__ == "__main__":
    unittest.main()
