"""HTR01: offline tests for the bounded recovery search -- the pure
BoundedDeltaDebugStrategy outcome matrix (no GPU/server needed), plus
AssignmentExecutor/run_recovery with a fully mocked executor so the
orchestration/budget/circuit-breaker logic is exercised without real
hardware. Matches test_behavioral_gate.py's own offline-first discipline.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.tuning import behavioral_gate as bg  # noqa: E402
from bigcherry.tuning import recovery as rec  # noqa: E402


def _trace(ids, draft_n=0, draft_n_accepted=0):
    return bg.BehavioralTrace(generated_token_ids=tuple(ids), draft_n=draft_n, draft_n_accepted=draft_n_accepted)


def _vector(name):
    return bg.BehavioralVector(name=name, prompt="p", n_predict=8)


def _assignment(dispatch, alternatives=()):
    return rec.SignatureAssignment(
        dispatch=dispatch, signature=f"sig-{dispatch}", current_candidate=f"cand-{dispatch}",
        alternatives=alternatives,
    )


class BoundedDeltaDebugStrategyOutcomeMatrixTests(unittest.TestCase):
    """GPT's exact outcome matrix (adversarial review, ses_330ae3c055084f38,
    2026-08-29): union FAIL keeps splitting; A fails/B passes recurses into
    A; both fail tracks as independent groups; both pass but union fails is
    an interaction group (bisection must stop, neither half is blamed);
    unstable/non-deterministic aborts recovery entirely."""

    def _state(self, assignments, hits, evaluated=(), budget=24):
        return rec.RecoveryState(
            assignments=assignments, failing_vectors=(_vector("v1"),),
            dispatch_hits=frozenset(hits), evaluated=evaluated, remaining_budget=budget,
        )

    def test_initial_propose_splits_full_implicated_pool_in_half(self):
        assignments = {f"d{i}": _assignment(f"d{i}") for i in range(4)}
        strategy = rec.BoundedDeltaDebugStrategy()
        state = self._state(assignments, assignments.keys())
        proposals = strategy.propose(state)
        self.assertEqual(len(proposals), 2)
        touched = set(proposals[0].overrides) | set(proposals[1].overrides)
        self.assertEqual(touched, set(assignments))
        # Every override in a bisection proposal forces native -- that's
        # what "swap a half back to native and re-probe" means.
        for p in proposals:
            self.assertTrue(all(v == "native" for v in p.overrides.values()))

    def test_minimal_singleton_group_proposes_alternatives_not_further_split(self):
        assignments = {"d0": _assignment("d0", alternatives=("alt-a", "alt-b"))}
        strategy = rec.BoundedDeltaDebugStrategy()
        strategy._initialized = True
        strategy._pending_groups = [("d0",)]
        state = self._state(assignments, ["d0"])
        proposals = strategy.propose(state)
        labels = {p.label for p in proposals}
        self.assertIn("alt:d0:alt-a", labels)
        self.assertIn("alt:d0:alt-b", labels)
        self.assertIn("native:d0", labels)

    def test_half_fails_half_passes_recurses_into_failing_half(self):
        assignments = {f"d{i}": _assignment(f"d{i}") for i in range(4)}
        strategy = rec.BoundedDeltaDebugStrategy()
        state = self._state(assignments, assignments.keys())
        proposals = strategy.propose(state)  # seeds _pending_groups with the full pool
        half_a = proposals[0]
        # half_a fails -> strategy should keep it queued for further splitting
        observation = rec.Observation(proposal=half_a, verdict="hard_fail", report=None)
        state = strategy.record(state, observation)
        self.assertTrue(strategy._pending_groups)
        self.assertEqual(set(strategy._pending_groups[-1]), set(half_a.overrides))

    def test_unstable_verdict_raises_recovery_error(self):
        assignments = {"d0": _assignment("d0")}
        strategy = rec.BoundedDeltaDebugStrategy()
        state = self._state(assignments, ["d0"])
        proposal = rec.AssignmentProposal(label="x", overrides={"d0": "native"})
        observation = rec.Observation(proposal=proposal, verdict="unstable", report=None)
        with self.assertRaises(rec.RecoveryError):
            strategy.record(state, observation)

    def test_confirmed_implicated_singleton_recorded_on_repeated_failure(self):
        assignments = {"d0": _assignment("d0", alternatives=("alt-a",))}
        strategy = rec.BoundedDeltaDebugStrategy()
        strategy._initialized = True
        strategy._pending_groups = [("d0",)]
        state = self._state(assignments, ["d0"])
        proposal = rec.AssignmentProposal(label="x", overrides={"d0": "native"})
        observation = rec.Observation(proposal=proposal, verdict="hard_fail", report=None)
        strategy.record(state, observation)
        self.assertIn("d0", strategy._confirmed_implicated)


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
        strategy = rec.BoundedDeltaDebugStrategy()
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

        def fake_validate(overrides, *, full_corpus):
            # The final native-fallback assignment must be accepted.
            verdict = "pass" if overrides == {"d0": "native"} else "hard_fail"
            return rec.Observation(
                proposal=rec.AssignmentProposal(label="final", overrides=overrides),
                verdict=verdict, report=None, full_corpus_validated=True,
            )

        executor.evaluate.side_effect = fake_evaluate
        executor.validate_full_corpus.side_effect = fake_validate
        executor.build_candidate_cache.return_value = Path("fake.cache")

        strategy = rec.BoundedDeltaDebugStrategy()
        report = self._initial_report(["v1"])
        result = rec.run_recovery(
            executor=executor, strategy=strategy, initial_assignments=assignments,
            initial_report=report, full_corpus=[_vector("v1")], dispatch_hits=frozenset(["d0"]),
            max_evaluations=6,
        )
        self.assertTrue(result.published)
        self.assertEqual(result.final_overrides, {"d0": "native"})

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
        strategy = rec.BoundedDeltaDebugStrategy()
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
        strategy = rec.BoundedDeltaDebugStrategy()
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


if __name__ == "__main__":
    unittest.main()
