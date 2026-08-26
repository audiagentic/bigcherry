"""HI68 source contract tests: canary retry is a stability probe, never a
ranking-data replacement (RV49/F2).

The behavioral guarantee has three parts, each pinned here against the
tuner sources:

1. The transition between measurement blocks is the pure header
   hip-autotune-canary.h -- GPU-free by construction (no hip/ggml include),
   which is what makes it host-unit-testable at all.
2. A failed initial canary runs at most ONE pair-only stability probe whose
   statistics are DISCARDED: the old code overwrote pair[i]->median_us and
   friends with exactly this self-selected fresh draw; that write pattern
   must not exist anywhere in the tuner.
3. Only if the probe passes is ONE complete finalist block re-measured via
   the same extracted unit the normal final stage uses (measure_finalist_block),
   and that fresh block gets the identical post-block scrutiny (E4 noisy +
   native-baseline checks) as the original one. Its canary is evaluated once,
   terminal: no retry branch exists in the FRESH stage of the state machine.
"""

import os
import re
import unittest
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TUNER = os.path.join(
    REPO_ROOT, "src", "ggml", "src", "ggml-cuda", "hip-autotune-tuner.cu"
)
TUNER_CUH = os.path.join(
    REPO_ROOT, "src", "ggml", "src", "ggml-cuda", "hip-autotune-tuner.cuh"
)
CANARY_H = os.path.join(
    REPO_ROOT, "src", "ggml", "src", "ggml-cuda", "hip-autotune-canary.h"
)
HOST_TEST_CPP = os.path.join(
    REPO_ROOT, "tools", "tests", "fixtures", "hardware", "canary_decision_host_test.cpp"
)
DRIVER_PY = os.path.join(REPO_ROOT, "tools", "tests", "hardware", "test_hi68_canary_decision.py")


def _read(path):
    return Path(path).read_text(encoding="utf-8")


def _group(pattern: str, text: str, name: str) -> str:
    """re.search group(1), failing the test with a named message if absent."""
    match = re.search(pattern, text, re.S)
    assert match is not None, f"{name}: pattern not found in source"
    return match.group(1)


class TestHi68CanaryContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tuner = _read(TUNER)
        cls.cuh = _read(TUNER_CUH)
        cls.canary_h = _read(CANARY_H)

    # -- 1. the transition is a GPU-free pure header -----------------------
    def test_canary_header_exists_and_is_gpu_free(self):
        for include in ("hip/", "ggml/", "hip-autotune-tuner", "hip-autotune-dispatch"):
            self.assertNotIn(
                include,
                self.canary_h,
                f"hip-autotune-canary.h must stay GPU-free (found {include!r})",
            )

    def test_canary_header_states_the_three_stages(self):
        for stage in (
            "GGML_HIP_CANARY_STAGE_INITIAL",
            "GGML_HIP_CANARY_STAGE_PROBE",
            "GGML_HIP_CANARY_STAGE_FRESH",
        ):
            self.assertIn(stage, self.canary_h)

    def test_fresh_stage_has_no_retry_branch(self):
        # The FRESH case may only end in RANK or STOP_UNRESOLVED: requesting
        # another probe or fresh block would re-open the self-selected
        # baseline (F2).
        body = _group(
            r"case GGML_HIP_CANARY_STAGE_FRESH:(.*?)break;",
            self.canary_h,
            "FRESH stage branch",
        )
        self.assertNotIn("RUN_PROBE", body)
        self.assertNotIn("RUN_FRESH", body)
        self.assertIn("STOP_UNRESOLVED", body)
        self.assertIn("RANK", body)

    def test_retry_budget_is_initial_stage_only(self):
        self.assertIn(
            "retries_allowed",
            _group(
                r"case GGML_HIP_CANARY_STAGE_INITIAL:(.*?)break;",
                self.canary_h,
                "INITIAL stage branch",
            ),
        )
        for stage in ("PROBE", "FRESH"):
            body = _group(
                rf"case GGML_HIP_CANARY_STAGE_{stage}:(.*?)break;",
                self.canary_h,
                f"{stage} stage branch",
            )
            self.assertNotIn(
                "retries_allowed",
                body,
                f"{stage} stage must not consult the retry budget",
            )

    # -- 2. the probe is a probe: no statistics write-back -----------------
    def test_probe_does_not_overwrite_measurement_statistics(self):
        # The F2 defect, verbatim: the pair re-measurement writing back into
        # the ranked Measurements' median/mad/p95/host/samples.
        self.assertNotIn("pair[i]->median_us", self.tuner)
        self.assertNotIn("pair[0]->median_us", self.tuner)
        for field in ("mad_us", "p95_us", "host_median_us"):
            self.assertNotRegex(self.tuner, rf"pair\[\d\]->{field}\s*=")

    def test_old_attempt_retry_loop_is_gone(self):
        self.assertNotIn("for (int attempt = 0; twin != nullptr", self.tuner)
        self.assertNotIn("attempt >= config.noise_canary_retries", self.tuner)

    # -- 3. the fresh block is a complete, equally-scrutinized block -------
    def test_final_stage_is_an_extracted_block_unit(self):
        self.assertIn("auto measure_finalist_block = [&]", self.tuner)
        # Two call sites: the normal final stage and the canary fresh path.
        self.assertGreaterEqual(self.tuner.count("measure_finalist_block();"), 2)

    def test_fresh_path_remeasures_via_the_same_unit(self):
        body = _group(
            r"GGML_HIP_CANARY_RUN_FRESH\)(.*?)\n            }",
            self.tuner,
            "RUN_FRESH dispatch branch",
        )
        self.assertIn("measure_finalist_block();", body)

    def test_post_block_rejections_apply_to_every_ranked_block(self):
        # E4 noisy + native-baseline checks, extracted so the fresh block
        # gets identical scrutiny to the original.
        self.assertIn("auto post_block_reject_reason = [&]", self.tuner)
        self.assertGreaterEqual(self.tuner.count("post_block_reject_reason()"), 2)

    def test_fresh_canary_is_evaluated_once_with_zero_budget(self):
        # Evaluated exactly once, with a zero budget, and judged ONLY on
        # medians from the fresh block itself: a finalist that failed to
        # launch there must not contribute its stale original-block median.
        self.assertIn(
            "GGML_HIP_CANARY_STAGE_FRESH,\n"
            "                        native_m->measured ? native_m->median_us : -1.0,\n"
            "                        twin->measured ? twin->median_us : -1.0,\n"
            "                        config.noise_canary_pct, 0",
            self.tuner,
        )

    def test_probe_uses_zero_budget_transition(self):
        self.assertIn("GGML_HIP_CANARY_STAGE_PROBE,", self.tuner)

    def test_fresh_block_flag_is_recorded_and_serialized(self):
        self.assertIn("bool canary_fresh_block = false;", self.tuner)
        # Serialized into the result JSON: the format string AND the
        # argument must both carry it (a format slot without an argument is a
        # classic row-corruption bug in this file.
        # The format string carries the field as a C-escaped JSON slot:
        self.assertIn('\\"canary_fresh_block\\":%s,', self.tuner)
        self.assertIn('r.canary_fresh_block ? "true" : "false"', self.tuner)
        # The invariant is documented at the field so future readers of
        # old/new rows know which measurement window covers the ranked
        # medians: a retried pass always comes from a fresh block, and a
        # fresh block may also end unresolved (terminal, native retained).
        self.assertIn(
            "canary_state == retried_pass\n    // implies canary_fresh_block",
            self.tuner,
        )

    def test_fresh_block_failure_restores_original_evidence(self):
        # GPT HI68 closure requirement (fault-path provenance): probe pass ->
        # fresh block BEGINS -> mechanical failure after mutation. The row
        # emitted on that path must still carry ORIGINAL-block evidence:
        # canary_fresh_block stays false, state stays UNRESOLVED, retries is 1.
        # measure_finalist_block() is not transactional -- it clears the
        # sample arrays and recomputes statistics as soon as any fresh round
        # completes -- so the guarantee can only come from snapshot-before +
        # restore-on-every-mechanical-failure-gate. Pinning both gates (the
        # retime-unresolved path is independently reachable via a fresh round
        # that reports unresolved clock drift).
        probe = self.tuner.index("else {  // GGML_HIP_CANARY_RUN_PROBE")
        branch_end = self.tuner.index("stability probe still divergent", probe)
        branch = self.tuner[probe:branch_end]

        snap_idx = branch.index("std::vector<FreshEvidenceSnapshot> fresh_snapshot")
        block_idx = branch.index("measure_finalist_block();")
        self.assertLess(
            snap_idx, block_idx, "snapshot must precede any fresh-block mutation"
        )

        # Gate 1: measurement failure / poison -> restore before serialize.
        gate1 = branch.index("if (result.measurement_failure ||", block_idx)
        r1 = branch.index("restore_fresh_evidence();", gate1)
        s1 = branch.index("record_result(dispatch_digest, result);", gate1)
        self.assertLess(
            r1,
            s1,
            "poison/failed fresh path must restore original evidence before the row is emitted",
        )

        # Gate 2: retime unresolved -> restore before serialize.
        gate2 = branch.index('if (result.retime_status == "unresolved")', gate1)
        r2 = branch.index("restore_fresh_evidence();", gate2)
        s2 = branch.index("record_result(dispatch_digest, result);", gate2)
        self.assertLess(
            r2,
            s2,
            "retime-unresolved fresh path must restore original evidence before the row is emitted",
        )

        # Commit point: the fresh flag is set only AFTER both failure gates,
        # so canary_fresh_block=false unambiguously means original-block
        # evidence on every serialized row.
        flag_idx = branch.index("result.canary_fresh_block = true;", gate2)
        self.assertGreater(flag_idx, r1)
        self.assertGreater(flag_idx, r2)

        # The probe already consumed the retry budget before any of this:
        # a fault-path row therefore serializes unresolved/1/false, which is
        # exactly the legal-matrix triple the promotion validator assigns the
        # meaning "original block retained diagnostically".
        self.assertLess(branch.index("++result.canary_retries;"), snap_idx)

    def test_pessimistic_provenance_ordering(self):
        # GPT HI68 follow-up (evidence finding 2): the RUN_PROBE branch must
        # go pessimistic BEFORE any GPU work (state UNRESOLVED up front), and
        # canary_fresh_block must be set AFTER measure_finalist_block() but
        # BEFORE the E4 post-block scrutiny, so a row rejected by that
        # scrutiny still says its evidence is the fresh block's.
        probe = self.tuner.index("else {  // GGML_HIP_CANARY_RUN_PROBE")
        branch_end = self.tuner.index("stability probe still divergent", probe)
        branch = self.tuner[probe:branch_end]
        state_idx = branch.index("result.canary_state = GGML_HIP_CANARY_UNRESOLVED")
        fresh_idx = branch.index("result.canary_fresh_block = true;")
        e4_idx = branch.index("post_block_reject_reason();")
        block_idx = branch.index("measure_finalist_block();")
        self.assertLess(
            state_idx, block_idx, "state must be pessimistic before the probe runs"
        )
        self.assertLess(block_idx, fresh_idx)
        self.assertLess(
            fresh_idx,
            e4_idx,
            "fresh flag must precede E4 scrutiny so a rejected "
            "row still carries authoritative window provenance",
        )

    # -- 4. the fresh attempt is transactional (fault-path provenance) -----
    def _run_probe_branch(self):
        probe = self.tuner.index("else {  // GGML_HIP_CANARY_RUN_PROBE")
        branch_end = self.tuner.index("stability probe still divergent", probe)
        return self.tuner[probe:branch_end]

    def test_fresh_block_attempt_is_transactional(self):
        # GPT HI68 closure requirement (fault-path provenance): when a fresh
        # attempt fails mechanically -- measurement_failure/poison or
        # retime-unresolved -- the serialized row must still carry ORIGINAL-
        # block evidence, because canary_fresh_block=false means exactly that
        # by contract (and the validator assigns unresolved/1/false the
        # meaning "probe failed; original block retained"). measure_
        # finalist_block() is not transactional on its own -- it clears the
        # sample arrays and recomputes statistics as soon as any fresh round
        # completes -- so the fresh path must (1) snapshot before the attempt,
        # (2) restore on BOTH mechanical-failure branches before the row is
        # serialized, and (3) set canary_fresh_block only past both gates.
        branch = self._run_probe_branch()

        snap_idx = branch.index("std::vector<FreshEvidenceSnapshot> fresh_snapshot")
        restore_def_idx = branch.index("const auto restore_fresh_evidence")
        block_idx = branch.index("measure_finalist_block();")
        self.assertLess(snap_idx, block_idx, "snapshot must precede the fresh attempt")
        self.assertLess(
            restore_def_idx, block_idx, "restore lambda must be defined before use"
        )

        poison_gate = branch.index("if (result.measurement_failure ||", block_idx)
        retime_gate = branch.index(
            'if (result.retime_status == "unresolved")', poison_gate
        )
        flag_idx = branch.index("result.canary_fresh_block = true;", retime_gate)

        poison_restore = branch.index("restore_fresh_evidence();", poison_gate)
        poison_record = branch.index("record_result(", poison_gate)
        self.assertLess(
            poison_restore,
            poison_record,
            "poison/failed-fresh path must restore original evidence before serializing",
        )

        retime_restore = branch.index("restore_fresh_evidence();", retime_gate)
        retime_record = branch.index("record_result(", retime_gate)
        self.assertLess(
            retime_restore,
            retime_record,
            "retime-unresolved path must restore original evidence before serializing",
        )

        self.assertGreater(flag_idx, poison_record)
        self.assertGreater(
            flag_idx,
            retime_record,
            "the fresh flag (commit point) must be set only after both "
            "mechanical-failure gates: an incomplete fresh attempt is not a "
            "committed fresh ranking block",
        )

    def test_fresh_evidence_snapshot_covers_all_mutable_fields(self):
        # The snapshot/restore pair must cover every field
        # measure_finalist_block() can mutate -- a missed field would let one
        # stale fresh value survive the rollback and re-create the provenance
        # ambiguity under a false flag.
        branch = self._run_probe_branch()
        fields = (
            "final_gpu_us",
            "final_host_us",
            "reason",
            "median_us",
            "mad_us",
            "p95_us",
            "host_median_us",
            "samples",
            "measured",
        )
        snap_struct = _group(
            r"struct FreshEvidenceSnapshot \{(.*?)\};", branch, "snapshot struct"
        )
        for field in fields:
            self.assertIn(field, snap_struct)

        capture_window = branch[
            branch.index(
                "std::vector<FreshEvidenceSnapshot> fresh_snapshot"
            ) : branch.index("const auto restore_fresh_evidence")
        ]
        restore_window = branch[
            branch.index("const auto restore_fresh_evidence") : branch.index(
                "measure_finalist_block();"
            )
        ]
        for field in fields:
            # Whitespace-tolerant: the assignments are column-aligned by the
            # formatter, and the guarantee is the field coverage, not spacing.
            self.assertRegex(
                capture_window,
                rf"s\.{field}\s*=\s*m->{field};",
                f"snapshot capture loop must read {field}",
            )
            self.assertRegex(
                restore_window,
                rf"m->{field}\s*=\s*s\.{field};",
                f"restore loop must write back {field}",
            )

    # -- host testability wiring ------------------------------------------
    def test_host_test_and_driver_exist(self):
        cpp = _read(HOST_TEST_CPP)
        driver = _read(DRIVER_PY)
        self.assertIn('#include "hip-autotune-canary.h"', cpp)
        self.assertIn("CANARY_DECISION_HOST_TEST_OK", cpp)
        self.assertIn("CANARY_DECISION_HOST_TEST_OK", driver)
        self.assertIn("hip-autotune-canary.h", driver)

    def test_config_comment_states_probe_semantics(self):
        # noise_canary_retries documents its HI68 meaning (probe allowance),
        # so a reader of the config cannot mistake it for a retry-until-quiet
        # budget again.
        idx = self.cuh.index("noise_canary_retries")
        window = self.cuh[max(0, idx - 700) : idx]
        self.assertIn("HI68", window)
        self.assertIn("stability", window.lower())


if __name__ == "__main__":
    unittest.main()
