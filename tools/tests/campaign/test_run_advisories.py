import unittest

from bigcherry.campaign.run_advisories import evaluate_ab_result, evaluate_build_result, evaluate_runtime_result


def child(**extra):
    value = {"performance_admitted": True, "metrics": {"tg128_tps": 1.0}}
    value.update(extra)
    return value


class RunAdvisoryTests(unittest.TestCase):
    def test_success_is_complete_and_silent(self):
        evaluation = evaluate_runtime_result({
            "state": "completed", "completed": 1, "total": 1,
            "results": [{"child_result": child()}],
        })
        self.assertTrue(evaluation.complete)
        self.assertEqual(evaluation.findings, ())

    def test_degraded_result_is_finding_but_does_not_change_state(self):
        evaluation = evaluate_runtime_result({
            "state": "completed", "completed": 1, "total": 1,
            "results": [{"child_result": child(performance_admitted=False)}],
        })
        self.assertIn("RUN_NOT_ADMITTED", [finding.tag for finding in evaluation.findings])
        self.assertTrue(evaluation.complete)

    def test_failure_is_stop_advisory(self):
        evaluation = evaluate_runtime_result({
            "state": "failed", "completed": 0, "total": 1, "results": [],
        })
        finding = next(finding for finding in evaluation.findings if finding.tag == "RUN_FAILURE")
        self.assertEqual(finding.severity, "stop")

    def test_no_evidence_is_reported(self):
        evaluation = evaluate_runtime_result({
            "state": "completed", "completed": 1, "total": 1,
            "results": [{"matrix_status": "executed", "child_result": {}}],
        })
        tags = [finding.tag for finding in evaluation.findings]
        self.assertIn("RUN_MISSING_METRICS", tags)

    def test_malformed_result_is_unknown_not_assurance(self):
        evaluation = evaluate_runtime_result({"state": "completed"})
        self.assertFalse(evaluation.complete)
        self.assertTrue(evaluation.errors)

    def test_ab_success_has_no_advisory(self):
        evaluation = evaluate_ab_result({
            "performance_admitted": True,
            "runs": [{"returncode": 0}],
            "comparisons": {"replay_vs_native": {"tg128_tps": {"effect_pct": 1.0}}},
        })
        self.assertTrue(evaluation.complete)
        self.assertEqual(evaluation.findings, ())

    def test_ab_failure_and_missing_evidence_are_distinct(self):
        evaluation = evaluate_ab_result({
            "performance_admitted": False,
            "runs": [{"returncode": 1}],
        })
        tags = [finding.tag for finding in evaluation.findings]
        self.assertIn("AB_RUN_FAILURE", tags)
        self.assertIn("AB_MISSING_COMPARISON", tags)
        self.assertIn("AB_NOT_ADMITTED", tags)

    def test_build_failure_and_missing_identity_are_distinct(self):
        class Result:
            build_plan_id = None
        evaluation = evaluate_build_result({"broken": RuntimeError("no build"), "empty": Result()})
        tags = [finding.tag for finding in evaluation.findings]
        self.assertIn("BUILD_FAILURE", tags)
        self.assertIn("BUILD_IDENTITY_MISSING", tags)
