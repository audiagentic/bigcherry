"""Run-stage advisories attached to runtime-matrix results (RHA03).

This is deliberately a reporting layer. It classifies what a run result does
and does not prove; it never changes the matrix state, child verdict, or
performance-admission decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RunAdvisory:
    tag: str
    headline: str
    body: tuple[str, ...]
    severity: str = "finding"

    def document(self) -> dict[str, Any]:
        return {
            "id": self.tag,
            "headline": self.headline,
            "body": list(self.body),
            "severity": self.severity,
        }


CHECK_IDS = (
    "RUN_FAILURE",
    "RUN_INCOMPLETE_MATRIX",
    "RUN_NO_CHILD_RESULT",
    "RUN_NOT_ADMITTED",
    "RUN_MISSING_METRICS",
    "RUN_DIAGNOSTIC_EVIDENCE",
)

AB_CHECK_IDS = (
    "AB_RUN_FAILURE",
    "AB_NO_EVIDENCE",
    "AB_NOT_ADMITTED",
    "AB_MISSING_COMPARISON",
    "AB_DIAGNOSTIC_EVIDENCE",
)


@dataclass(frozen=True)
class RunEvaluation:
    evaluated: tuple[str, ...]
    errors: tuple[str, ...]
    findings: tuple[RunAdvisory, ...]

    @property
    def complete(self) -> bool:
        return not self.errors

    def document(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "complete": self.complete,
            "evaluated": list(self.evaluated),
            "errors": list(self.errors),
            "findings": [finding.document() for finding in self.findings],
        }


def evaluate_runtime_result(result: Mapping[str, Any]) -> RunEvaluation:
    """Evaluate a matrix summary without changing it or raising on bad input."""
    errors: list[str] = []
    findings: list[RunAdvisory] = []
    if not isinstance(result, Mapping):
        return RunEvaluation((), ("result must be an object",), ())
    state = result.get("state")
    completed = result.get("completed")
    total = result.get("total")
    rows = result.get("results")
    if not isinstance(state, str):
        errors.append("state missing or malformed")
    if not isinstance(completed, int) or not isinstance(total, int):
        errors.append("completed/total missing or malformed")
    if not isinstance(rows, list):
        errors.append("results missing or malformed")
        rows = []

    if state == "failed":
        findings.append(RunAdvisory(
            "RUN_FAILURE", "Runtime matrix failed before full completion.",
            ("Do not interpret partial child metrics as a matrix-level result.",), "stop"
        ))
    if isinstance(completed, int) and isinstance(total, int) and completed != total:
        findings.append(RunAdvisory(
            "RUN_INCOMPLETE_MATRIX", f"Only {completed} of {total} runtime cells completed.",
            ("A partial matrix cannot support a complete model/topology conclusion.",), "stop"
        ))

    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("child_result"), Mapping):
            findings.append(RunAdvisory(
                "RUN_NO_CHILD_RESULT", "A matrix cell has no child result artifact.",
                ("Inspect the worker boundary before trusting this cell.",), "stop"
            ))
            continue
        child = row["child_result"]
        if child.get("performance_admitted") is False:
            findings.append(RunAdvisory(
                "RUN_NOT_ADMITTED", "A child result is explicitly not performance-admitted.",
                ("This run may prove wiring or liveness, but it is not a parity claim.",),
            ))
        if not isinstance(child.get("metrics"), Mapping) or not child.get("metrics"):
            findings.append(RunAdvisory(
                "RUN_MISSING_METRICS", "A completed cell has no throughput metrics.",
                ("Treat it as liveness/evidence plumbing only until metrics are present.",),
            ))
        evidence_role = child.get("evidence_role")
        execution_evidence = child.get("execution_evidence")
        if evidence_role in {"diagnostic", "observe"} or execution_evidence == "observe":
            findings.append(RunAdvisory(
                "RUN_DIAGNOSTIC_EVIDENCE", "A child result is diagnostic or observation-only evidence.",
                ("Do not promote its timings to a production performance conclusion.",),
            ))

    return RunEvaluation(tuple(CHECK_IDS), tuple(errors), tuple(findings))


def evaluate_ab_result(result: Mapping[str, Any]) -> RunEvaluation:
    """Evaluate a maintained paired A/B result without changing its policy."""
    errors: list[str] = []
    findings: list[RunAdvisory] = []
    if not isinstance(result, Mapping):
        return RunEvaluation((), ("result must be an object",), ())
    rows = result.get("runs")
    if not isinstance(rows, list):
        errors.append("runs missing or malformed")
        rows = []
    if not rows:
        findings.append(RunAdvisory(
            "AB_NO_EVIDENCE", "The A/B boundary produced no arm observations.",
            ("Do not interpret the configuration as a comparison.",), "stop"
        ))
    if any(isinstance(row, Mapping) and row.get("returncode") not in (None, 0) for row in rows):
        findings.append(RunAdvisory(
            "AB_RUN_FAILURE", "At least one A/B arm failed.",
            ("Partial arm output cannot support a paired conclusion.",), "stop"
        ))
    if result.get("performance_admitted") is False:
        findings.append(RunAdvisory(
            "AB_NOT_ADMITTED", "This A/B result is explicitly not performance-admitted.",
            ("Treat it as exploratory or wiring evidence until all admission gates pass."),
        ))
    comparisons = result.get("exploratory_comparisons") or result.get("comparisons")
    if not isinstance(comparisons, Mapping) or not comparisons:
        findings.append(RunAdvisory(
            "AB_MISSING_COMPARISON", "No paired comparison summary is present.",
            ("Check work equivalence and metric extraction before interpreting arms."),
        ))
    if result.get("evidence_role") == "diagnostic" or result.get("execution_evidence") == "observe":
        findings.append(RunAdvisory(
            "AB_DIAGNOSTIC_EVIDENCE", "The A/B result is diagnostic or observation-only evidence.",
            ("Do not transfer its timings to a production performance claim."),
        ))
    return RunEvaluation(AB_CHECK_IDS, tuple(errors), tuple(findings))


def render(evaluation: RunEvaluation) -> str:
    if not evaluation.findings and evaluation.complete:
        return ""
    lines = ["", "-- run advisories " + "-" * 57]
    for finding in evaluation.findings:
        prefix = "STOP" if finding.severity == "stop" else finding.tag
        lines.append(f"[{prefix}] {finding.headline}")
        lines.extend(f"    {line}" for line in finding.body)
    if evaluation.errors:
        lines.append("[UNKNOWN] " + "; ".join(evaluation.errors))
    return "\n".join(lines) + "\n"


__all__ = ["CHECK_IDS", "RunAdvisory", "RunEvaluation", "evaluate_runtime_result", "render"]
