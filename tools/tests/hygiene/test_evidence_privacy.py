"""No authentication values or host-specific build paths in new evidence."""
from __future__ import annotations

from pathlib import Path

import pytest

from bigcherry import forbidden_terms as ft

ROOT = Path(__file__).resolve().parents[3]

BENCHMARK_CAPTURES = (
    "docs/evidence/2026-09-10-rha10-admission-gate/raw/bench.stdout",
    "docs/evidence/2026-09-10-rha10-admission-gate/raw/diagnostic-bench.stdout",
    "docs/evidence/2026-09-10-rha10-admission-gate/raw/production-bench.stdout",
    "docs/evidence/2026-09-10-rha11-correctness/raw/final-complete-diag/bench.stdout",
    "docs/evidence/2026-09-10-rha11-correctness/raw/final-complete-prod/bench.stdout",
    "docs/evidence/2026-09-10-rha11-correctness/raw/final-diag/bench.stdout",
    "docs/evidence/2026-09-10-rha11-correctness/raw/final-prod/bench.stdout",
)


def labels(text: str) -> set[str]:
    return set(ft.scan_line(text, [], []))


@pytest.mark.parametrize("rel", BENCHMARK_CAPTURES)
def test_committed_benchmark_logs_have_redacted_auth(rel: str) -> None:
    lines = (ROOT / rel).read_text(encoding="utf-8").splitlines()
    auth = [line for line in lines if '"API_KEY"' in line]
    assert len(auth) == 1, rel
    assert '"API_KEY": "[REDACTED]"' in auth[0], rel
    for line in lines:
        assert not labels(line).intersection({"credential token", "credential assignment", "private-key header"}), rel


def test_credential_shapes_are_rejected_without_using_real_secrets() -> None:
    synthetic = "sk-" + "A" * 42
    assert {"credential token", "credential assignment"} <= labels(
        'Env vars: {"API_KEY": "' + synthetic + '"}')
    assert "credential assignment" in labels('ACCESS_TOKEN="' + "b" * 44 + '"')
    assert "credential token" in labels("github_pat_" + "x" * 45)
    assert "private-key header" in labels("-" * 5 + "BEGIN PRIVATE KEY" + "-" * 5)


def test_redacted_and_example_values_are_allowed() -> None:
    assert not labels('Env vars: {"API_KEY": "[REDACTED]"}')
    assert not labels('API_KEY="example-value"')


def test_generated_local_paths_are_rejected_but_workspace_variables_are_not() -> None:
    example = "/" + "work" + "/bigcherry/runs/example-run/RESULT.md"
    assert "local build or run path" in labels(example)
    assert "local build or run path" in labels(example.replace("/runs/", "/builds/"))
    assert "local build or run path" not in labels("${BC_WORK_ROOT}/runs/example-run/RESULT.md")
