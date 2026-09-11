"""Shared real-perplexity (PPL) comparison primitive -- extracted 2026-09-11
from patches/1207_rd17_moe_topk_down_fold/validation/rd17_correctness.py so
a second patch (RD13) needing the same real-model-forward-pass correctness
proof does not duplicate the parsing/comparison logic. Runs llama.cpp's own
llama-perplexity tool for real and parses its "Final estimate: PPL = X +/-
Y" summary line -- never a synthetic test-backend-ops shape, which cannot
exercise a graph-fusion pattern (RESHAPE-mediated add, MoE topk-scale fold)
that only a real model's real forward pass produces.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PPL_LINE = re.compile(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)")


class PerplexityError(RuntimeError):
    """A real llama-perplexity run could not be produced, or two runs'
    PPL estimates disagreed beyond tolerance."""


@dataclass(frozen=True)
class PerplexityRun:
    ppl: float
    uncertainty: float
    raw_stdout: str
    raw_stderr: str


def run_perplexity(
    binary: Path, *, model: Path, corpus: Path, ctx_size: int = 2048,
    runner=subprocess.run, extra_args: tuple[str, ...] = (),
) -> PerplexityRun:
    """Run llama-perplexity for real and parse its own "Final estimate:
    PPL = X +/- Y" summary line. Fails closed if that line is absent --
    a run that exits 0 but produces no parseable PPL line must never be
    silently treated as passing evidence."""
    argv = [
        str(binary), "-m", str(model), "-f", str(corpus),
        "-c", str(ctx_size), "-ngl", "99", *extra_args,
    ]
    completed = runner(argv, capture_output=True, text=True)
    if completed.returncode != 0:
        raise PerplexityError(
            f"llama-perplexity exited {completed.returncode} for {binary}: "
            f"{completed.stderr[-2000:]}"
        )
    combined = f"{completed.stdout}\n{completed.stderr}"
    match = _PPL_LINE.search(combined)
    if match is None:
        raise PerplexityError(
            f"llama-perplexity ({binary}) produced no parseable "
            "'Final estimate: PPL = ...' line -- cannot treat this as evidence"
        )
    return PerplexityRun(
        ppl=float(match.group(1)), uncertainty=float(match.group(2)),
        raw_stdout=completed.stdout, raw_stderr=completed.stderr,
    )


@dataclass(frozen=True)
class PerplexityComparison:
    subject: PerplexityRun
    control: PerplexityRun
    # ppl_equality (not bit_identical): the two PPL estimates must agree
    # within their COMBINED uncertainty -- the real statistical bar a
    # perplexity measurement supports, never a bare float `==`.
    max_sigma: float = 3.0

    @property
    def delta(self) -> float:
        return abs(self.subject.ppl - self.control.ppl)

    @property
    def combined_uncertainty(self) -> float:
        return (self.subject.uncertainty ** 2 + self.control.uncertainty ** 2) ** 0.5

    @property
    def sigma(self) -> float:
        if self.combined_uncertainty == 0:
            return float("inf") if self.delta > 0 else 0.0
        return self.delta / self.combined_uncertainty

    @property
    def ok(self) -> bool:
        return self.sigma <= self.max_sigma


def require_ppl_equality(
    *, subject_binary: Path, control_binary: Path, model: Path, corpus: Path,
    ctx_size: int = 2048, runner=subprocess.run, extra_args: tuple[str, ...] = (),
    max_sigma: float = 3.0,
) -> PerplexityComparison:
    """Run llama-perplexity against BOTH binaries on the identical real
    corpus and compare. Fails closed (raises PerplexityError) if the two
    PPL estimates disagree beyond max_sigma combined uncertainty -- never
    a silent partial pass."""
    subject_run = run_perplexity(
        subject_binary, model=model, corpus=corpus, ctx_size=ctx_size,
        runner=runner, extra_args=extra_args,
    )
    control_run = run_perplexity(
        control_binary, model=model, corpus=corpus, ctx_size=ctx_size,
        runner=runner, extra_args=extra_args,
    )
    comparison = PerplexityComparison(subject=subject_run, control=control_run, max_sigma=max_sigma)
    if not comparison.ok:
        raise PerplexityError(
            f"ppl_equality failed: subject PPL={subject_run.ppl:.4f}+/-{subject_run.uncertainty:.5f}, "
            f"control PPL={control_run.ppl:.4f}+/-{control_run.uncertainty:.5f}, "
            f"delta={comparison.delta:.5f}, sigma={comparison.sigma:.2f} "
            f"(max allowed {comparison.max_sigma})"
        )
    return comparison


def comparison_to_dict(comparison: PerplexityComparison) -> dict[str, Any]:
    return {
        "ok": comparison.ok,
        "subject": {"ppl": comparison.subject.ppl, "uncertainty": comparison.subject.uncertainty},
        "control": {"ppl": comparison.control.ppl, "uncertainty": comparison.control.uncertainty},
        "delta": comparison.delta,
        "combined_uncertainty": comparison.combined_uncertainty,
        "sigma": comparison.sigma,
        "max_sigma": comparison.max_sigma,
    }
