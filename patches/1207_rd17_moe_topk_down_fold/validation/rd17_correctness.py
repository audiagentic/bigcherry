"""RD17 correctness producer -- proves patches/1207_rd17_moe_topk_down_fold.py's
MoE topk-weights-into-down-projection fusion does not change model output
quality, via real perplexity (PPL) equality between subject (fusion active)
and control (fusion detection reverted) on a real MoE model.

Design mirrors patches/1204_rd08_q6k_mmvq_vdr2/validation/rd08_correctness.py's
established shape (materialize_source_variant() with an explicit checked
control reversion, both built from the same immutable base + patch stack),
but the correctness metric is llama.cpp's own llama-perplexity tool's
"Final estimate: PPL = X +/- Y" output, not test-backend-ops digests --
RD17's fork claim is "bit-identical PERPLEXITY", not bit-identical raw
output bytes (this fusion changes only WHERE a per-token scale multiply
happens, not floating-point summation order the way RD08's VDR=2 does), so
ppl_equality (config/experiment-contracts.toml's RD17-MOE-TOPK-DOWN-FOLD
contract) is the right bar here, not bit_identical.

Control reversion: RD17's ggml-cuda.cu detection block is the ONLY thing
that ever sets fusion_data.x_scale_channel_dst = true; reverting just that
block (back to the plain anchor it was inserted before) means the
common.cuh struct field and mmvq.cu kernel plumbing stay compiled but are
never exercised -- the same "revert only the routing, not the full patch"
principle RD08's apply_vdr1_control uses.
"""

from __future__ import annotations

import importlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

psi = importlib.import_module("bigcherry.patch.source")

RD17_PATCH_STACK: tuple[str, ...] = ("1207_rd17_moe_topk_down_fold",)


class Rd17CorrectnessError(RuntimeError):
    """RD17 correctness evidence could not be produced, or failed to validate."""


# Exactly RD17's own ggml-cuda.cu detection-block insertion (patch.py's
# _DETECT_ANCHOR/_DETECT_BLOCK), reverted back to the bare anchor. This is
# the ONLY site that ever sets x_scale_channel_dst=true -- reverting it
# alone is sufficient to make the fusion never trigger while leaving the
# rest of the patch's struct/kernel edits compiled (matching what RD08's
# control does: revert the minimum that disables the behavior, not undo
# every hunk).
# Imported directly from the real patch.py rather than duplicated here --
# an earlier version of this file hand-copied _DETECT_BLOCK/_DETECT_ANCHOR,
# and a real-hardware-driven fix to patch.py's copy (2026-09-11, GPT root-
# cause req_6cf169798c784380) silently desynced this control-reversion
# copy, which would have made apply_no_fusion_control() either fail its
# own anchor-count check or, worse, revert the WRONG (pre-fix) text. A
# single source of truth eliminates that whole bug class.
def _load_patch_module() -> object:
    import importlib.util

    patch_path = Path(__file__).resolve().parent.parent / "patch.py"
    spec = importlib.util.spec_from_file_location("_bigcherry_rd17_patch_module", patch_path)
    if spec is None or spec.loader is None:
        raise Rd17CorrectnessError(f"cannot load RD17 patch module at {patch_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_patch_module = _load_patch_module()
_DETECT_ANCHOR = _patch_module._DETECT_ANCHOR
_DETECT_BLOCK = _patch_module._DETECT_BLOCK
_DETECT_NEW = _DETECT_BLOCK + _DETECT_ANCHOR

_CONTROL_EDITS: tuple[tuple[Path, str, str], ...] = (
    (Path("ggml/src/ggml-cuda/ggml-cuda.cu"), _DETECT_NEW, _DETECT_ANCHOR),
)


def apply_no_fusion_control(source_dir: Path) -> None:
    """Exactly one checked semantic reversion: remove RD17's detection
    block, leaving the anchor bare. Fails closed if the anchor does not
    match exactly once before or after -- a silently broader or narrower
    control would invalidate the entire comparison without any visible
    symptom (same discipline as RD08's apply_vdr1_control)."""
    for rel_path, old, new in _CONTROL_EDITS:
        source_root = source_dir.resolve()
        target = (source_dir / rel_path).resolve()
        try:
            target.relative_to(source_root)
        except ValueError as exc:
            raise Rd17CorrectnessError(
                f"RD17 control path escapes source root: {rel_path}"
            ) from exc
        text = target.read_text(encoding="utf-8")
        before = text.count(old)
        if before != 1:
            raise Rd17CorrectnessError(
                f"RD17 control: expected exactly 1 occurrence of the detection block in "
                f"{rel_path}, found {before}"
            )
        text = text.replace(old, new, 1)
        after = text.count(new)
        if after != 1:
            raise Rd17CorrectnessError(
                f"RD17 control: expected exactly 1 occurrence of the bare anchor in "
                f"{rel_path} after the reversion, found {after}"
            )
        target.write_text(text, encoding="utf-8", newline="")


def materialize_rd17_variants(
    *, base_repo: Path, worktree_root: Path, base_revision: str,
) -> tuple[Path, Path]:
    """Return (subject_src, control_src): the fusion-active subject and
    no-fusion control isolated worktrees, both carrying RD17_PATCH_STACK on
    top of the source's explicit named composition, differing only by
    apply_no_fusion_control's single reversion on the control."""
    resolved_revision, composition = psi.resolve_source_composition(
        "bigcherry", extra_patches=RD17_PATCH_STACK, base_ref=base_revision, base_repo=base_repo,
    )
    subject_src = psi.materialize_source_variant(
        base_repo=base_repo, worktree_root=worktree_root,
        resolved_revision=resolved_revision, composition=composition,
        overlay_root=psi.REPO_ROOT / "src", requested_revision=base_revision,
        variant_name="rd17-fusion-subject", variant_digest="none",
    )
    control_src = psi.materialize_source_variant(
        base_repo=base_repo, worktree_root=worktree_root,
        resolved_revision=resolved_revision, composition=composition,
        overlay_root=psi.REPO_ROOT / "src", requested_revision=base_revision,
        variant_name="rd17-no-fusion-control", variant_digest="no-fusion-v1",
        apply_variant=apply_no_fusion_control,
    )
    return subject_src, control_src


_PPL_LINE = re.compile(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)")


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
        raise Rd17CorrectnessError(
            f"llama-perplexity exited {completed.returncode} for {binary}: "
            f"{completed.stderr[-2000:]}"
        )
    combined = f"{completed.stdout}\n{completed.stderr}"
    match = _PPL_LINE.search(combined)
    if match is None:
        raise Rd17CorrectnessError(
            f"llama-perplexity ({binary}) produced no parseable "
            "'Final estimate: PPL = ...' line -- cannot treat this as evidence"
        )
    return PerplexityRun(
        ppl=float(match.group(1)), uncertainty=float(match.group(2)),
        raw_stdout=completed.stdout, raw_stderr=completed.stderr,
    )


@dataclass(frozen=True)
class Rd17PplComparison:
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


def require_rd17_ppl_equality(
    *, subject_binary: Path, control_binary: Path, model: Path, corpus: Path,
    ctx_size: int = 2048, runner=subprocess.run, extra_args: tuple[str, ...] = (),
) -> Rd17PplComparison:
    """Run llama-perplexity against BOTH binaries on the identical real
    corpus and compare. Fails closed (raises Rd17CorrectnessError) if the
    two PPL estimates disagree beyond max_sigma combined uncertainty --
    never a silent partial pass."""
    subject_run = run_perplexity(
        subject_binary, model=model, corpus=corpus, ctx_size=ctx_size,
        runner=runner, extra_args=extra_args,
    )
    control_run = run_perplexity(
        control_binary, model=model, corpus=corpus, ctx_size=ctx_size,
        runner=runner, extra_args=extra_args,
    )
    comparison = Rd17PplComparison(subject=subject_run, control=control_run)
    if not comparison.ok:
        raise Rd17CorrectnessError(
            f"RD17 ppl_equality failed: subject PPL={subject_run.ppl:.4f}+/-{subject_run.uncertainty:.5f}, "
            f"control PPL={control_run.ppl:.4f}+/-{control_run.uncertainty:.5f}, "
            f"delta={comparison.delta:.5f}, sigma={comparison.sigma:.2f} "
            f"(max allowed {comparison.max_sigma})"
        )
    return comparison


def comparison_to_dict(comparison: Rd17PplComparison) -> dict[str, Any]:
    return {
        "ok": comparison.ok,
        "subject": {"ppl": comparison.subject.ppl, "uncertainty": comparison.subject.uncertainty},
        "control": {"ppl": comparison.control.ppl, "uncertainty": comparison.control.uncertainty},
        "delta": comparison.delta,
        "combined_uncertainty": comparison.combined_uncertainty,
        "sigma": comparison.sigma,
        "max_sigma": comparison.max_sigma,
    }
