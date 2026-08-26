"""HI83: RD08 correctness producer -- proves patches/1204_rd08_q6k_mmvq_
vdr2.py's VDR=2 Q6_K MMVQ decode kernel is bit-identical to the VDR=1 kernel
it replaces.

Design: GPT (gpt-auto-agent), req_51d04d6dd98547da (session
ses_aeddb42e0c9d4b59, 2026-08-22), reviewed against the real pushed state of
patch_source_isolation.py, correctness_evidence.py, and the RD08 patch.

Two isolated worktrees, both built from patch_source_isolation.
materialize_source_variant() against the SAME immutable base_revision +
patch stack (RD08_PATCH_STACK below) -- never from the shared, currently
mid-flight vendor/llama.cpp working tree, and never from an already-
materialized single-patch worktree (which may carry uncommitted changes
relative to its own detached HEAD):

  - subject: the ordinary RD08 tree (VDR=2 active).
  - control: the SAME tree with exactly two checked semantic reversions
    (apply_vdr1_control) putting the Q6_K MMVQ path back on the VDR=1
    kernel RD08 replaces -- everything else (activation marker, test cases,
    op-timing guard) identical between the two.

Each of the 5 RD08 decode shapes x 3 deterministic seeds is run as its own
test-backend-ops process against BOTH binaries (dispatch_mode=native, no
forced candidate -- this is a source-level A/B, not a dispatch-tuner
comparison), reusing correctness_evidence.run_test_backend_ops() for the
env/argv construction and patches/1222+1223 for the deterministic-seed and
digest-emitting machinery. Bit-identity is proven by exact backend1_digest
equality between the two binaries' runs for every (shape, seed) pair -- NOT
by NMSE agreement, which two different implementations could satisfy without
producing identical bytes. Fails closed if either binary predates the 1223
digest extension (backend1_digest missing) rather than silently degrading to
an NMSE-only check.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ce = importlib.import_module("bigcherry.tuning.correctness_evidence")
psi = importlib.import_module("bigcherry.patch.source")

RD08_PATCH_STACK: tuple[str, ...] = (
    "1204_rd08_q6k_mmvq_vdr2",
    "1222_hi67_deterministic_test_backend_ops_seed",
    "1223_hi67_machine_readable_correctness_metrics",
)

# Exactly RD08's own two semantic routing changes (patches/1204_rd08_q6k_
# mmvq_vdr2.py's rd08-vdr-define and rd08-switch-q6k edits), reverted. The
# VDR=2 function bodies stay compiled but unreachable in the control tree --
# only routing changes, so the activation marker, test cases, and op-timing
# guard are byte-identical between subject and control.
_CONTROL_EDITS: tuple[tuple[Path, str, str], ...] = (
    (
        Path("ggml/src/ggml-cuda/vecdotq.cuh"),
        "#define VDR_Q6_K_Q8_1_MMVQ 2",
        "#define VDR_Q6_K_Q8_1_MMVQ 1",
    ),
    (
        Path("ggml/src/ggml-cuda/mmvq.cu"),
        "        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1_vdr2;",
        "        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;",
    ),
)


class Rd08CorrectnessError(RuntimeError):
    """RD08 correctness evidence could not be produced, or failed to validate."""


def _control_variant_digest() -> str:
    """Stable digest over the ACTUAL transform content (path/old/new
    triples), not a bare name -- so two different control transforms can
    never collide on the same materialize_source_variant() source_key, and
    this transform's own identity is independently checkable."""
    payload = [
        {"path": str(rel_path), "old": old, "new": new}
        for rel_path, old, new in _CONTROL_EDITS
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def apply_vdr1_control(source_dir: Path) -> None:
    """Exactly two checked semantic reversions. Fails closed (raises) if
    either anchor does not match exactly once before the edit, or the
    replacement does not match exactly once after -- a silently broader or
    narrower control would invalidate the entire comparison without any
    visible symptom."""
    for rel_path, old, new in _CONTROL_EDITS:
        source_root = source_dir.resolve()
        target = (source_dir / rel_path).resolve()
        try:
            target.relative_to(source_root)
        except ValueError as exc:
            raise Rd08CorrectnessError(
                f"VDR1 control path escapes source root: {rel_path}"
            ) from exc
        text = target.read_text(encoding="utf-8")
        before = text.count(old)
        if before != 1:
            raise Rd08CorrectnessError(
                f"VDR1 control: expected exactly 1 occurrence of {old!r} in "
                f"{rel_path}, found {before}"
            )
        text = text.replace(old, new, 1)
        after = text.count(new)
        if after != 1:
            raise Rd08CorrectnessError(
                f"VDR1 control: expected exactly 1 occurrence of {new!r} in "
                f"{rel_path} after the reversion, found {after}"
            )
        target.write_text(text, encoding="utf-8", newline="")


def materialize_rd08_variants(
    *,
    base_repo: Path,
    worktree_root: Path,
    base_revision: str,
) -> tuple[Path, Path]:
    """Return (subject_src, control_src): the VDR2-subject and VDR1-control
    isolated worktrees, both carrying RD08_PATCH_STACK on top of the source's
    explicit named composition (recipes [source.bigcherry] patch-sets),
    differing only by apply_vdr1_control's two reversions on the control.

    RV80/B6: the composition is resolved explicitly (exact-composition
    validator, topological order) -- never a lifecycle-state scan -- and the
    base ref resolves to the immutable SHA that enters the v2 identity."""
    resolved_revision, composition = psi.resolve_source_composition(
        "bigcherry",
        extra_patches=RD08_PATCH_STACK,
        base_ref=base_revision,
        base_repo=base_repo,
    )
    subject_src = psi.materialize_source_variant(
        base_repo=base_repo,
        worktree_root=worktree_root,
        resolved_revision=resolved_revision,
        composition=composition,
        overlay_root=psi.REPO_ROOT / "src",
        requested_revision=base_revision,
        variant_name="rd08-vdr2-subject",
        variant_digest="none",
    )
    control_src = psi.materialize_source_variant(
        base_repo=base_repo,
        worktree_root=worktree_root,
        resolved_revision=resolved_revision,
        composition=composition,
        overlay_root=psi.REPO_ROOT / "src",
        requested_revision=base_revision,
        variant_name="rd08-vdr1-control",
        variant_digest=_control_variant_digest(),
        apply_variant=apply_vdr1_control,
    )
    return subject_src, control_src


@dataclass(frozen=True)
class Rd08Shape:
    name: str
    m: int
    n: int
    k: int


# The exact 5 decode shapes patches/1204_rd08_q6k_mmvq_vdr2.py's own
# rd08-perf-cases edit adds to tests/test-backend-ops.cpp.
RD08_SHAPES: tuple[Rd08Shape, ...] = (
    Rd08Shape("ffn", 10240, 1, 5120),
    Rd08Shape("lm_head", 248320, 1, 5120),
    Rd08Shape("qkv", 12288, 1, 5120),
    Rd08Shape("ssm_z", 6144, 1, 5120),
    Rd08Shape("kv_proj", 1024, 1, 5120),
)

RD08_SEEDS: tuple[int, ...] = (1, 2, 3)


def op_filter(shape: Rd08Shape) -> str:
    return (
        f"type_a=q6_K,type_b=f32,m={shape.m},n={shape.n},k={shape.k},"
        r"bs=\[1,1\],nr=\[1,1\]"
    )


@dataclass(frozen=True)
class ShapeSeedComparison:
    shape_name: str
    seed: int
    subject_status: str
    control_status: str
    subject_digest: Any
    control_digest: Any
    subject_metric: Any
    control_metric: Any

    @property
    def ok(self) -> bool:
        return (
            self.subject_status == "ok"
            and self.control_status == "ok"
            and self.subject_digest is not None
            and self.control_digest is not None
            and self.subject_digest.digest == self.control_digest.digest
            and self.subject_metric is not None
            and self.control_metric is not None
            and self.subject_metric.backend1_digest is not None
            and self.control_metric.backend1_digest is not None
            and self.subject_metric.backend1_digest
            == self.control_metric.backend1_digest
        )


def compare_one_shape_seed(
    *,
    subject_binary: Path,
    control_binary: Path,
    shape: Rd08Shape,
    seed: int,
    runner=subprocess.run,
) -> ShapeSeedComparison:
    """Run one RD08 decode shape at one deterministic seed against BOTH
    binaries (native dispatch, no forced candidate -- this compares source
    variants, not dispatch candidates) and reduce to one comparison row.

    Does not raise on a mismatch or a missing digest -- callers that need a
    fail-closed pass/fail should check .ok (or use
    require_rd08_correctness_evidence(), which raises with the specific
    reason). Keeping this function non-raising lets a caller collect ALL 15
    rows before deciding, rather than stopping at the first failure.
    """
    subject_run = ce.run_test_backend_ops(
        subject_binary,
        op_filter=op_filter(shape),
        seed=seed,
        dispatch_mode="native",
        forced_candidate=None,
        runner=runner,
    )
    control_run = ce.run_test_backend_ops(
        control_binary,
        op_filter=op_filter(shape),
        seed=seed,
        dispatch_mode="native",
        forced_candidate=None,
        runner=runner,
    )

    subject_status = "ok" if subject_run.returncode == 0 else "failed"
    control_status = "ok" if control_run.returncode == 0 else "failed"

    subject_digest = ce.find_digest_for_tensor(subject_run.stderr, "dst")
    control_digest = ce.find_digest_for_tensor(control_run.stderr, "dst")
    subject_metric = ce.find_metric_for_tensor(subject_run.stderr, "dst")
    control_metric = ce.find_metric_for_tensor(control_run.stderr, "dst")

    return ShapeSeedComparison(
        shape_name=shape.name,
        seed=seed,
        subject_status=subject_status,
        control_status=control_status,
        subject_digest=subject_digest,
        control_digest=control_digest,
        subject_metric=subject_metric,
        control_metric=control_metric,
    )


def require_rd08_correctness_evidence(
    *,
    subject_binary: Path,
    control_binary: Path,
    shapes: tuple[Rd08Shape, ...] = RD08_SHAPES,
    seeds: tuple[int, ...] = RD08_SEEDS,
    runner=subprocess.run,
) -> tuple[ShapeSeedComparison, ...]:
    """Run every (shape, seed) pair and fail closed with a specific reason
    on the first row that isn't .ok -- never a silent partial pass.

    Returns all rows (in shapes x seeds order) only when every single one
    passed: subject and control both executed cleanly, saw the identical
    CPU-reference input (input digest equality -- proves both processes
    generated the same random tensors), and produced bit-identical GPU
    output (backend1_digest equality -- the actual VDR1-vs-VDR2 proof)."""
    rows: list[ShapeSeedComparison] = []
    for shape in shapes:
        for seed in seeds:
            row = compare_one_shape_seed(
                subject_binary=subject_binary,
                control_binary=control_binary,
                shape=shape,
                seed=seed,
                runner=runner,
            )
            if not row.ok:
                raise Rd08CorrectnessError(
                    f"RD08 correctness evidence failed for shape={row.shape_name!r} "
                    f"seed={row.seed}: subject_status={row.subject_status} "
                    f"control_status={row.control_status} "
                    f"subject_input_digest={row.subject_digest.digest if row.subject_digest else None} "
                    f"control_input_digest={row.control_digest.digest if row.control_digest else None} "
                    f"subject_output_digest={row.subject_metric.backend1_digest if row.subject_metric else None} "
                    f"control_output_digest={row.control_metric.backend1_digest if row.control_metric else None}"
                )
            rows.append(row)
    return tuple(rows)
