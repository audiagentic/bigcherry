"""RD94: wire EC13's routing generator into RD30/31/32's boundary sweeps.

RD30 (AMD-MOE-001), RD31 (AMD-MOE-002), RD32 (AMD-MOE-003) are the
mean-expert-occupancy MoE tile/launch-sizing cluster -- all three are
currently NEEDS-REDESIGN (RV53, 2026-08-20: their AMD-fork PR anchors no
longer exist against the table-driven ggml_cuda_mmq_config system that
replaced them), so there is no materialized selector kernel yet to run
hostile-routing evidence against. What this module CAN do today, honestly:
generate the exact boundary matrix RD30/31/32's own code_samples specify
(ubatch x distribution mode) using EC13's real routing generator
(moe_routing_gen.py), and compute the mean/p95/max tokens-per-expert
statistics RD30's own acceptance criteria name explicitly ("record mean/p95/
max tokens-per-expert"). This is real, useful evidence infrastructure for
scoping the eventual redesign -- it is NOT proof that a selector (which does
not exist yet) survives hostile routing, and must not be represented as such.

Boundary dimensions ubatch=(128,256,512,1024,2048,4096) and the
uniform/mild-skew-zipf/concentrated/single-hot distribution set are taken
directly from RD30/RD31/RD32's own "code_samples"/"boundary" sections
(mcp__ag-planning__plan_get_item), not invented here.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import moe_routing_gen as mrg

#: RD30/31/32's own boundary sweep dimension (their code_samples sections,
#: verbatim: "ubatch 128,256,512,1024,2048,4096").
BOUNDARY_UBATCH: tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096)

#: RD30's own boundary language: "routing uniform/captured/mild skew/Zipf/
#: concentrated/single-hot". "captured" (real model routing) is out of
#: scope here -- it needs a real model run, not a synthetic generator.
#: "mild skew" and "concentrated" are both realized via EC13's zipf/
#: concentration modes at different parameters (see _sweep_cells_for_ubatch).
SWEEP_LABELS: tuple[str, ...] = (
    "uniform", "mild-skew-zipf", "concentrated-zipf", "concentrated", "single-hot",
)


@dataclass(frozen=True)
class SweepCell:
    """One (distribution, ubatch) point in the boundary matrix, with the
    exact statistics RD30's acceptance criteria name (mean/p95/max
    tokens-per-expert) plus n_active for context."""

    label: str
    mode: str
    n_tokens: int
    n_experts: int
    top_k: int
    seed: int
    n_active: int
    tpe_min: int
    tpe_mean: float
    tpe_p95: float
    tpe_max: int


def _percentile(active_counts: list[int], pct: float) -> float:
    """Nearest-rank percentile over the ACTIVE experts' realized counts
    (the same population hist_stats' min/max/mean are drawn from) -- EC13's
    own hist_stats does not compute p95, only min/max/mean; RD30's
    acceptance criteria explicitly want p95 too, so it is computed here
    rather than added to EC13's core module (keeps EC13's own scope --
    generation + basic stats -- unchanged)."""
    if not active_counts:
        return 0.0
    ordered = sorted(active_counts)
    rank = max(0, min(len(ordered) - 1, round(pct * (len(ordered) - 1))))
    return float(ordered[rank])


def _active_counts(result: mrg.RoutingResult) -> list[int]:
    hist = [0] * result.n_experts
    for expert in result.ids:
        hist[expert] += 1
    return [count for count in hist if count > 0]


def _cell(label: str, result: mrg.RoutingResult) -> SweepCell:
    active_counts = _active_counts(result)
    return SweepCell(
        label=label, mode=result.mode, n_tokens=result.n_tokens,
        n_experts=result.n_experts, top_k=result.top_k, seed=result.seed,
        n_active=result.n_active, tpe_min=result.tpe_min, tpe_mean=result.tpe_mean,
        tpe_p95=_percentile(active_counts, 0.95), tpe_max=result.tpe_max,
    )


def _sweep_cells_for_ubatch(
    n_tokens: int, *, n_experts: int, top_k: int, seed: int,
) -> tuple[SweepCell, ...]:
    cells = [
        _cell("uniform", mrg.generate(
            "uniform", n_tokens=n_tokens, n_experts=n_experts, top_k=top_k, seed=seed)),
        _cell("mild-skew-zipf", mrg.generate(
            "zipf", n_tokens=n_tokens, n_experts=n_experts, top_k=top_k, seed=seed, alpha=0.5)),
        _cell("concentrated-zipf", mrg.generate(
            "zipf", n_tokens=n_tokens, n_experts=n_experts, top_k=top_k, seed=seed, alpha=2.0)),
        _cell("single-hot", mrg.generate(
            "single", n_tokens=n_tokens, n_experts=n_experts, top_k=top_k, seed=seed)),
    ]
    feasible = mrg.concentration_targets(n_experts, top_k, n_tokens)
    if feasible:
        # The most concentrated feasible target that is not the degenerate
        # single-expert case (already covered by "single-hot" above) --
        # the largest tpe_target below the theoretical single-expert total.
        total = n_tokens * top_k
        candidates = [t for t in feasible if t < total] or feasible
        tpe_target = max(candidates)
        cells.append(_cell("concentrated", mrg.generate(
            "concentration", n_tokens=n_tokens, n_experts=n_experts, top_k=top_k,
            seed=seed, tpe_target=tpe_target)))
    return tuple(cells)


def sweep(
    *,
    ubatches: tuple[int, ...] = BOUNDARY_UBATCH,
    n_experts: int = mrg.DEFAULT_N_EXPERTS,
    top_k: int = mrg.DEFAULT_TOP_K,
    seed: int = 1234,
) -> tuple[SweepCell, ...]:
    """The full boundary matrix: every ubatch in ``ubatches`` crossed with
    every distribution EC13 can realize (uniform/mild-skew/concentrated-
    skew/single-hot, plus a feasible "concentrated" point per ubatch where
    one exists). Deterministic: same (ubatches, n_experts, top_k, seed)
    always produces byte-identical results, matching EC13's own
    determinism guarantee (every call here goes through mrg.generate())."""
    cells: list[SweepCell] = []
    for n_tokens in ubatches:
        cells.extend(_sweep_cells_for_ubatch(
            n_tokens, n_experts=n_experts, top_k=top_k, seed=seed))
    return tuple(cells)


def render_table(cells: tuple[SweepCell, ...]) -> str:
    """Human-readable boundary-sweep table -- the concrete form of RD30's
    own acceptance language ("record mean/p95/max tokens-per-expert")."""
    header = f"{'label':<18} {'ubatch':>7} {'n_active':>9} {'tpe_min':>8} {'tpe_mean':>9} {'tpe_p95':>8} {'tpe_max':>8}"
    lines = [header, "-" * len(header)]
    for cell in cells:
        lines.append(
            f"{cell.label:<18} {cell.n_tokens:>7} {cell.n_active:>9} "
            f"{cell.tpe_min:>8} {cell.tpe_mean:>9.1f} {cell.tpe_p95:>8.1f} {cell.tpe_max:>8}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(render_table(sweep()))
