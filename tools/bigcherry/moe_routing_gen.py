"""EC13 (AMD-TEST-001): synthetic MoE expert-routing generator.

Ported from the routing-generation ALGORITHM in AMD-Ecosystem/llama.cpp PR #62
("tests: add MoE MMQ benchmark with routing-distribution generator",
https://github.com/AMD-Ecosystem/llama.cpp/pull/62, draft, NOT merged as of
2026-08-20 -- `gh api repos/AMD-Ecosystem/llama.cpp/pulls/62` confirms
state=open, merged=false, merge_commit_sha unused). PR #62 itself is a
standalone C++ benchmark tool (tests/benchmark-moe-mmq.cpp) that builds a real
ggml_mul_mat_id graph and times it on a GPU backend. This module ports ONLY
the routing-distribution math (counts_uniform/counts_zipf/assign_ids/
hist_stats) into Python, backend-neutral, so it can generate reproducible
routing tables usable by BigCherry's own tooling (feeding RD30/RD31/RD32's
boundary sweeps per EC13's own notes) without requiring a GPU build. The
actual GPU timing/roofline half of PR #62 is out of scope here -- see EC13's
own framing ("backend-neutral harness ... testing/harness, preferably
BigCherry tooling rather than a permanent runtime patch").

Defaults (n_experts=256, top_k=8, n_embd=2048, n_ff=512) match PR #62's own
defaults for Qwen3.6-35B-A3B, per EC13's own code_samples ("real expert
dimensions from Qwen3.6-35B-A3B").

Every generator function takes an explicit seed and is deterministic: same
(seed, mode, params) always produces byte-identical output, matching EC13's
own acceptance criteria ("generated expert-assignment statistics and
MUL_MAT_ID output must be deterministic").
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

DEFAULT_N_EXPERTS = 256
DEFAULT_TOP_K = 8
DEFAULT_N_EMBD = 2048
DEFAULT_N_FF = 512

MODES = ("uniform", "single", "zipf", "concentration")


class RoutingGenError(ValueError):
    pass


@dataclass(frozen=True)
class RoutingResult:
    """One generated routing table plus its own observed statistics.

    ``ids`` is the per-token expert-assignment list, flattened as
    ``ids[token * top_k + slot]`` -- the same logical shape PR #62's
    ``assign_ids`` produces and llama.cpp's ``ggml_mul_mat_id`` `ids` tensor
    expects (one row of ``top_k`` distinct expert indices per token).
    """

    mode: str
    n_experts: int
    top_k: int
    n_tokens: int
    seed: int
    ids: tuple[int, ...]
    tpe_target: int
    n_active: int
    tpe_min: int
    tpe_max: int
    tpe_mean: float


def _validate_shape(n_experts: int, top_k: int, n_tokens: int) -> None:
    if n_experts < 1:
        raise RoutingGenError(f"n_experts must be >= 1, got {n_experts}")
    if top_k < 1 or top_k > n_experts:
        raise RoutingGenError(
            f"top_k must be in [1, n_experts]; top_k={top_k}, n_experts={n_experts}")
    if n_tokens < 1:
        raise RoutingGenError(f"n_tokens must be >= 1, got {n_tokens}")


def counts_uniform(n_experts: int, n_active: int, total: int) -> list[int]:
    """Target tokens-per-expert, spread evenly across ``n_active`` experts.

    Direct port of PR #62's ``counts_uniform``: the first ``total % n_active``
    active experts get one extra token so the counts sum exactly to ``total``.
    """
    n_active = max(1, min(n_active, n_experts))
    counts = [0] * n_experts
    base, rem = divmod(total, n_active)
    for i in range(n_active):
        counts[i] = base + (1 if i < rem else 0)
    return counts


def counts_zipf(n_experts: int, n_active: int, alpha: float, total: int) -> list[int]:
    """Target tokens-per-expert under a power-law (Zipf) distribution.

    Direct port of PR #62's ``counts_zipf``: expert rank ``i`` (0-indexed)
    gets weight ``1 / (i + 1)**alpha``, normalized to ``total``; rounding
    remainder is handed to experts round-robin starting from rank 0 (the
    heaviest), matching the fork's own remainder-assignment loop.
    """
    n_active = max(1, min(n_active, n_experts))
    weights = [1.0 / ((i + 1) ** alpha) for i in range(n_active)]
    sum_w = sum(weights)
    counts = [0] * n_experts
    assigned = 0
    for i in range(n_active):
        counts[i] = math.floor(weights[i] / sum_w * total)
        assigned += counts[i]
    i = 0
    while assigned < total:
        counts[i % n_active] += 1
        assigned += 1
        i += 1
    return counts


def concentration_targets(n_experts: int, top_k: int, n_tokens: int) -> list[int]:
    """Feasible tokens-per-expert (tpe) targets for the concentration sweep.

    Direct port of PR #62's concentration-mode loop: tpe doubles from 16,
    total = n_tokens * top_k must divide evenly by tpe, and the resulting
    active-expert count must land in [top_k, n_experts] (fewer than top_k
    experts is impossible with distinct top-k routing per token; more than
    n_experts does not exist).
    """
    total = n_tokens * top_k
    targets = []
    tpe = 16
    while tpe <= total:
        if total % tpe == 0:
            n_active = total // tpe
            if top_k <= n_active <= n_experts:
                targets.append(tpe)
        tpe *= 2
    return targets


def assign_ids(
    counts: list[int], n_experts: int, top_k: int, n_tokens: int, seed: int,
) -> list[int]:
    """Realize a target per-expert count histogram as a per-token expert list.

    Direct port of PR #62's ``assign_ids``: each token greedily takes the
    ``top_k`` experts with the most remaining budget (after an RNG shuffle to
    break ties without bias), which spreads any single heavy expert's tokens
    across many different token slots rather than clustering them -- the same
    behavior the fork's benchmark relies on to keep every token's expert set
    distinct. Deterministic: a Python ``random.Random(seed)`` instance is used
    exclusively (no global RNG state touched), so the same seed always
    produces the same shuffle sequence and therefore the same output.
    """
    rng = random.Random(seed)
    remaining = list(counts)
    ids: list[int] = [0] * (top_k * n_tokens)
    idx = list(range(n_experts))
    for t in range(n_tokens):
        rng.shuffle(idx)
        idx.sort(key=lambda e: remaining[e], reverse=True)
        for u in range(top_k):
            e = idx[u]
            ids[t * top_k + u] = e
            if remaining[e] > 0:
                remaining[e] -= 1
    return ids


def hist_stats(ids: list[int], n_experts: int) -> tuple[int, int, int, float]:
    """Return (n_active, tpe_min, tpe_max, tpe_mean) over the realized ids.

    Direct port of PR #62's ``hist_stats``.
    """
    hist = [0] * n_experts
    for e in ids:
        hist[e] += 1
    active_counts = [c for c in hist if c > 0]
    if not active_counts:
        return 0, 0, 0, 0.0
    return (
        len(active_counts),
        min(active_counts),
        max(active_counts),
        sum(active_counts) / len(active_counts),
    )


def generate(
    mode: str,
    *,
    n_tokens: int,
    n_experts: int = DEFAULT_N_EXPERTS,
    top_k: int = DEFAULT_TOP_K,
    seed: int = 1234,
    alpha: float = 1.0,
    tpe_target: int | None = None,
) -> RoutingResult:
    """Generate one deterministic routing table under ``mode``.

    ``mode`` is one of :data:`MODES` (uniform/single/zipf/concentration),
    mirroring PR #62's four distributions exactly:

    - uniform: every expert gets an (almost) equal token share.
    - single: one expert gets every token (worst-case load imbalance).
    - zipf: power-law imbalance, tunable via ``alpha`` (higher = more skewed).
    - concentration: a specific tokens-per-expert target from
      :func:`concentration_targets`, interpolating between uniform (large
      active set) and single (small active set) -- ``tpe_target`` is
      required for this mode and must be one of that function's outputs.
    """
    _validate_shape(n_experts, top_k, n_tokens)
    total = n_tokens * top_k

    if mode == "uniform":
        counts = counts_uniform(n_experts, n_experts, total)
        target = total // n_experts
    elif mode == "single":
        counts = counts_uniform(n_experts, 1, total)
        target = total
    elif mode == "zipf":
        counts = counts_zipf(n_experts, n_experts, alpha, total)
        target = 0
    elif mode == "concentration":
        if tpe_target is None:
            raise RoutingGenError("mode='concentration' requires tpe_target")
        feasible = concentration_targets(n_experts, top_k, n_tokens)
        if tpe_target not in feasible:
            raise RoutingGenError(
                f"tpe_target={tpe_target} is not feasible for "
                f"n_experts={n_experts}, top_k={top_k}, n_tokens={n_tokens} "
                f"(feasible targets: {feasible})")
        n_active = total // tpe_target
        counts = counts_uniform(n_experts, n_active, total)
        target = tpe_target
    else:
        raise RoutingGenError(f"unknown mode {mode!r}; must be one of {MODES}")

    ids = assign_ids(counts, n_experts, top_k, n_tokens, seed)
    n_active, tpe_min, tpe_max, tpe_mean = hist_stats(ids, n_experts)

    return RoutingResult(
        mode=mode, n_experts=n_experts, top_k=top_k, n_tokens=n_tokens,
        seed=seed, ids=tuple(ids), tpe_target=target, n_active=n_active,
        tpe_min=tpe_min, tpe_max=tpe_max, tpe_mean=tpe_mean,
    )
