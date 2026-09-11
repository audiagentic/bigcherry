# 1237 (RD30 / AMD-MOE-001): compact the MoE MMQ launch grid

## Scope

Adds a prep kernel (`mmq_build_moe_block_map`) that turns `expert_bounds[]`
(already computed every MoE dispatch by `mm_ids_helper`) into
`block_start[n_experts+1]` (prefix sum of per-expert real tile counts) and
`block_expert[max_m_blocks]` (a flat expert-per-block map), then launches
`dim3(nty, max_m_blocks, 1)` instead of upstream's rectangular
`dim3(nty, ntx, ntzw)` worst-case grid, which gives every expert the same
tile-column count regardless of its real routed-token occupancy. Falls back
to the exact legacy grid whenever the compact map would exceed the grid.y
limit or the prep kernel's shared-memory prefix sum would exceed the
device's per-block budget. Gated to `nsamples_y == 1` and
`cc == GGML_CUDA_CC_RDNA3` exactly (gfx1100 only, deliberately excluding
RDNA3.5's different cc range).

## Why

A real rocprofv3 capture on Qwen3.6-35B-A3B (n_expert=256) confirmed the
rectangular grid launches far more blocks than routed-token occupancy
requires. AMD-Ecosystem/llama.cpp PR #63 reports a modest +1.9-5.4% prefill
gain from this exact compaction on their own hardware/models.

## Upstream / provenance

Concept ported from AMD-Ecosystem/llama.cpp PR #63 (`f8864197a4e2`),
redesigned against this project's real pinned source rather than ported
verbatim -- confirmed during implementation that the kernel body's
non-stream-K branch already early-returns for out-of-range tiles under the
legacy rectangular grid, so this patch's real benefit is reduced
launch/dispatch overhead, not eliminated compute cycles (consistent with
AMD's own modest reported gain, and bounding the correctness blast radius:
a compaction bug shows up as missing/misrouted work, not an unexplained
large timing swing). Design verified against the real materialized source
via `dev-gpt-agent` review (session `ses_35768076803a4361`).

## Real hardware evidence (2026-08-24/25, dual gfx1100 Brutus)

This is the most extensively validated untested patch in the RD-series:

1. **Build**: clean compile, both single-arch (gfx1100, 343/343 targets)
   and multi-arch (gfx1100+gfx1201+gfx1030, matching production build
   coverage, 473/473 targets). One real compile bug found and fixed on
   first attempt (an anchor matched only the launch function's signature
   line, not the `template<...>` line directly above it, orphaning the
   template) -- exactly the class of defect real-hardware compilation
   catches that offline anchor-matching cannot.
2. **Correctness, native dispatch mode**: `test-backend-ops` MUL_MAT_ID,
   869/869 passed across all supported quant types (native mode still runs
   through `launch_mul_mat_q`, so the compact grid is exercised here too --
   it's gated on hardware cc + real MoE shape, not on dispatch mode).
3. **Correctness, tune dispatch mode**: q4_K MUL_MAT_ID 73/73 passed, 100%
   dispatch coverage; q8_0 MUL_MAT_ID 75/75 passed, 100% coverage -- both
   quant types match the real production model's expert weight
   quantization. A temporary instrumented rebuild confirmed the compact
   path (not the legacy fallback) actually executed on every relevant
   case, correctness held throughout.
4. **Hostile-routing correctness at real production scale** (n_expert=256):
   a dedicated standalone HIP test
   (`tools/tests/hardware/test_rd30_hostile_routing.py` +
   `tools/tests/fixtures/hardware/rd30_hostile_test.cu`) against a
   host-side reference implementation of the same prefix-sum +
   binary-search-scatter algorithm: single-hot, concentrated-8-of-256,
   zipf-skew, uniform, and all-zero-degenerate routing -- 0/5 cases failed,
   on both this project's Windows dev machine and Brutus (dual gfx1100).
5. **Real production-model timing** (Qwen3.6-35B-A3B Q4_K_M, dual gfx1100,
   `-sm tensor`, real n_expert=256): sequential A/B showed pp512 +0.94%,
   pp2048 +0.79%, tg128 +0.19% (decode near-noise, unaffected as designed
   -- decode does not enter the compacted MMQ path).
6. **Interleaved A/B** (addressing the legitimate concern that a single
   sequential run cannot rule out systematic drift): 3 interleaved rounds
   (baseline/RD30/baseline/RD30/baseline/RD30). Baseline clustered tightly
   at 1976-1978 t/s pp512 across all 3 independent rounds; RD30 clustered
   tightly at 1992-1995 t/s across all 3. The two clusters never
   overlapped (closest gap ~13.5 t/s / 0.7%, well outside any single run's
   own +/-4-6 t/s stdev), RD30 won every round, no ordering flip -- ruling
   out drift/noise as the explanation.

## Lifecycle status: deliberately still `untested`

This patch's real-hardware evidence (build, correctness across native and
tune dispatch modes and two quant types, hostile-routing at real scale, and
an interleaved A/B establishing the ~0.7-0.9% pp gain as genuine signal
rather than noise) is complete for its **core performance and correctness
claim**. It is deliberately NOT promoted to `"validated"` yet, per an
explicit GPT solution-approval review requested for this README
(dev-gpt-agent, session `ses_118f7ca91aac496e`): PRBE23 (RD30's
capability-rebaseline successor plan item) names "durable candidate
identity" in BigCherry's own dispatch/disposition registry as an explicit
acceptance criterion, not a nice-to-have -- meaning a full
record->tune->promote->replay campaign through BigCherry's own
candidate-tuning pipeline (distinct from, and more rigorous than, the
manual `llama-bench` A/B already run) is still required before promotion.
This matches this project's own lifecycle doctrine: "a campaign completing
successfully is not enough by itself... promotion is a deliberate mutation
after qualification, never an automatic campaign side effect." The manual
A/B evidence above is real and strong, but is not a substitute for that
qualification path.

## Known limitations

- No `validation.toml` adapter exists yet (no bound Experiment Contract).
- The record->tune->promote->replay campaign establishing durable candidate
  identity (PRBE23's remaining acceptance-criterion gap) is the concrete
  next step before this patch can be promoted.
- Not measured at real `n_expert` values other than 256 (the compact-map
  algorithm is shape-generic -- a prefix sum + binary-search scatter over
  `n_experts` -- so this is a lower-confidence extrapolation for other
  scales, not directly measured).
