---
id: PNRO04
order: 0
plan: patching-nasone-rdna-optimizations
state: in_progress
created-at: '2026-09-09T10:52:17.766702+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P0
---

# gfx1100 BF16/WMMA fused chunked GatedDeltaNet

## 2026-10-15 authoritative source/receipt reconciliation (BCOP225; pinned b11552)

**Disposition: preserve 1253 as the existing validated, default-on BF16 chunked-GDN implementation on its qualified shapes; do not port a second kernel.** This supersedes the historical predicate-only/untested text below. `engines/llamacpp/patches/1253_nro04_gfx1100_bf16_chunked_gdn/patch.toml` is `validated`, `requires=[]`, `conflicts=[1221_rd50_gdn_chunked_recurrence]`; 1254 MTP prefix/tail independently remains `untested`. The 1253 README/TESTING previously claimed `untested`, an opt-in and a dependency on conflicting 1221; those claims are stale. Historical qualification is b11126, **not** current b11552 qualification.

### Exact implementation and shape/lifetime contract

- `patch.py::GDN_DISPATCH` modifies `ggml/src/ggml-cuda/gated_delta_net.cu::ggml_cuda_op_gated_delta_net_impl`: non-KDA, K=1, n_tokens>1, S_v=128, runtime RDNA3/gfx11 or RDNA4/gfx12 and env `GGML_CUDA_GDN_CHUNKED[ _BF16]` not `0` selects the architecture-specific BF16 kernel by default. S_v=16/32/64 and unsupported architecture go through the sequential kernel; `GGML_CUDA_GDN_CHUNKED_BF16=0` disables BF16 and **does not** select an fp32 chunked alternative (1221 conflicts). gfx1030 is not a qualified BF16 target.
- `patch.py::BF16_GFX11` and `BF16_GFX12` create `gated_delta_net_chunked_bf16_gfx11.cu` and `gated_delta_net_chunked_bf16.cu`, respectively; `gated_delta_net_chunked.cuh` owns the common launcher signature. Each BF16 path enqueues KKT then scan on `ctx.stream()`, uses a `ctx.pool()` BF16 A scratch of `ceil(n_tokens/64)*H*n_seqs*64*64*2` bytes, writes F32 attention output and F32 recurrent state (or existing fused-cache state pointer), and preserves sequential fallback for synchronously rejected launches. The fallback is **not** an async-fault recovery mechanism; do not free/alias a captured scratch or state pointer before its producing stream completes. At n_tokens=512,H=48,n_seqs=1 the nominal scratch is 3 MiB (theoretical allocation request, not measured VRAM).
- **Unqualified shape/abort gap (source-proven conditional):** `GDN_DISPATCH` checks BF16 arch, S_v and K but does not check `H/neqk1`. Both BF16 launchers use a KKT switch supporting only ratios `{1,2,3,4,6,8}`; their default branch `GGML_ABORT("...unsupported GQA ratio")` is reachable for a divisible but unsupported ratio (e.g. H=80,H_k=16 => 5, or H=32,H_k=2 => 16). These are synthetic admissible shapes, **not a reproduced production crash**. Before enabling any new shape/model, add a cheap host-side `H_k>0 && H%H_k==0 && ratio in {1,2,3,4,6,8}` admission check *before allocation/launch*; otherwise retain the sequential path. Mirror the check in 1254's prefix eligibility if modifying that independent untested path. Never hide a failed launch by pretending the BF16 route executed. Use the existing activation marker plus actual kernel identity and expected work accounting.

### Current upstream/fork decision (source inspected)

- [llama.cpp #29353](https://github.com/ggml-org/llama.cpp/pull/29353), merged **2026-10-14 14:00 UTC** as `62df1815bc053066761f4eac58e288f23778e164`, adds `ggml/src/ggml-cuda/gated-delta-net-mma.{cu,cuh}` and a `ggml_cuda_gdn_mma_launch` hook in `gated_delta_net.cu`. Its HIP selector `select_gdn_mma_path` requires `GGML_CUDA_CC_IS_RDNA3_5`, n_tokens>=2048, H_k=16, H in {16,32,48,64}, S_v=128, scalar gate and K<n_tokens; **gfx1100 and gfx1201 take its stock fallback**. It uses a single-block BF16-high/low-residual MMA path with F32 state, rather than BigCherry's two-launch KKT+scan and BF16 scratch. It is **post-b11552** and not a drop-in replacement or AMD performance result for our cards.
- #29353 also relaxes the GGML GDN snapshot contract: when n_tokens>K, only slot 0 is guaranteed. BigCherry `1254_nro05_gdn_mtp_prefix_tail` separately promises exact K snapshot/tail semantics. **Do not silently stack/replace the dispatch** or adopt the relaxed snapshot contract without comparing all MTP slots, rollback and final state. On future pin bump, prefer upstream-first composition and benchmark upstream stock vs existing 1253; preserve 1253 only if it remains source-compatible, correct and faster on gfx1100/gfx1201. Upstream reported NVIDIA/Spark gains and older gfx1151 data are not BigCherry measurements.
- vLLM `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` selects a platform-specific GDN backend and prewarms Triton chunk kernels; use **explicit capability/shape admission and preallocation/warmup** as a test-design reference, not a port. The nasone/RDNA fork 4169fbbf is the already-ported source for 1253; no second fork implementation should be opened.

### Measured evidence, qualification and terminal gates

- Existing `1253/evidence/validation.json` has 8 historical b11126 records, including validated gfx1100 and gfx1201 campaigns. Final validated records report **pp512 +9.31884%** on gfx1100 (CI95 [8.4784%,10.1856%]) and **+10.99672%** on gfx1201 (CI95 [10.8858%,11.1053%]); tg128 controls -0.0071% and +0.0026% respectively. Both report subject GATED_DELTA_NET backend-reference 46/46, control 36/36 within CPU tolerance and positive activation. These are existing, older-pin receipts, not a current b11552 or tensor-split/Flash-Next gain.
- Cheap discriminator actually run: **13/13 source assertions** against current 1253 and merged upstream; **20/20 host-only selector tests** (including unsupported ratios 5/16, Qwen ratio3, KDA, MTP, gfx1030, upstream RDNA3.5 gate). No BigCherry repository suite, patch application, HIP build, GPU execution, storage or new benchmark was run.
- **Gate 0 (offline, before hardware):** verify composed b11552 `patch-rebase-check` and patch-lint with 1253 only (never 1221), then implement a bounded *existing-owner* pre-launch GQA guard if still absent. Test ratios 1/2/3/4/6/8 positive, 5/7/16 negative, S_v!=128, KDA, K>1, disabled env, gfx1030, and graph replay/capture scratch lifetime. `UNSUPPORTED_RATIO_ABORT` is a correctness blocker for any expanded model scope; `NO_COMPOSE` terminates promotion.
- **Gate 1 (only if Gate 0 passes):** on a single gfx1100 and gfx1201, compare same-pin stock, 1253, and upstream #29353 composed only when compatible, at GDN model pp128/512/2048, 8K/24K/98K and ubatch 256/512/1024; record actual KKT+scan or upstream MMA kernels, scratch/VRAM, event/launch time, F32 recurrent-state and full-vocab/logits/KLD, greedy, multi-ubatch, same-process repeated requests, graph replay and no-P2P Meta negative control. Separately qualify 1254 with MTP K=2/4/7 snapshots/acceptance; do not borrow 1253's validation.
- **Promotion/retention:** four independent sessions/architecture, >=10 balanced pairs/session, CI95-low >=3% end-to-end pp and <=1% decode/control regression, no lost work or snapshot divergence. Terminal `KEEP_1253`, `UPSTREAM_SUPERSEDES`, `SOURCE_CONFLICT`, `CORRECTNESS_FAIL`, `NO_GAIN` or `HARDWARE_NOT_QUALIFIED`. No new scheduler, allocator, telemetry, patch or hardware job is queued by this documentation audit.

## Description

TODO, NOT-READY (rescoped). Verified via patch.py: 1253 only defines the eligibility predicate `bigcherry_nro04_gfx1100_bf16_ready()`, which unconditionally `return false;` -- no BF16/WMMA kernel exists and nothing is dispatched. A separate, already-implemented gating predicate (line ~22) uses `GGML_CUDA_CC_IS_RDNA3(cc)`, which is broader than exact gfx1100 (RDNA3 covers more than just gfx1100) -- if exact-gfx1100-only scope is intended, that predicate needs narrowing too.

## Steps

1. Materialize the reviewed BF16/WMMA GatedDeltaNet kernel into ggml/src/ggml-cuda/gated_delta_net.cu (verify this is the correct existing/target file at implementation time), reusing RD50's existing chunked-recurrence structure.
2. Wire dispatch from ggml_cuda_op_gated_delta_net_impl() (verify exact function name) BEFORE the sequential fallback, gated on the corrected eligibility check.
3. Fix `bigcherry_nro04_gfx1100_bf16_ready()` to return a real eligibility result once the kernel exists (it currently unconditionally returns false, so nothing can ever activate); if the intended scope is exact gfx1100 only, also narrow the existing `GGML_CUDA_CC_IS_RDNA3(cc)` gate (line ~22) accordingly, since RDNA3 is broader than gfx1100 alone.
4. Add an actual-launch activation trace marker (BIGCHERRY_PATCH_TRACE-gated) -- none exists today since nothing is ever dispatched.
5. Compile WMMA only in the gfx11 device pass and never cross-select gfx12; add opt-out/force control and preserve sequential fallback.
6. Validate WMMA fragment layout with deterministic identity/structured/random/adversarial matrix probes.
7. Compare one/multi-chunk recurrence, recurrent state, CPU/reference outputs, long-sequence PPL/KL/greedy quality, graph/non-graph execution.
8. Profile KKT/scan separately and promote only gfx1100 exact shape after quality and E2E evidence.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

patches/1253_nro04_gfx1100_bf16_chunked_gdn; GDN translation unit/helpers; shared package tests; WMMA/GDN fixtures

## Validation

Patch mechanics: `PYTHONPATH=tools python -m bigcherry patch-lint patches/1253_nro04_gfx1100_bf16_chunked_gdn`; `PYTHONPATH=tools python -m bigcherry patch-rebase-check --focal-overlay 1253_nro04_gfx1100_bf16_chunked_gdn --source bigcherry-tuning`; package pytest offline (static arch-guard/non-gfx11 compile-exclusion checks). Deterministic WMMA fragment matrix probe (identity/structured/random/adversarial). Hardware (Brutus, gfx1100 required): `python -m bigcherry.patch.validation_campaign --overlay 1253_nro04_gfx1100_bf16_chunked_gdn --arch gfx1100` comparing one/multi-chunk recurrence vs CPU/reference, long-sequence PPL/KL/greedy quality, graph/non-graph execution, KKT/scan profiling; gfx1201/gfx1030 must remain non-selecting controls.

## Effort & Risk



## Standards

Correctness before performance; architecture-specific promotion; source SHA fixed; no synthetic inheritance of RDNA4 evidence.

## Acceptance Criteria

gfx1100 WMMA primitive and GDN/state pass registered tolerances; non-gfx11 devices compile and cannot select; real gfx1100 effect is positive without quality regression; fallback remains verified.

## Notes

Supersedes: NRO04
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-nasone-rdna-optimizations-nro04

Supersedes: NRO04
Inherited semantic scope: preserve gfx1100 geometry, WMMA fragment probe, FP32 state, architecture separation, and fallback gates.
Migration: capability-rebaseline-v3-2026-09

2026-09-24 relevance at b11126: IMPLEMENTED-AS-PATCH. patches/1253_nro04_gfx1100_bf16_chunked_gdn exists, state=untested. Builds on existing RD50 chunked recurrence (already in bigcherry patch history). No upstream gfx1100 BF16/WMMA GDN equivalent found relevant to this project's patch history. Disposition: validate/qualify existing patch; no GPT design needed.

2026-09-24 GPT review req_215c89d0b13a4bb7 applied: verified bigcherry_nro04_gfx1100_bf16_ready() unconditionally returns false and no BF16/WMMA kernel or dispatch exists in 1253 -- added the required kernel-materialization and dispatch-wiring steps plus a required activation marker (none existed). Flagged the separate GGML_CUDA_CC_IS_RDNA3 gate as broader than exact gfx1100 if strict scope is intended.

2026-09-25 (d423644d, f4c75c02): 1253 now ports nasone 4169fbbf (block 02) BF16/WMMA chunked GDN: creates gated_delta_net_chunked.cuh, _bf16_gfx11.cu (RDNA3), _bf16.cu (RDNA4) via new FilePatch(create=True); K==1 S_v==128 prefill on RDNA3/RDNA4 routes to BF16 by default (opt-out GGML_CUDA_GDN_CHUNKED_BF16=0), sequential fallback on launch rejection; fork GDN test cases added. fp32 chunked kernel NOT ported (RD50/1221 scope; reciprocal conflict, no longer requires 1221). Contract NRO04-GDN-CHUNKED-BF16 (prefill positive, decode control, owner policy x4 sessions); producer = GATED_DELTA_NET test-backend-ops backend_reference + marker + paired lanes. Hardware sessions not yet queued.

## Change Log

- 2026-09-09T10:52:17.766702+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:08:46.573893+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.067141+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.708292+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:26:15.401708+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes
- chg_20260910_022800_five-nasone-successor-plans-no_4030
- 2026-09-10T02:28:00.297184+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:26:11.680244+00:00 (updated-by): Updated: section:validation, section:notes
- 2026-09-24T04:48:19.490128+00:00 (updated-by): Updated: section:description, section:steps, section:notes
- 2026-09-25T04:23:36.714622+00:00 (updated-by): Updated: section:notes
- 2026-09-25T04:23:39.602014+00:00 (state-transition): State: pending → in_progress
- chg_20260926_012534_qwen3536-hybrid-model-promp_1365
- 2026-09-26T01:25:37.283467+00:00 (updated-by): Updated: section:ledger-events
