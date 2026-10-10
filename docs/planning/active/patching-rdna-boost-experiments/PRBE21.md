---
id: PRBE21
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:53.078402+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P3
---

# Fold SSM conv_input concat into qkv mmvq; rpb=2 for small-K MoE

## 2026-10-10 authoritative implementation audit — conv-input CONCAT/CPY only

**Disposition:** PRBE21's proposed `CONCAT -> SSM_CONV`/QKV-MMVQ fold is **not ready to implement**. Pinned llama.cpp b11474 and current upstream still materialize `conv_input` for *two independent consumers*: SSM convolution and recurrent-state CPY. A conv-only fused read that skips the CONCAT producer leaves the CPY reading unwritten/stale `conv_input` storage. This is a source-level graph/lifetime invariant, not a reproduced GPU defect or performance measurement. The original Steps 1–4 and proposed `12xx_rd27_ssm_conv_fold` are historical hypotheses, **superseded by the gate below**. Do not allocate a patch ID or add `BIGCHERRY_SSM_CONV_INPUT_FOLD` until this gate passes.

### Source trace / ownership (b11474)

- `src/models/qwen35.cpp::build_layer_attn` (~399–409): `build_conv_state(...)` returns `conv_input`; `ggml_ssm_conv(ctx0, conv_input, conv_kernel)` and SiLU consume it. `qwen35moe.cpp` and `qwen3next.cpp` also call the same shared builder; check each graph rather than assuming identical shapes.
- `src/models/delta-net-base.cpp::build_conv_state` (~463–524): `conv_states = build_rs(...)`, reshape; `qkv_mixed = ggml_transpose(...)`; `ggml_concat(ctx0, conv_states, qkv_mixed, 0)` produces the contiguous `conv_input`. For `n_rs_seq==0`, `ggml_view_3d(conv_input, ...)` is copied to `conv_states_all` by an explicit `ggml_build_forward_expand(gf, ggml_cpy(...))`. For `n_rs_seq>0`, K=`n_rs_seq+1` distinct view/CPY tail slots are built, with `s_idx=max(0,n_tokens-K+t)`. These are live state effects, not dead views.
- `ggml/src/ggml-cuda/concat.cu::ggml_cuda_op_concat` writes the full output; the dimension-0 fast path selects contiguous versus strided input. The transposed QKV source can require the noncontiguous path. `ssm-conv.cu::ggml_cuda_op_ssm_conv` asserts F32 and `src0->nb[1] == src0->ne[0]*sizeof(float)`, dispatching widths 3/4/5/9/15; it **cannot** simply receive the original transposed QKV pointer.
- `ggml/src/ggml-cuda/ggml-cuda.cu::ggml_cuda_try_fuse` (~4282–4289) only recognizes SSM_CONV+SiLU and SSM_CONV+ADD+SiLU. `ggml_cuda_can_fuse` requires graph-adjacent matches and checks output memory ranges. No CONCAT+SSM_CONV or CONCAT+CPY matcher exists. Skipping a non-adjacent CPY is not a valid shortcut.
- Existing stock fusion owns the SSM_CONV+SiLU epilogue. **PRBE21** owns only the `conv_input` producer and recurrent-state publication opportunity. **PRBE18** owns longer SSM pre-scan fusion; **PRBE41/1263** owns channels-major SSM_CONV layout; `0600_mmvq_geometry` owns MMVQ geometry. PRBE21's independent small-K rpb=2 catalog candidate is **not audited or changed here**. Do not merge these mechanisms into one patch or queue.

### Cheapest discriminator completed (no GPU)

Pinned-source structural checks: 12/12 passed (CONCAT producer, Qwen35 SSM consumer, both CPY modes, F32 contiguous SSM requirement, independent CUDA dispatch and existing fusion bounds). A disposable NumPy host reference checked width 3/4/5/9/15, token counts 1/3/8/32, normal and K-tail state updates: **35/35** virtual two-input convolution plus explicit state-write cases matched the materialized CONCAT reference; **35/35** omission-of-state-write controls differed from the required CPY output. These prove the *dataflow requirement in the host model*, not GPU graph legality, performance or an implementation.

### Bounded decision before any code

1. Reuse the existing PRBE113 decode profiler/graph trace when its owner completes; **do not queue a competing hardware lane**. Census actual Qwen3.5/3.6 27B F32 decode and verify graph node IDs, device ownership, `CONCAT -> VIEW/CPY` and `CONCAT -> SSM_CONV` edges, adjacency, copy count, alias/overlap, n_rs_seq, CUDA launch count, non-overlapped CONCAT+CPY wall time and total request time. Record separately prefill and decode, one/dual XTX and R9700; gfx1030 is a fallback/control, not a presumed accelerator. No P2P dependency.
2. **Terminal reject** if no safe adjacent producer/CPY match exists, if state updates cannot be preserved without new graph rewrites, or if the measured non-overlapped CONCAT+CPY fraction is below `0.03/1.03 = 2.9126%` of E2E time. This fraction is the theoretical ceiling for a 3% speedup even if both operations became free; real gain is smaller. No measured PRBE21 improvement exists.
3. Only if all gates pass, prototype a **single opt-in CONCAT+state-CPY producer**, not a QKV-MMVQ or SSM_CONV rewrite: in `concat.cu` write the unchanged contiguous `conv_input` and the exact state-tail destination from one launch; in `ggml_cuda_try_fuse` admit only a proven adjacent `CONCAT,VIEW?,CPY` subgraph with output-node/range checks and a verified downstream SSM consumer. Start with F32, `n_rs_seq=0`, one sequence, one token, same device, non-overlapping destinations and supported conv widths. Use source `nb[]` for transposed QKV; retain existing SSM_CONV(+SiLU) and CPY semantics. Any unproven adjacency, alias, shape, cross-device edge, graph-capture or state-tail case takes native CONCAT+CPY unchanged. Do not introduce a new allocator, cache, scheduler or dispatcher.
4. Before promotion: host shape/offset/alias fixture; CPU/backend F32 reference for CONCAT and every CPY slot, `n_rs_seq=0` and K-tail negative controls, state-restore/MTP acceptance, logits/greedy parity, repeated same-process multi-request, multi-ubatch and long-context graph replay; verify exact writes/bytes/launch counts (no disappearing work). Benchmark matched 8K/24K/98K prefill/decode on gfx1100/gfx1201; tensor-split, gfx1030 and Flash-Next non-activation controls. Four independent sessions per architecture, >=10 paired ABBA rounds/session; promote only CI95-low >=3% E2E and <=1% controls, otherwise close. No new build, HIP run or benchmark occurred in this audit.

### Current external mechanisms (not AMD qualification)

- Pinned b11474 and current `llama.cpp` `delta-net-base.cpp` and `ssm-conv.cu` retain this producer/consumer topology (last substantive source commits in July/June); `llama.cpp` [#30198](https://github.com/ggml-org/llama.cpp/pull/30198) changes redundant `ggml_cont` placement in related model code, **not** this CONCAT/CPY boundary. **Wait; do not port a new fusion from upstream.**
- [vLLM causal_conv1d.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py) explicitly couples convolution reads to state publication and speculative-accept offsets. It is a *state-lifetime design reference*, not a CUDA/HIP speed result.
- [SGLang #38623](https://github.com/sgl-project/sglang/pull/38623) (open) corrects oversized/circular conv-state indexing with dedicated tests; its actual diff reinforces tail-offset/rollback coverage, not an adoptable RDNA kernel. Avoid transplanting NVIDIA/Triton assumptions.

**Evidence:** source-level and host-model only. No BigCherry PRBE21 activation, hardware performance, GPU correctness, build or commit existed before this documentation change. All earlier "expected one launch saved" statements are hypotheses pending graph census.



## Description

TODO, two independent sub-candidates. (1) SSM conv_input-concat folding into qkv MMVQ (fork source 0510d7cfa) -- genuinely new fusion work, same family as PRBE18. (2) rpb=2 for small-K MoE -- this project ALREADY has the mechanism: mmvq.cu's calc_rows_per_block(ncols_dst, table_id, small_k, nwarps) has a small_k parameter (verified b11126) returning `small_k ? nwarps : 1` for ncols_dst==1 on GENERIC/GCN/TURING/GB10 tables, and the HI09 explicit-geometry template params (nwarps_explicit/rows_per_block_explicit, patch 0600_mmvq_geometry, state=validated core framework) already let a generated instance force rows_per_block=2 directly -- sub-candidate 2 is a catalog-driven candidate-enumeration task, not new kernel code.

## Steps

Sub-candidate 1 (conv_input fold):
1. CORRECTED per source audit: at b11126 the conv_input concat is explicitly a real `GGML_OP_CONCAT` op, not an internal SSM_CONV detail -- verified anchor: src/models/delta-net-base.cpp:472 `ggml_tensor * conv_input = ggml_concat(ctx0, conv_states, qkv_mixed, 0);`. conv_input also feeds recurrent-state VIEW/CPY updates (the conv_state_last -> conv_state_update CPY consumers) which any fusion rewrite must NOT break -- audit and explicitly preserve those consumers before authoring the fold.
2. Define the exact CONCAT->SSM_CONV subgraph: the edges, out-nodes, and the memory-safety check (via ggml_check_edges/output-range check) that proves the fold only fires when conv_input's sole meaningful consumers are the SSM_CONV read and the preserved CPY updates; guard: contiguous conv_state, F32, ne[1]==1 (decode shape).
3. Insert the fold at the same fusion-detection site used for PRBE18 (ggml-cuda.cu's ggml_cuda_try_fuse-family idiom) as a sibling branch anchored on the exact `ggml_concat(ctx0, conv_states, qkv_mixed, 0)` model-builder line plus its corresponding CUDA fusion dispatch site.
4. Fallback to native concat+conv when guard fails, preserving the conv_state_last->conv_state_update CPY path unchanged in all cases.
Sub-candidate 2 (rpb=2 small-K MoE):
5. Split this into its OWN separate candidate package/set (CORRECTED -- do not bundle with sub-candidate 1's patch). Do NOT hand-write a new kernel. Use tools/bigcherry/tuning/catalog.py's enumerate_mmvq() to add exact (type, ncols, K, nwarps, rpb=2) candidate rows for the target quant types at small expert hidden-dim (K) shapes seen in tiny/A3B MoE configs, gated gfx1100. Declare `requires=["0600_mmvq_geometry"]` on this package (CORRECTED -- was previously omitted despite depending on the HI09 explicit-geometry template mechanism from that patch).
6. Run the existing record->tune->promote campaign (tools/bigcherry/campaign/campaign.py) scoped to this candidate set; independent of PRBE05/PRBE18.
7. Compare against the native (small_k=false) row as control.

## Detailed Solution & Technical Design

Sub-candidate 1 needs new fusion-detection/kernel work (see PRBE18's plan for the analogous mechanism). Sub-candidate 2 is materially smaller than the item text implies once mapped onto the real framework: `calc_rows_per_block`'s `small_k` branch and the HI09 `rows_per_block_explicit` template parameter already exist and are validated infrastructure (patch 0600_mmvq_geometry, GROUP=core, STATE=validated) -- PRBE21's job for sub-candidate 2 is to define the right (type, shape-signature) candidate rows in the catalog and run the campaign, not to touch mmvq.cu's kernel body at all.

## Code Samples & Guidance

Real b11126 anchor (verified), ggml/src/ggml-cuda/mmvq.cu calc_rows_per_block:
```
if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING || table_id == MMVQ_PARAMETERS_GB10) {
    switch (ncols_dst) {
        case 1:
            return small_k ? nwarps : 1;
        ...
```
Note MMVQ_PARAMETERS_RDNA3_0 (gfx1100's real table_id) and MMVQ_PARAMETERS_RDNA4 are NOT in this small_k-aware list -- their calc_rows_per_block falls through to the final `return 1;` regardless of small_k. This means gfx1100's native calc_rows_per_block never uses small_k today; sub-candidate 2's rows must be forced via the HI09 explicit template params (nwarps_explicit/rows_per_block_explicit), not by expecting small_k to already do anything on RDNA3. Patch package for sub-candidate 1: patches/12xx_rd27_ssm_conv_fold/patch.toml (state=untested, plan-item=RD27, requires=[] unless PRBE18's fusion-detection branch lands first, in which case list it). Sub-candidate 2 is expressed as new catalog.py candidate rows plus a campaign config, not a hand-authored patch package edit to mmvq.cu.

## Files

Sub-candidate 1: ggml/src/ggml-cuda/ssm-conv.cu/.cuh, ssm-scan.cu/.cuh, ggml-cuda.cu (fusion detection); patches/12xx_rd27_ssm_conv_fold/. Sub-candidate 2: tools/bigcherry/tuning/catalog.py (enumerate_mmvq), campaign config for the small-K MoE signature set, no source-file Edit.

## Validation

Sub-candidate 1: test-backend-ops SSM_CONV correctness, fused-vs-unfused equality, gfx1100 timing. Sub-candidate 2: `PYTHONPATH=tools python -m bigcherry patch-lint`/campaign dry-run on the new candidate rows, test-backend-ops MUL_MAT_ID small-K coverage, then the existing record->tune->promote->replay campaign (tools/bigcherry/patch/validation_campaign.py) on Brutus comparing small_k=2 vs native for the target MoE shapes -- not run here.

## Effort & Risk

Sub-candidate 1: M (new fusion + kernel work, similar risk profile to PRBE18). Sub-candidate 2: S (catalog-only, reuses validated HI09 infrastructure) -- the two should be tracked and possibly promoted independently since they have very different effort/risk.

## Standards

Dependency audit; exact pattern; memory safety; isolated qualification.

## Acceptance Criteria

Fused path is correct for exact SSM/small-K patterns, unsupported patterns fall back, and any benefit is shown in isolated causal evidence before composition.

## Notes

Supersedes: RD27
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd27

2026-09-24 relevance at b11126: TODO. Sub-candidate 2 is materially cheaper than the item text implies: verified that calc_rows_per_block's small_k parameter and the HI09 nwarps_explicit/rows_per_block_explicit template mechanism already exist as validated core framework (patch 0600_mmvq_geometry); gfx1100's RDNA3_0 table_id does not consume small_k natively, so the explicit-geometry override path is required. GPT design request submission for this item hit a rejected/queue-saturated gateway (concurrency limit reached: 8 queued, 2 running gateway-wide) and was not resubmitted successfully in-session; plan authored directly from verified source.

2026-09-24 GPT review req_2b717df095b44703 applied: resolved sub-candidate 1's central design choice -- verified conv_input is explicitly GGML_OP_CONCAT at src/models/delta-net-base.cpp:472, and added the requirement to preserve its conv_state_last->conv_state_update CPY consumers, which the prior plan did not account for. Split sub-candidate 2 into its own separate package/candidate set with requires=["0600_mmvq_geometry"] (previously bundled and missing that dependency), with exact (type,ncols,K,nwarps,rpb=2) candidate rows to be enumerated at implementation time.

## 2026-10-08 b11474 experiment scope

**Rank 8b, 27B GDN-only; two independent experiments.** (a) Source `src/models/delta-net-base.cpp` `ggml_concat(ctx0, conv_states, qkv_mixed, 0)` feeding SSM_CONV and recurrent CPY updates; existing `ggml/src/ggml-cuda/ssm-conv.cu` candidate folds only eligible contiguous decode CONCAT without deleting the CPY/state consumer. Gate `BIGCHERRY_SSM_CONV_INPUT_FOLD=0|1`, default 0; expected one launch + intermediate buffer saved per activated GDN step, perhaps <1% decode. (b) rpb=2 for small-K MoE MMVQ is already expressible through validated `0600_mmvq_geometry` explicit nwarps/rpb template and `ggml/src/ggml-cuda/mmvq.cu::calc_rows_per_block`, not a new kernel; gate `BIGCHERRY_MMVQ_SMALLK_RPB2=0|1` via independently compiled candidate and shape selector. Do **not** bundle (a)/(b), or infer wins without actual 27B GDN/small-K MoE signatures. `queue-env-ab.sh`: width 1..8 decode, 8K/24K/98K, op parity/temp0 identity, CPY state parity, capture safety, kernel launch/occupancy and unaffected dense/Flash-Next control. Both remain low priority until PRBE113 profile.

Recheck every historical b11126 anchor on the composed b11474 source before coding; no GPU run or patch implementation is claimed by this plans-only triage.

## Change Log

- 2026-10-08 (triage): Kept experiment pending (P3); source/shape, coverage against #29901/1202/1253, env gate, expected effect and separated hardware test defined above.

- 2026-09-09T10:54:53.078402+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:12:06.531616+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.222985+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.942391+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:51:58.669575+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025218_rdna-successors-prbe2022-now_5714
- 2026-09-10T02:52:18.419086+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:30:22.125041+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
- 2026-09-24T04:42:14.202872+00:00 (updated-by): Updated: section:steps, section:notes
