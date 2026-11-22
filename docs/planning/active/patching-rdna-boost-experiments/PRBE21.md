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

# SSM conv_input concat fusion — conditional real-graph and state-CPY proof; no MoE rpb2 rewrite

## Description

One possible SSM-specific GGML_OP_CONCAT-to-SSM_CONV producer fusion only. The prior multi-token MoE small-K rpb2 proposal is TERMINAL: b11552 already routes has_ids && ncols_dst>1 to dedicated mul_mat_vec_q_moe_launch with rows_per_block=2; existing catalog.py generates generic small_k candidates. Do not add a new MoE geometry candidate, kernel, catalog sweep or patch. This SSM candidate is subject to a fresh current-source graph trace, producer/conv state-copy lifetimes, and a >=3% removable E2E opportunity; otherwise close without code.

## Steps

1. Reconfirm the current pinned llama.cpp SSM graph (delta-net-base.cpp or its current equivalent), source conv_input GGML_OP_CONCAT, SSM_CONV consumer and conv_state_last/conv_state_update VIEW/CPY edges.
2. With current profiling, establish actually executed concat/conv pattern and at least 2.9126% non-overlapped E2E time attributable to a removable boundary; if absent, close NO_EXPOSED_GAIN without a patch.
3. Only on a positive gate, reuse existing ggml_cuda_try_fuse/graph and kernel owners to propose a narrowly opt-in producer/SSM_CONV fusion; preserve every recurrent-state VIEW/CPY and graph-replay consumer, and retain unchanged fallback.
4. Test exact backend-op positive/negative shapes, resource/graph lifetime, multiple requests and MTP state-restore/greedy/logit parity. Compare paired whole-lane ABBA with >=3% reproducible gain and <=1% controls regression.
5. Never create a small-K MoE rpb2 path or a second MMVQ geometry selector; 0600 generic and the dedicated native MoE route already cover those cases.

## Detailed Solution & Technical Design

Only one source-specific conditional SSM-conv_input boundary remains. Reuse the existing model GGML_OP_CONCAT and native SSM_CONV consumer path, with explicit recurrent VIEW/CPY safety before altering any producer. If source graph/overlap or route cannot be proven, stop without code. The historical multi-token MoE rpb2 suggestion has no independent implementation or performance acceptance; its exact source gate and negative controls are archived with BCOP139. No second fusion family or architecture dispatch table.

## Code Samples & Guidance

SSM source guidance only: trace the actual current model's GGML_OP_CONCAT conv_input, the SSM_CONV data consumer and the recurrent-state VIEW/CPY update graph before any fusion branch. Continue only on measured real-path non-overlapped E2E exposure. Reuse the native ggml_cuda_try_fuse-family source/graph builder and existing state lifetime rules, retain the original fallback. Native multi-token MoE rpb2 is already the dedicated upstream route; never copy old small_k MMVQ geometry snippets into an SSM candidate. Historical multi-token notes are preserved in archived BCOP139.

## Files

Current pinned llama.cpp src/models/delta-net-base.cpp or resolved equivalent; existing ggml-cuda.cu try_fuse dispatcher and SSM_CONV implementation only after gate; existing tools/bigcherry/tuning and tests. No new MoE rpb2 package or tuning catalog.

## Validation

Source/graph producer and all recurrent state consumers; executed route; bound on exposed E2E; backend-op and fallback; restored recurrent state, multiple requests and replay; MTP/greedy/logit integrity; paired ABBA. No GPU queue until source gate succeeds.

## Effort & Risk

S for executed-graph and overlap attribution, M only if a real SSM-only fusion passes the source gate. Main correctness risk is lost/reordered conv_state_last-to-conv_state_update CPY, changed recurrent state or graph replay. The old MoE rpb2 candidate is terminal and carries no implementation effort.

## Standards

Dependency audit; exact pattern; memory safety; isolated qualification.

## Acceptance Criteria

Either CLOSE with reproducible NO_RELEVANT_GRAPH/NO_EXPOSED_GAIN, or opt-in SSM-only source branch preserves exact conv_state CPY consumers, multi-request/replay semantics, logits/greedy/MTP, and >=3% whole-lane gain with <=1% control regression. Native MoE rpb2 and generic small_k receive no extra patch, candidate or campaign.

## Notes

Supersedes: RD27
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd27

2026-09-24 relevance at b11126: TODO. Sub-candidate 2 is materially cheaper than the item text implies: verified that calc_rows_per_block's small_k parameter and the HI09 nwarps_explicit/rows_per_block_explicit template mechanism already exist as validated core framework (patch 0600_mmvq_geometry); gfx1100's RDNA3_0 table_id does not consume small_k natively, so the explicit-geometry override path is required. GPT design request submission for this item hit a rejected/queue-saturated gateway (concurrency limit reached: 8 queued, 2 running gateway-wide) and was not resubmitted successfully in-session; plan authored directly from verified source.

2026-09-24 GPT review req_2b717df095b44703 applied: resolved sub-candidate 1's central design choice -- verified conv_input is explicitly GGML_OP_CONCAT at src/models/delta-net-base.cpp:472, and added the requirement to preserve its conv_state_last->conv_state_update CPY consumers, which the prior plan did not account for. Split sub-candidate 2 into its own separate package/candidate set with requires=["0600_mmvq_geometry"] (previously bundled and missing that dependency), with exact (type,ncols,K,nwarps,rpb=2) candidate rows to be enumerated at implementation time.

2026-11-22 owner handoff from decision-only BCOP112: Next gate / terminal outcome. Source-bounded action/stop rule: Use existing PRBE113 profiling when available; no duplicate queue. Prove actual CONCAT+CPY graph adjacency, state lifetime and non-overlapped E2E share >=2.9126%. Otherwise **close the fold without code**. A surviving default-off producer+CPY fusion must preserve exact state/graph/replay, full logits/greedy/MTP, and >=3% CI95-low E2E with <=1% controls across four sessions/architecture. No separate cache/allocator/scheduler. See docs/planning/completed/patching-bc-optimizations/BCOP112.md for the complete source/shape/negative case and evidence. PRBE21 is the only active implementation owner; this handoff does not qualify a kernel, change a source pin or authorize extra dispatch code.

2026-11-22 PRBE21 scope correction from source-reviewed BCOP139: multi-token MoE rpb2 is already native; this plan retains only the separately measured SSM conv_input path. Archived proof and exact b11552 decision: docs/evidence/2026-11-22-bcop-final-disposition/README.md

## 2026-10-19 Vulkan GDN transposed-CONCAT kernel audit (BCOP328)

**Decision:** PRBE21 remains the single owner of the GDN conv_input producer. Qualify an upstream Vulkan operator-kernel improvement, distinct from this item's HIP CONCAT->SSM_CONV fusion and its retired MoE rpb2 half. RRVP02's Vulkan implementation pause forbids local shader changes and hardware queues. No new technical plan, dispatcher, allocator, telemetry or configuration.

**Eligibility:** Independently active feat/rad13-fp8-worker at Audumla 45c6edc01e14 (2026-10-19 14:08:39 UTC), newer than main f792f51e. PRBE21 last independent edit Oct 11 13:13 UTC; PRBE18 Oct 11 03:05 UTC. No repository-visible matching 12-hour patch, plan, PR, benchmark or queued experiment. Exclude RAD13 FP8/PR #330, RCS09/RCS05, RAD11/12/10, WSL and release work; off-repository queues unverified. Novelty: transposed GDN CONCAT Vulkan shader, not recently audited PRVP04 Q4 KV transpose, PRVP05 RMSNorm, RNX08 Q8 writes or PRBE41 SSM layout.

### Exact source and dispatch

Pinned llama.cpp b11552 (23b0202a189c44a54625aadcb37a946dd1d6278d) src/models/delta-net-base.cpp::llm_build_delta_net_base::build_conv_state: conv_states = reshape_3d(build_rs(...), k-1, C, n_seqs); qkv_mixed = ggml_transpose(qkv_mixed); conv_input = ggml_concat(conv_states,qkv_mixed,0). For one sequence: A=[k-1,C], B=[T,C] transposed view with nb10=C*type_size, nb11=type_size; D=[k-1+T,C]. D also feeds conv_state_last -> conv_state_update CPY, including MTP rollback. Preserve every output, state update, buffer lifetime, alias and stream dependency.

ggml/src/ggml-vulkan/ggml-vulkan.cpp::ggml_vk_concat passes unit-based ne/nb push constants to ggml_vk_op_f32; ggml_vk_op_get_pipeline selects pipeline_concat_i32 for F32. Pinned vulkan-shaders/concat.comp is a flattened 512-thread shader: contiguous destination stores, but adjacent B columns load from C*type_size-strided addresses. This is a source-level transaction-coalescing hypothesis, not a measured BigCherry bottleneck.

[llama.cpp #30591](https://github.com/ggml-org/llama.cpp/pull/30591/files) adds vulkan-shaders/concat_transpose.comp, three i8/i16/i32 pipelines in ggml-vulkan.cpp, ggml-vulkan-types.h and vulkan-shaders-gen.cpp. Eligibility guard: dim0, nonquantized 1/2/4-byte units, src0/src1 ne[2]=ne[3]=1, matching rows, src0/dst nb[0]==type_size, src1 nb[0]>nb[1]. Multi-sequence n_seqs>1 and other shapes use the original shader. Shader tile 32x32, local_size=(32,8,1) = 256 threads, four 8-row loops, one workgroup barrier, shared tile[32][33] (F32 4224 B; F16 2112 B; i8 1056 B). Grid ceil(C/32) x ceil((k-1+T)/32). B reads coalesced along channels through LDS, D writes coalesced, A read directly. Check maxComputeWorkGroupCount[1] before selecting; overflow must fall back, never truncate. Actual wave32/64, SPIR-V VGPR/spills, LDS bank conflicts, occupancy and physical bandwidth are unverified.

For C=10240,T=2048,k-1=3,F32,n_seqs=1: 80.1172 MiB logical reads and 80.1172 MiB writes, 160.2344 MiB total, 20,800 output tiles. This is not measured HBM traffic and no logical work is eliminated. Faster because SSM or state CPY disappeared is a correctness failure.

### Evidence, alternatives, and bounded admission

#30591 OPEN, updated Oct 19 13:32 UTC. External R9700 Qwen3.6-35B-A3B Q4_K_M pp2048/ub2048 4940.6 -> 5723.6 tok/s (+15.8%); R9700 Qwen3.8-27B 1287.1 -> 1336.7 (+3.9%); gfx1151 Flash-Next 403.9 -> 552.1 (+36.7%). Three-repeat same-quant external data, not BigCherry or CI95. Decode approximately neutral. Competing [#30149](https://github.com/ggml-org/llama.cpp/pull/30149) uses two ggml_vk_cpy launches and existing copy_transpose, no new shader: W7800/gfx1100 35B Q6_K pp2048/ub2048 3518.7 -> 3784.5 (+7.55%). In #30591's same-device R9700 comparison 35B: 5705.05 versus #30149 5591.43; 27B: 1352.02 versus 1349.81. These are alternatives, not additive. [AMD HIP tiled transpose](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/transpose) demonstrates the mechanism; SGLang's [concat_and_cast_mha_k_kernel](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/kvcache/cache_ops.py) has a different KV-cache layout, not a direct port.

Host-only oracle executed: 1140 contiguous + 120 padded-stride byte-exact address/coverage cases; 8137 modeled tile barriers; six selector/fallback cases; py_compile passed. Not SPIR-V compilation, GPU testing or benchmark. No matching first-party BigCherry Vulkan CONCAT receipt found in inspected plan/patch/commit evidence.

Gate 0: keep PAUSED until RRVP02 resumes; check #30591/#30149 merge and pin absorption first. Gate 1: prove actual gfx1100/gfx1201 Vulkan route with ICD/driver, n_seqs, T/C, ne/nb, dtype, pipeline, workgroup count and GPU timestamp for pp512/2048/4096, ubatch 128/512/2048/4096, tg128, MTP width 1/3/5, long context and multi-rank. No route -> NO_VK_CONCAT_ROUTE. If eliminating all exposed CONCAT latency could not reach 3% throughput (exposed fraction <2.9126%), close BELOW_CEILING. For op speedup S and exposed fraction f, model throughput factor = 1/(1-f+f/S).

Gate 2: compare pinned flattened, #30591 one-launch tile and #30149 two-launch reuse on identical pin and workload. Direct test-backend-ops -o CONCAT: F32/F16/i8 where supported, A widths 0/1/3/31/32/33, B widths 1/31/32/33, channels 31/32/33/8192/10240, noncontiguous strides, transposed A, n_seqs>1, quantized/dim!=0 controls, workgroup-limit fallback and canaries. Verify complete SSM_CONV/conv_state_update state, multi-request same-process, multi-ubatch, graph replay, long-context, logits/KLD/greedy and MTP accept/reject. Profile actual transactions/L2, wave/LDS/VGPR/spills and launches; no physical-traffic claims from logical bytes.

Gate 3: >=10 paired ABBA rounds per device/model, CI95-low >=3% E2E throughput, <=1% material control regression, full correctness; otherwise NUMERIC_FAIL, REGRESSION, NO_GAIN or UNSUPPORTED_SHAPE. Keep original shader fallback. PRBE18 owns pre-scan, PRBE41 SSM layout, RNX03 recurrence, PRVP02 CM1, PRVP05 RMSNorm, RRVP02 Vulkan pause. No neighbouring plan or implementation changed; BCOP328 is a thin action ledger.

## 2026-10-11 b11552 audit — small-K MoE rpb=2 is already native; retire duplicate candidate

**Disposition (small-K MoE half only): close the proposed MoE rpb=2 implementation/candidate addition as redundant.** Do not change the separate SSM conv_input/CONCAT proposal in this run: PRBE18 received independent planning work at 2026-10-11 03:05 UTC and its fusion path is protected by the 12-hour rule. The older two-candidate instructions below are historical, superseded for the MoE rpb=2 half only. No patch, benchmark queue, or GPU gain is claimed.

### Pinned implementation and routing proof

- At llama.cpp **b11552** `ggml/src/ggml-cuda/mmvq.cu::mul_mat_vec_q_switch_ncols_dst`, **`has_ids && ncols_dst > 1` returns early** through `mul_mat_vec_q_moe_launch<type>`. This route never reaches the generic `switch (ncols_dst)`, `calc_rows_per_block`, or `mul_mat_vec_q<...small_k...>` geometry path.
- `mul_mat_vec_q_moe_launch` already hardcodes **`constexpr int rows_per_block = 2`**, grid `ceil(nrows_x / 2) × expert_channels` and block `(warp_size, ncols_dst)`; both fused/unfused `mul_mat_vec_q_moe<type, 2, ...>` specializations consume it. This is an existing **upstream implementation**, not a new BigCherry result. Changing the generic rows-per-block selector does **nothing** for multi-token `MUL_MAT_ID` in this route.
- For **`has_ids && ncols_dst == 1`**, the ordinary `mul_mat_vec_q` path remains reachable and selects expert IDs per channel. Upstream `should_use_small_k` explicitly disables the automatic small-K heuristic on RDNA, and RDNA3/RDNA4 parameter tables do not apply the GENERIC/GCN `small_k ? nwarps : 1` row rule. An explicit single-token geometry *could* differ, but it is a **different, unmeasured** candidate, not an absent multi-token MoE rpb2 implementation.
- BigCherry `tools/bigcherry/tuning/catalog.py::enumerate_mmvq` **already emits** `rows_per_block=2` among `MMVQ_ROWS_PER_BLOCK` and `small_k=True` variants. `0600_mmvq_geometry/patch.py` extends **only** the generic `mul_mat_vec_q` template and generated launch bridge; it does not change `mul_mat_vec_q_moe`. Therefore the old instructions to add rpb2 catalog rows or a new MoE patch are obsolete. Do not duplicate existing candidates, dispatch keys, selectors or compiled variants.

### Ownership, external references and cheapest terminal gate

- **PRBE21** owns only this historical small-K MoE disposition; **HI09/0600 and the existing catalog** own generic MMVQ geometry. The dedicated upstream MoE kernel owns multi-token rpb2. **PRBE18** owns overlapping SSM/CONCAT fusion analysis and is not edited; PRBE111/1273 owns IQ MMVQ instruction/geometry tuning. Active QFP48 MMQ and QFP49 scheduling work are untouched.
- Current upstream `master` retains the same MoE early-return and hardcoded rpb2. The external [P100 patch 07](https://github.com/shinbunbun/llama-cpp-p100-patches/blob/main/docs/patches.md) tests **rpb4** instead of 2 and reports +1.9% MoE decode on **sm_60**; it is a different GPU/ISA and is **not** evidence for gfx1100/gfx1201. [vLLM's gfx1100 MoE GPTQ kernel](https://github.com/vllm-project/vllm/blob/main/csrc/rocm/moe_q_gemm_rdna3.cu) uses routing-sorted tokens and GPTQ W4A16; neither its format nor tile contract is a direct GGUF MMVQ replacement. **Reject both ports without typed, same-workload AMD evidence.**
- **Cheap source/host discriminator completed:** 16 static source/catalog/adapter checks (one initial broad-text ordering check corrected to function scope) and six routing cases. The final scoped early-return assertion passed. No HIP compilation, repository pytest, real model, GPU benchmark or new measured speedup.
- **No hardware queue for the already-native rpb2.** If a future, independently eligible single-token (`has_ids && ncols_dst == 1`) case is demonstrated by existing trace evidence, first record actual quant type, K, expert count, physical GPU, kernel identity, `nwarps`, `rpb`, occupancy, duration and end-to-end exposed fraction; compare existing catalog rpb1/rpb2 under **HI09** rather than adding code. Require full-vocab logits/KLD, greedy/MTP acceptance, same-process repeated requests, multi-ubatch and graph replay. Reject if best-case removable E2E latency <2.9126%; otherwise four independent sessions/architecture, >=10 paired ABBA rounds, CI95-low >=3% E2E and <=1% controls. No current lane is authorized, and there is no proof that such a candidate is beneficial.

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
- 2026-11-22T05:15:48.085146+00:00 (updated-by): Updated: section:notes
- 2026-11-22T06:45:45.856025+00:00 (updated-by): Updated: section:title, section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria, section:notes
- 2026-11-22T06:52:29.725407+00:00 (updated-by): Updated: section:code_samples, section:effort_risk
