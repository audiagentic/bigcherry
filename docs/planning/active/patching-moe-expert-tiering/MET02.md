---
id: MET02
order: 2
plan: patching-moe-expert-tiering
state: pending
created-at: '2026-10-02T04:44:50.160922+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# 1281 moe_mul_mat_id_range: range-aware MUL_MAT_ID primitive (CPU + HIP)

## Description

Generic ggml primitive: `ggml_mul_mat_id_range(ctx, weights[K,M,n_local], act, ids, id_base)`. For each selected global expert id `g`, compute `local = g - id_base`; if `0 <= local < ne02`, execute the local expert, otherwise emit an exact zero lane and skip GEMM/GEMV. Implement this as an op-param variant of `GGML_OP_MUL_MAT_ID`; plain `ggml_mul_mat_id` retains strict `0 <= id < ne02` semantics.

This item owns range semantics and the optional compact active-lane representation only. MET01 owns placement/profile policy, MET03 owns tier graph/CPU-tail execution, MET04 owns cache-vs-EP policy, MET05 owns auxiliary-device execution, and MET06 owns compact expert materialisation. Do not create a second router, cache, placement table, expert store or scheduler.

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

Planned patch root: `patches/1281_moe_mul_mat_id_range/` only after Phase A reference design is validated. Composed-tree targets are `ggml/include/ggml.h`, `ggml/src/ggml.c`, `ggml/src/ggml-cpu/ops.cpp`, current `ggml/src/ggml-cuda/` MMID/MMQ/MMVQ grouping helpers, existing HIP-autotune keys, and backend-op/tier integration tests.

## Validation



## Effort & Risk



## Standards



## Acceptance Criteria

- CPU/reference Phase A exists and passes before GPU optimisation work begins.
- Range semantics pass the full correctness matrix on CPU, gfx1100 and gfx1201; gfx1030 has reference/correctness coverage.
- Ordinary `MUL_MAT_ID` has no material pp/tg regression.
- No OOB/guard failure on partial MMQ tiles; native MMQ J/scratch ownership is reused.
- Sparse compaction satisfies the promotion gate with expected-versus-observed active-work accounting.
- HIP autotune owns any sparse/dense crossover; no second dispatch registry, router, cache, placement layer or scheduler is introduced.

## Notes

2026-10-05: implementation audit found 1281 is not yet present on the live branch. Converted the item to a three-phase semantic-first plan and bounded compact dispatch behind correctness/work-accounting gates. External grouped-expert implementations are mechanism evidence only; no CUDA/NVLink transport assumptions transfer to BigCherry AMD hardware.

References: https://github.com/ggml-org/llama.cpp/issues/21948 ; https://github.com/ggml-org/llama.cpp/issues/27792 ; https://github.com/ggml-org/llama.cpp/pull/29911 ; https://github.com/ggml-org/llama.cpp/pull/29963 ; vLLM MoonEP/DeepEP grouped-expert paths.

2026-10-06 PHASE A AUTHORED (owner direction: expert parallelism is a worthy goal for higher-quant testing, so the MET02 -> MET03 -> MET04 chain is started). patches/1281_moe_mul_mat_id_range (state untested, experiment moe-range). ggml_mul_mat_id_range(ctx, as, b, ids, id_base) = ggml_mul_mat_id plus op params i32[6] = marker 0x52414E47 and i32[7] = id_base (0..3 are used by precision / hints); accessors ggml_mul_mat_id_is_range / ggml_mul_mat_id_range_base. CPU: in the row grouping of ggml_compute_forward_mul_mat_id the id is widened to int64, the base subtracted, out-of-range ids skipped before any grouping or weight indexing; the whole dst is cleared on thread 0 before the barrier so an inactive lane is an exact +0. The ordinary op keeps its assertion and its path. HIP/CUDA supports_op refuses the variant (phase A: no global id may reach a GPU kernel). New test program tests/test-mul-mat-id-range.cpp (built by llama_build_and_test): bit-for-bit comparison with the ordinary op on hand-translated ids and +0 on inactive lanes, for id_base {0, 1, 7, INT32_MAX-3}, n_local {1, 2, 4}, all-active / all-inactive / mixed-with-boundaries, 1 / 3 / 40 tokens, 1 / 4 threads, F32 and Q8_0. n_local = 0 is not representable as a tensor and is not tested. Offline mechanics tests pass; applies on the pin and on the production-patched tree. GPT REVIEW req_61670b6161b94b09 (focused, five questions): op params 6/7 free at b11402 and preserved by the generic op_params memcpy - yes; memset before the barrier race-free - yes; supports_op refusal sufficient to keep it off the GPU (assignment, offload and fusion all pass through supports_op) - yes; exact +0 by memset appropriate - yes; OTHER ID READERS - two real hits: (a) the scheduler's selective expert copy asserts id < n_expert (in this tree that code is llama_context::sched_copy_experts once 1336 is applied) - not reachable in phase A because the op never lands on a GPU split, MUST be translated in phase B; (b) ggml-cpu/repack.cpp tensor_traits::forward_mul_mat_id groups rows from the ids itself for weights repacked at load - FIXED in the patch with the same translation and pre-barrier clear (edit mmid-range-cpu-repack). Limitation: the standalone test builds its weights in a plain context, so it exercises the generic CPU path, not the repacked one; a repack-buffer case is to be added with phase B. Build + reference test queued on Brutus (queue-moe-range.sh b11402j; that build predates the repack edit and will be rebuilt).

2026-10-06 PHASE A PASSED ON HARDWARE HOST; PHASE B CODE DRAFTED WITH GPT. (1) tests/test-mul-mat-id-range on Brutus (CPU): 432 cases, 0 failed, on build b-moerange-b11402j and again on b-moerange-b11402k (the rebuild with the repack-path translation) - bit-exact equality with the ordinary op on hand-translated ids and exact +0 on inactive lanes for every base / n_local / pattern / token count / thread count, F32 and Q8_0. Phase A promotion criterion (exact reference equivalence, ordinary op unchanged) is met for the generic CPU path; the repacked-weights path has the same code but no dedicated test case yet. (2) Phase B drafts from GPT (dev-gpt-agent, verified against d89651a7b205 by its own statement, NOT build-tested): req_e116405983eb4a28 - MMVQ / MMVF: ids are read on the device at b11402 (not host-resolved); add id_base / id_count to ggml_cuda_mm_fusion_args_device, translate where mul_mat_vec_q, mul_mat_vec_q_moe and mul_mat_vec_f read the id, an out-of-range lane writes 0.0f for its rows and returns before touching src0; id_count == 0 selects today's path; caller passes base/count from ggml_cuda_mul_mat_id. req_153666dc4a4646f8 - MMQ / MMF / host-sorted fallback: translate inside ggml_cuda_launch_mm_ids_helper (mm_ids_helper maps an out-of-range id to INT_MAX so it is never counted or gathered); MMQ pre-fills ids_src1 with -1 and the three quantize kernels skip negative ids; dst is cleared with cudaMemsetAsync before the scatter; dedup_bcast is disabled for range ops; the fallback gathers / computes / scatters only the active rows through a small scatter kernel; existing J / tail / scratch sizing reused (n_tokens * n_expert_used stays a safe upper bound). Outstanding for phase B before any of it is trusted: integrate both drafts as anchored edits on the pin AND on the production-patched tree (1307-1313 Q8_1 fusions and 1334 edit the same files), remove the phase-A supports_op refusal, disable the MUL_MAT_ID + GLU fusion for range nodes (or make it range-aware), translate ids in llama_context::sched_copy_experts (1336), add range cases to test-backend-ops so GPU vs CPU is compared, and a repack-buffer case.

2026-10-06 PHASE B: GPU RANGE OP PASSES THE STANDALONE TEST ON ALL FOUR DEVICES (build b-moerange-b11402o = production + 1281; tests/test-mul-mat-id-range --gpu, the op computed directly on each device against the CPU ordinary op with hand-translated ids; NMSE < 5e-4 on held lanes, exact +0 on the others; Q8_0 / Q4_K / IQ4_XS / F32 / F16, id_base 0 and 5, n_local 1 / 4 / 12, 4 and 10 experts per token, distinct ids per token). Results: 1 token 720 / 720 ok; 2-4 tokens 1440 / 1440 ok; 8-9 tokens 1152 ok + 288 declined; 33 tokens 432 ok + 288 declined; 300 tokens 432 ok + 288 declined; 0 failures, no crash. 'Declined' = the device's supports_op refuses the range op (float weights on the large-batch MMF path, not converted by design - the scheduler keeps those on the CPU). CPU reference test still 432 / 432. IMPLEMENTATION AS BUILT: (1) MMVQ / MMVF kernels (mul_mat_vec_q, mul_mat_vec_q_moe, mul_mat_vec_f) translate the global id through two new fields of ggml_cuda_mm_fusion_args_device (id_base, id_count; 0 = ordinary) and return early for a lane that is not held; the host clears dst on the stream first. (2) MMQ: a small kernel translates the ids on the device into a scratch buffer (local index, INT_MAX when not held) that is handed to the unchanged ggml_cuda_launch_mm_ids_helper; ids_src1 and dst are cleared first; the token-dedup quantizer is not used for range ops. (3) supports_op accepts a range op only where the dispatch takes MMVQ, MMVF or MMQ (bc_cuda_mul_mat_id_range_supported mirrors the dispatch order). (4) ggml_cuda_should_fuse_mul_mat refuses range nodes (v1; decode loses the fused gate + up + GLU kernel). FIRST GPU RUN WAS A TEST DEFECT, NOT A PATCH DEFECT (kept for the record): with ids drawn at random the same expert could appear twice in one token; the upstream MMQ grouping records one row per (token, expert) and drops duplicates, so every large-batch case with a held lane failed with NMSE = 1 - 1/n_used (0.75 at 4 experts, 0.90 at 10) and one run crashed at the 8-9 token boundary. A control arm (the ordinary op on the device) reproduced the same NMSE, proving the conversion was not the cause; GPT reached the same diagnosis independently from the source (req_89dd2d29f1054ae0; upstream's own test shuffles ids so they are unique) and cleared the pool lifetime, the ids_src1 clear, the zero filler and the dst clear. The test now draws distinct ids per token, as a top-k router does. GPT review of the kernel edits (req_f0dac9a3c42f4d88): all three early returns are safe (block- or warp-uniform predicates). OPEN for phase B: range-aware fusion (GPT site list req_3b5bb3c3e4834bc8: seven fused launches at b11402; carry base / count in ggml_cuda_mm_fusion_args_host, clear the fused dst, keep ADD_ID-bias variants excluded because a bias makes an inactive lane non-zero, and scaled variants strictly excluded because a negative scale gives -0); llama_context::sched_copy_experts (1336) still assumes local ids; a repack-buffer case for the CPU repacked path; duplicate ids per token remain unsupported on the large-batch GPU path exactly as for the ordinary op.

2026-10-06 - after the range-aware fusion edits and the guard-collision fix (65ffcaea), build b-moerange-b11402s: CPU 432/432; GPU bands 1 / 2-4 / 8-9 / 33 / 300 tokens: 720 / 1440 / 1152 (+288 declined) / 432 (+288) / 432 (+288) ok, 0 fail. The op test does not exercise the fused gate+up+GLU launches; those are covered end to end by MET04 (fidelity 21/24, TV 0.113 and 0.078, decode equal to the row split). b-moerange-b11402q failed on the guard collision and b-moerange-b11402r on a clang bus error (compiler crash, unrelated to the code).

## 2026-10-05 implementation audit

Repository fact: no `patches/1281_moe_mul_mat_id_range/` implementation exists on the current live branch; this is still a design item. Therefore the first deliverable must be a correctness-only CPU/reference implementation plus backend-op tests, not a speculative HIP compaction kernel.

Upstream fact: current `MUL_MAT_ID` safety/precision work is still moving. #29911 changes quantized `MUL_MAT_ID` precision eligibility, while the #27792/#29941/#29953 chain changes MMQ tail allocation/J ownership. The range op must consume the current upstream/native MMQ selector and scratch sizing; it must not copy those formulas into 1281.

External mechanism evidence: vLLM MoonEP consumes expert-grouped activations as contiguous per-expert segments and runs grouped GEMMs without an additional permute/unpermute. This supports a BigCherry compact representation of `(lane, local_expert)` grouped by local expert, but not its CUDA/NVLink transport. vLLM DeepEP also uses `-1` as an invalid/non-local expert sentinel before expert-map remapping. BigCherry should preserve its stronger exact-zero inactive-lane contract rather than remapping inactive lanes to a valid expert.

## Phase A — semantic primitive (required before optimisation)

Exact files/functions to change in the composed llama.cpp tree:

- `ggml/include/ggml.h`: declare `ggml_mul_mat_id_range`.
- `ggml/src/ggml.c`: construct `GGML_OP_MUL_MAT_ID` with an op-param version/flag and signed `id_base`; ordinary constructor remains byte-for-byte behaviourally unchanged.
- `ggml/src/ggml-cpu/ops.cpp`: in the existing `MUL_MAT_ID` execution path, translate only when the range flag is present; inactive lanes write exact zero.
- backend op tests: add full-range equivalence and mixed/out-of-range cases before HIP work.

Reference semantics:

```text
for token,row,slot:
    g = ids[token,slot]
    l = int64(g) - int64(id_base)        # widen before subtract: no signed overflow
    if 0 <= l < n_local:
        out[lane] = native_mul_mat_id(weights, x, l)
    else:
        out[lane] = +0 exactly
```

Ownership/lifetime: `id_base` is immutable op metadata. No translated-ID tensor is materialised in Phase A. CPU and GPU implementations must derive local IDs from the original IDs in backend-owned execution; there is no H2D/D2H round trip and no per-token allocation.

### Phase-A correctness gate

A cheap host mock/reference test must enumerate boundaries without a GPU: `id_base` in `{0,1,INT32_MAX-3}`, `n_local` in `{0,1,2,4}`, and IDs `{id_base-1,id_base,id_base+n_local-1,id_base+n_local}` where representable. Compare range output against a reference assembled from ordinary per-expert matmul plus exact-zero inactive lanes. Include all-inactive, all-active, duplicate expert, unsorted expert and zero-token cases. Build CPU-only first. Do not start HIP compaction until this passes.

## Phase B — HIP correctness-only range path

Use the existing `MUL_MAT_ID` grouping/sort and MMVQ/MMQ paths. Add range translation at the earliest point that already owns expert-ID interpretation; do not allocate a second routing list. For native grouped metadata, represent non-local entries as inactive and prevent them from reaching expert weight indexing. Zero inactive destination lanes once.

Required source audit before implementation: identify the current pin's exact expert-ID grouping/sort entry points in `ggml/src/ggml-cuda/mmq.cu`, `mmq.cuh`, `mmvq.cu`/`mmvq.cuh` and related `mmid` helpers. Record those symbols in the patch README because names move upstream. Apply range arithmetic once at that ownership boundary rather than separately in every quant kernel.

Safety invariants:

1. No inactive/global ID may be used to index `src0` expert weights.
2. Ordinary `MUL_MAT_ID` takes the identical dispatch path when the range flag is absent.
3. `GGML_PREC_F32` remains available for quantized weights when native upstream allows it (#29911).
4. MMQ scratch/tail/J sizing comes from the current native selector (#29941/#29953-shaped ownership); 1281 adds no J table/formula.
5. No device-wide synchronization is introduced.

## Phase C — compact sparse dispatch (only after Phase B attribution)

The optimisation target is not merely skipping an `if`: at low local ownership, avoid launching/iterating inactive expert work. Reuse existing sorted/grouped expert metadata to emit a compact list grouped by local expert:

```text
entry = { output_lane, local_expert }
active_count = compact(existing_grouped_ids, id_base, n_local, entry)
zero_inactive_outputs_once()
if active_count != 0:
    native_grouped_mmvq_or_mmq(entry, active_count)
scatter only if native grouped output is not already lane-addressable
```

Prefer a representation where output lane remains attached to each grouped row so no separate dense permutation buffer is required. If current native grouping already carries destination row indices, extend/reuse it. Workspace comes from existing backend scratch and is sized to the maximum selected lanes for the graph; no `hipMalloc`, host vector or host synchronization per token/ubatch.

The theoretical removable work fraction is bounded by `1 - local_density`; at 10% ownership at most 90% of expert-row compute/iteration can disappear, but list construction, zeroing and fixed launch costs remain. This is an upper bound, not an expected speedup.

Profitability gate:

`T_compact = T_filter/group + T_zero + T_active_compute + T_optional_scatter`

versus

`T_dense = T_native_group + T_inactive_checks + T_active_compute`.

Do not compact at 100% density. Start with a static experiment at <=25% density; only after measurements should HIP-autotune own a threshold keyed by architecture, quant type, ubatch bucket, density and MMVQ/MMQ regime.

## Validation and benchmark matrix

Correctness before timing:

- ordinary `MUL_MAT_ID` unchanged;
- full-range range-op equals ordinary op;
- 0%/mixed/100% local ownership;
- first/last local expert and both adjacent out-of-range IDs;
- duplicate/unsorted IDs;
- partial final MMQ tile;
- default and `GGML_PREC_F32`;
- F32/Q8_0/Q6_K/IQ4_XS where supported;
- CPU, gfx1100 and gfx1201 required; gfx1030 reference/correctness coverage;
- multi-ubatch and two consecutive requests in one process to catch stale workspace/grouping state;
- Flash-Next tier-graph greedy/logit/KLD comparison once MET03 integrates it.

Performance: gfx1100/gfx1201; ub1/4/16/32/64/128/256/512; local ownership 0/10/25/50/75/100%; MMVQ/MMQ; pp512/pp2048 and tg128/tg512. Record selected lanes, active lanes, density, grouping/filter/zero/scatter/kernel microseconds, kernel count, workspace bytes, H2D/D2H bytes, VGPRs/spills and end-to-end tok/s. Compare ordinary baseline, range-dense, range+compact at identical model/placement.

Physical plausibility: measured active expert rows must equal routing-derived expected active rows; bytes/work may fall only in proportion to skipped non-local rows. A speedup accompanied by missing active rows, changed logits outside tolerance or unexplained transfer disappearance is a correctness failure.

## Promotion / rejection

- Phase A promotes only with exact reference equivalence and unchanged ordinary op tests.
- Phase B promotes only with CPU/HIP semantic parity and no ordinary-op regression >1% in repeated primary lanes.
- Phase C promotes for a declared density/batch regime only if a representative <=25% local-density lane improves >=5% end-to-end or >=10% `MUL_MAT_ID` kernel time, with no required dense lane >2% slower and no correctness/transfer-accounting anomaly.
- Reject/stop compact dispatch if filter+zero+scatter consumes >=80% of the compute time it removes in two representative sparse lanes, or if the native grouping cannot retain destination-lane identity without another dense materialisation. Keep correctness-only range semantics.
- Do not add a permanent architecture/density table here; successful crossover data is handed to HIP-autotune.

## Dependencies and consolidation

MET01 supplies placement/range ownership. MET03 consumes this primitive for tier graphs. MET04/MET05/MET06 must not introduce alternative range semantics. #29963's host-RAM pipeline parallelism is a separate whole-layer/scheduler mechanism and does not replace expert-range semantics; if it wins the target workload without expert tiering, MET03 may become unnecessary, but 1281 should not absorb its scheduler/event machinery.

## Change Log

- 2026-10-02T04:44:50.160922+00:00 (created-by): Created by agent
- 2026-10-05: BCOP15 audit backfill added compact-dispatch gate.
- 2026-10-05: Transplanted structured sparse-range dispatch, MMQ safety, precision, validation and ownership details from `automation-qfp-indexer-20261004`.
- 2026-10-05: Deep audit made implementation sequencing semantic-first, added exact ownership/lifetime/work-accounting gates and bounded compaction against current upstream/fork mechanisms.
- 2026-10-06T01:25:53.938882+00:00 (updated-by): Updated: section:notes
- 2026-10-06T02:50:06.326771+00:00 (updated-by): Updated: section:notes

## Ledger-events

- chg_20261006_025038_groundwork-for-running-moe-mod_5459
- 2026-10-06T02:50:45.468950+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-06T04:14:18.031411+00:00 (updated-by): Updated: section:notes
- chg_20261006_061542_experimental-expert-parallel-t_6967
- 2026-10-06T06:15:46.589711+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-06T07:14:51.471850+00:00 (updated-by): Updated: section:notes
