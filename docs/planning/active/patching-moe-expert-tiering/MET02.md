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

## Files

Planned patch root: `patches/1281_moe_mul_mat_id_range/` only after Phase A reference design is validated. Composed-tree targets are `ggml/include/ggml.h`, `ggml/src/ggml.c`, `ggml/src/ggml-cpu/ops.cpp`, current `ggml/src/ggml-cuda/` MMID/MMQ/MMVQ grouping helpers, existing HIP-autotune keys, and backend-op/tier integration tests.

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

## Change Log

- 2026-10-02T04:44:50.160922+00:00 (created-by): Created by agent
- 2026-10-05: BCOP15 audit backfill added compact-dispatch gate.
- 2026-10-05: Transplanted structured sparse-range dispatch, MMQ safety, precision, validation and ownership details from `automation-qfp-indexer-20261004`.
- 2026-10-05: Deep audit made implementation sequencing semantic-first, added exact ownership/lifetime/work-accounting gates and bounded compaction against current upstream/fork mechanisms.
