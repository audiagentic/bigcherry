---
id: QFP04
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-03T15:20:34.367684+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# 1295/1296 QSA gather-based sparse decode attention (gather in stored KV type, dequant in FA)


## 2026-10-18 b11552 kernel audit: F16 QSA decode GET_ROWS-to-CAST staging (BCOP288)

**Disposition: receipt-first; no implementation or GPU job authorized.** Narrow residual: validated 1295 emits GET_ROWS into F32 and CAST to F16 separately for K and V even when source KV is F16. The original 1296 'typed gather' cannot be achieved by changing a flag on the existing ggml_get_rows constructor: pinned ggml/src/ggml.c fixes non-I32 GET_ROWS output to F32. Q8_0 packed-copy and dequant-on-load are separate, unadmitted routes; do not infer their performance from this F16 audit.

**Exact source/dispatch:** engines/llamacpp/patches/1295_qsa_gather_decode/patch.py::_GATHER enters only with bc_qsa_idx, n_tokens <= 8, n_kv >= BIGCHERRY_QSA_GATHER_MIN (32768 default), contiguous single-stream K/V and a V-stride guard. It pads selected indices per query to 256, uses sentinel n_kv to fetch a -infinity mask column, clamps K/V fetch indices to n_kv-1, then computes F32 GET_ROWS -> F16 CAST for each K and V, permutes into FLASH_ATTN_EXT and preserves self_v_rot. QFP03/1294 owns the deterministic selected-index order. Pinned ggml/src/ggml-cuda/getrows.cu::get_rows_cuda already has destination-F16 template instantiations, including F16 source, but graph construction never requests that output. ggml/src/ggml-cpu/ops.cpp::ggml_compute_forward_get_rows_f16 writes F32, so globally changing the tensor dtype without a CPU fallback is unsafe. ggml/src/ggml-backend-meta.cpp::handle_get_rows preserves split axis 0 only with mirrored indices, otherwise applies scalar-only generic split handling. Verify physical rank, graph allocation, stream, strides and consumer count on the composed recipe before reusing this backend primitive.

**Measured control, not incremental benefit:** validated 1295/README.md reports ordered-topk gathered versus masked 98K decode 65.4/65.7 versus 56.9/57.6 tokens/s (~14%); earlier 196K screen ~13%; 24K neutral. Those gains already belong to 1295. Its old dense attention cost increased 0.44 to 3.36 ms/token from 10K to 80K in summed kernel observations; this is not a measurement of removable F32 staging.

**Theoretical byte ceiling:** per selected K or V scalar, F16 read 2 B + F32 write 4 B + F32 CAST read 4 B + F16 write 2 B = 12 logical B. Direct F16 gather would read 2 and write 2 = 4 B, at most 8 B avoidable per scalar plus two CAST launches per eligible QSA layer. Illustrative D=256, one KV head, 2048 selected, four queries, K+V: 4,194,304 scalars, 32 MiB redundant logical traffic and 16 MiB F32 intermediates. These are neither measured DRAM bytes nor end-to-end gains; multiply only by actually executed layers/ranks and account for concurrent GPU intervals.

**Gate 0 (host/static, completed):** 19/19 source assertions against 1295, pinned ggml.c, CUDA/CPU GET_ROWS and FA dispatch; 4/4 NumPy host tests including 65,536 F16 bit patterns, 64 deterministic index/pad/sentinel cases, duplicate/tail and logical-byte accounting. Host equivalence is not HIP bit-identity or a real QSA workload.

**Gate 1 (no new code):** first inspect terminal receipts, then only on idle hardware capture the exact b11552 1295+1294 F16 decode path with existing rocprofv3 and tools/bigcherry/profiling/rocprof.py, tools/lab/flash-next/kernel-step-census.py, QFP16/17 timing. For widths 1..8 at 24K/98K/160K/196K (fit tiers), record physical GPU UUID/rank, per-layer GET_ROWS/CAST calls, duration and serial critical-path overlap, VGPR/scratch/spills, allocated F32 bytes, graph UID, actual K/V type/strides, MTP acceptance, output/selected-work counts and PCIe traffic. If F16 1295 is not executed, close NO_ROUTE. If optimistic *serialized* removable F32-staging+CAST time is <2.9126% of baseline decode latency (minimum for +3% throughput), close BELOW_CEILING. Do not queue a lane conflicting with active RAD06 experiments.

**Gate 2 (conditional single-owner implementation):** extend only QFP04/1295, default-off in an isolated build A/B. Reuse CUDA/HIP GET_ROWS destination-F16 kernel via a properly constructed typed GGML_OP_GET_ROWS result with valid F16 strides, not by mutating tensor->type or aliasing F32 storage. Prefer a narrowly scoped F16-source/F16-destination constructor with tested CPU F16-copy semantics and fail-closed other-backend/Meta admission. If this requires unsafe generic API or graph rewrites, close UNSUPPORTED_TYPED_GRAPH; do not add another index cache, selector, scheduler, allocator or global dispatch table. Preserve exactly the same bc_qsa_idx, order, sentinel, mask, GQA and FA output. Q8_0 remains excluded. Pseudocode:

~~~text
if 1295_admitted && K.type == F16 && V.type == F16
   && one_consumer_per_GET_ROWS_is_CAST_F16
   && typed_backend_supported && mirrored_indices
   && meta_split_and_graph_lifetime_safe:
    K_sel = typed_get_rows_f16(K_rows, rows_idx)
    V_sel = typed_get_rows_f16(V_rows, rows_idx)
else:
    K_sel = cast_f16(get_rows(K_rows, rows_idx))
    V_sel = cast_f16(get_rows(V_rows, rows_idx))
# preserve existing per-token mask and FLASH_ATTN_EXT
~~~

**Gate 3:** compare to *validated 1295*, not masked attention, with full logits/KLD and CPU-F32 reference, deterministic/greedy and MTP acceptance, widths 1..8, multi-request same process, multi-ubatch, 24K/98K/160K/196K, sentinel/duplicates/partial tail/NaN/Inf, Meta split 2xgfx1100+gfx1201, graph recapture and allocation checks, and expected-versus-observed selected work. Record physical bandwidth plausibility. Require >=4 independent sessions per architecture, >=10 paired ABBA rounds, CI95-low >=3% incremental end-to-end decode throughput at 98K/160K, <=2% shorter-context/prefill regression, no correctness or lifetime failure. Otherwise retain 1295 and close NO_GAIN or CORRECTNESS_FAIL.

**External mechanism and consolidation:** merged llama.cpp [#29639](https://github.com/ggml-org/llama.cpp/pull/29639) uses per-row sparse Vulkan indices, direct quantized K/V loads and an AMD-measured 16x quantized sparsity threshold (external Q8_0 +15.6% decode at 64K, +4.1% at 128K; not HIP/BigCherry). Its shader routes scalar quantized K loads through fa_kv_index. [vLLM AMD QSA](https://github.com/vllm-project/vllm/blob/main/vllm/models/qwen4_exp/amd/qsa.py) feeds packed selected indices directly to Triton sparse GQA but currently requires BF16 KV; not a drop-in GGUF F16/Q8 HIP port. QFP25/1334 owns sparse HIP *prefill* and eight-query union-capacity correctness; QFP03/1294 owns order; QFP17/1335 owns indexer; QFP04/1295 alone owns decode gather. No neighbouring protected plans, implementation or hardware queues changed.

**Terminal outcomes:** NO_ROUTE, BELOW_CEILING, UNSUPPORTED_TYPED_GRAPH, CORRECTNESS_FAIL, NO_GAIN, QUALIFIED_F16_ONLY.
## 2026-10-15 1295 activation witness correction (BCOP237)
**Source finding, not a reproduced GPU bug:** `patch.py::_GATHER` evaluates `bc_qsa_gather_enabled()` before computing/checking `plain`. `_ENABLED` logs `BIGCHERRY_PATCH_HIT patch=1295_qsa_gather_decode` once on the first enabled call. An eligible non-plain graph can therefore log a false activation, while a later actual gather has no new marker. The existing `kqv_out_qsa_gather` callback names a selected graph, not executed HIP kernels. Historical ABBA/numerical results are not disproved, but the once-per-process marker cannot prove per-request execution.
**Ownership/action:** QFP04/1295 alone owns a minimal selected-route marker relocation/versioning and existing graph/profiler correlation to K/V GET_ROWS plus FA execution; QFP28 owns multi-request/cache/abort corruption qualification. QFP25/1334 owns sparse prefill union capacity. Do not introduce another gather dispatcher, cache, scheduler or telemetry registry. Negative controls: `n_tokens>8`, cache below threshold, QSA absent, switch off, multi-stream K/V and noncontiguous V. The `n_kv` sentinel is clamped for K/V and masked by appended `-inf`; it is not evidence of external DEVGATHER's uninitialized destination.
**Gate:** host route/sentinel fixture, then current-pin gfx1100/gfx1201 executed-route proof per request, full logits/KLD, graph reuse, multi-request tokens, MTP acceptance and bytes/time accounting. Retain QFP04's existing >=5% GPU-share or >=0.5 ms/token admission before any 1296/direct-index work. No new GPU measurement in BCOP237.

## 2026-10-14 sparse-prefill ownership correction

The historical proposal below to add a separate direct-index sparse prefill op is **not** an open QFP04 implementation request: validated 1334 already enables upstream sparse-mask FA on HIP for Qwen4Exp prefill. QFP25/1334 exclusively owns its eight-query union-capacity proof and any bounded fix; see QFP25's 2026-10-14 gate. QFP04 retains decode gather/typed-KV qualification only. Do not introduce a second selected-cell representation or count historical 1334 prefill gains as a new QFP04 result.

## Description

Historical v2 description (1295 is now validated/default-on): for n_tokens<=8 and n_kv>=BIGCHERRY_QSA_GATHER_MIN (32768) build_attn_qsa gathers only the QSA-selected cells (bc_qsa_idx) via ggml_reshape_2d(k_all, hd*nh_kv, n_kv) + get_rows (f32) + cast f16, mask gathered with a -inf sentinel column, padded to 256, row indices clamped. At the b11402 screen, upstream sparse FA was NVIDIA-only; without later 1334 HIP QSA attention was dense. 1296 = next step: (1) gather rows in the cache's stored type (f16 or q8_0 byte gather) with no f32 round trip, one rows_idx/mask gather per layer; (2) tile FA dequant-on-load for q8_0 so no full-cache conversion.

Upstream llama.cpp PR #29825 is now the memory baseline for this item. It merged 2026-10-03 and changes Qwen4Exp/QSA indexer scoring from all-heads-at-once plus a separate ReLU copy to sequential-head scoring with in-place ReLU and accumulation. Published CUDA compute-buffer usage roughly halves without pp/tg loss (131K/ub2048 3.0 -> 1.5 GiB; 131K/ub4096 6.1 -> 3.0 GiB; 262K/ub4096 11.7 -> 5.6 GiB). BigCherry must rebase this upstream change before attributing any remaining long-context OOM or workspace cost to QSA gather/FA.

## Steps

1. Rebase/confirm upstream #29825 at the current pin first; capture AMD gfx1100/gfx1201 compute-buffer high-water marks at 131K/262K and ub2048/4096. Do not carry a local equivalent if the upstream implementation applies cleanly.
2. Qualify upstream PR #29901's 4-head tiled lightning indexer independently on HIP before touching selected-cell attention. Measure indexer kernel time and whole PP at 64K/128K/240K with ub512/1024/2048 where memory permits. Keep decode as a negative control because #29901 deliberately retains the vector kernel below a token tile.
3. Read v2 screens (80K/160K) and profile the post-#29825/#29901 decode critical path. QFP04 work proceeds only if selected-cell attention/gather remains >=5% of decode GPU time or >=0.5 ms/token at >=160K.
4. Implement 1296 step 1 (typed gather) from GPT code; numerics bit-identical to 1295 for f16 caches.
5. Before step 2, prototype a direct selected-index sparse-FA seam that consumes `bc_qsa_idx` without materializing compact K/V tensors. Compare it against typed gather. Prefer the direct seam if it removes >=1 large intermediate or >=3% decode wall at 160K/220K.
6. Only if q8 KV is again a production target and direct selected-index FA is not viable, implement FA tile q8_0 dequant-on-load. Do not maintain both a q8 typed-gather path and a direct selected-index path without independent wins.
7. Full ABBA on profile v2 at 80K/160K/220K, tracking acceptance separately from ms/step.
8. Treat the remaining dense `[n_kv,n_tokens]` QSA selection mask as part of the direct selected-index FA seam. Do not create a second score/mask-buffer optimization.

## Detailed Solution & Technical Design

Memory ownership is layered: upstream #29825 owns indexer score-buffer compaction; #29901 owns indexer compute tiling if it proves beneficial on HIP; QFP04 owns selected-cell attention/gather cost. The three should compose and must be benchmarked separately.

The preferred end-state is one canonical selection representation feeding attention directly:

```cpp
// selection/indexer owns canonical compact ids
bc_qsa_idx = qsa_select(...);

// preferred sparse decode seam: FA consumes selected KV ids directly
flash_attn_ext_selected(q, k_cache, v_cache,
                        bc_qsa_idx, n_selected,
                        kv_type, scale, ...);
```

The kernel should stage only selected K/V tiles, dequantize the stored KV type while loading when needed, and perform softmax over the selected set. It must not allocate `K_selected_f32 -> K_selected_f16` or a second dense mask. The existing 1295 GET_ROWS path remains the correctness/performance control.

This direction is supported by the independent SYCL sparse-FA Qwen4Exp experiment in llama.cpp discussion #28695: at 131K/q8_0 KV on Intel Arc B70 it reports roughly 3x long-context decode improvement. That result is not transferable performance evidence for AMD, but it is useful mechanism evidence that backend-native selected-cell FA can outperform graph-level gather plus dense attention. The HIP decision must be made from gfx1100/gfx1201 measurements.

Upstream PR #29901 provides a second useful separation-of-concerns example. For the 4-head Qwen4Exp indexer it stages 64 keys once and scores them against 8 tokens, retaining the vector kernel for smaller batches. On RTX PRO 6000 the PR reports indexer kernel time 16.4 -> 6.4 ms at kv=65536, nb=2048 (2.5x), PP 3867 -> 4038 t/s at 128K, indexer GPU share 9.6% -> 4.0%, with TG unchanged. This is prefill/indexer work, not evidence for changing QFP04 decode attention.

## Code Samples & Guidance

For memory qualification, record peak allocator/compute-buffer bytes beside throughput and context. A speed-neutral change that recovers `-b 4096` at deep context is still a valid prerequisite because it increases the feasible operating point.

For direct sparse FA, instrument bytes read from K/V, selected-cell count, gather/materialization bytes, kernel duration, and total attention segment time. A kernel microbenchmark win is insufficient if graph setup/materialization remains on the critical path.

## Files

Current Qwen4Exp/QSA model builder at the rebased llama.cpp pin; existing 1295/1296 patch sources; HIP Flash Attention extension only if the selected-index interface passes the gate. #29901 qualification should remain an upstream patch/control rather than be copied into QFP04.

## Validation

f32 CPU reference max|dp| <= masked path; ms/step ABBA at 80K and 160K; deep fill to ~227K without OOM. For #29825 baseline qualification: compare peak compute-buffer bytes before/after on one XTX and R9700, then the production tensor split; pp/tg must remain within noise and greedy/logit contract must match upstream expectations.

For #29901: test gfx1100 and gfx1201, PP at 64K/128K/240K, ub512/1024/2048 where feasible, plus TG128/TG512 negative controls. Record indexer kernel ms and GPU-time share. Adopt/track upstream only if whole-PP improves >=3% or indexer time falls >=20% without >1% TG regression.

For direct selected-index FA versus typed gather: f16 and q8_0 KV where supported; 80K/160K/220K; selected counts around operational min/median/max; zero OOB under sentinel/tail cases; logits/reference accuracy no worse than 1295; promotion requires >=3% decode wall or >=0.5 ms/token improvement at >=160K and no >2% regression below 80K.

## Effort & Risk

Medium-high. #29825 is merged upstream and should reduce local code by replacing any equivalent memory workaround. #29901 should be qualified as an upstream prefill optimisation, not folded into QFP04 code. QFP04's sparse-attention changes remain numerically sensitive and require explicit acceptance tracking.

## Standards

Upstream-first; one owner per buffer class; memory-attribution before new allocation work; no duplicate QSA score path; one selected-cell representation; separate prefill indexer and decode-attention claims.

## Acceptance Criteria

- Current BigCherry pin contains or cleanly rebases #29825 and its AMD memory effect is measured.
- #29901 is qualified separately on HIP before local indexer work; no local duplicate tiling implementation is created.
- No local duplicate indexer-score compaction remains.
- 1296 typed gather is retained only if it beats the direct selected-index seam or provides a necessary correctness fallback.
- Selected-cell attention reduces cost without full-cache conversion or extra large intermediates.
- Dense-mask removal reuses `bc_qsa_idx`; no parallel selection buffer is introduced.

## Notes

Evidence 1295 (old q8 profile): -3.7% ms/step @80K, -12% @160K, unchanged <32K; more accurate than masked path vs f32 CPU reference (max|dp| 0.056 vs 0.073). Its 192K OOM was the hipGraphInstantiate OOM fixed by 1302 (QFP06), not its workspace. Upstream sparse-FA work gathers compact q8_0 blocks and dequantizes during FA, +143% at 88K on CUDA. Supersedes PNRO12 (same intent, never implemented). RNX02 keeps the dense head-dim-256 decode side. Screening on production profile v2: flashnext-v2-1295-d65536/-d131072 (queue-v2-1295.sh). GPT: 1296 step-1 code req_8d8c8c85fecf4332; review of 1295 req_af96db92b8e34cfe.

2026-10-04 1295 on production profile v2 (f16/f16 240K, KV on both XTX; flashnext-v2-1295-d65536/-d131072, quick ABA, 256 tokens): ~80K 53.3 vs 53.5/52.8 ms/step = neutral (with f16 KV there is no full-cache q8->f16 conversion for the gather to avoid); ~160K 64.0 vs 65.9/66.3 ms/step = -3.2%, and t/s 47.6 vs 41.8/41.1 (+15%) because acceptance rose 162-163/276-279 -> 172/246 (gather changes numerics; earlier accuracy test found the gather closer to the f32 reference). Single new arm; needs a full ABBA at ~160K and ~220K, with acceptance tracked, before adoption. 1296's main target (q8 conversion) matters less on f16 KV; its typed gather is still worth it to drop the f32 round trip.

Upstream references verified 2026-10-05:
- #29825 merged indexer memory compaction: https://github.com/ggml-org/llama.cpp/pull/29825
- #29901 open tiled 4-head lightning indexer: https://github.com/ggml-org/llama.cpp/pull/29901
- SYCL sparse FA Qwen4Exp mechanism: https://github.com/ggml-org/llama.cpp/discussions/28695
- Latest published release: b11401 (`a7fb71fab83b474a0892b9a05aaa3a8ddca2729b`), 2026-10-05 00:36 UTC: https://github.com/ggml-org/llama.cpp/releases/tag/b11401

## Change Log

- 2026-10-03T15:20:34.367684+00:00 (created-by): Created by agent
- 2026-10-03T15:45:02.801299+00:00 (updated-by): Updated: section:notes
- 2026-10-04 (agent): Folded merged upstream #29825 into QFP04 as the canonical QSA indexer-memory baseline; separated score-buffer compaction from selected-cell sparse-attention ownership.
- 2026-10-05 (agent): Deep scan: staged #29901 HIP indexer qualification before local work; added direct selected-index sparse-FA as the preferred consolidation target over maintaining typed gather plus dense-mask paths; added explicit performance/ownership gates and SYCL mechanism reference.
