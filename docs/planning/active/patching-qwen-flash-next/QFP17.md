---
id: QFP17
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-04T07:22:34.987044+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Flash-Next prefill program (attribution, ubatch via upstream #29825, QSA sparse prefill, MoE MMQ, GDN, rank balance)

## Description

Prefill today (v4, 240K f16 KV, -ub 512 -b 512): ~1,100 t/s at ~10K, ~910 t/s at ~80K. GPT survey + methods (req_6010ee79a7df4529, 2026-10-04) ranked levers; key challenge: attribute the ~17% 10K->80K decay first (context-growing QSA/attention/indexer vs constant MoE/GDN), optimize the critical-rank wall path (not summed kernel time), and gate every item by Amdahl (reject optimistic bound < ~3%).

## Steps

0. VOID: 1237 + 1265 + 1253 are already composed by validated-enhancements.
1. PREFILL-ATTR: rocprof at 10K/80K/240K; bucket ubatches by context position; per GPU attribute MoE, GDN, QSA/indexer/mask/FATTN, collectives, copies and host enqueue.
2. QSA-UBATCH: #29825 is already relevant but insufficient alone; unlock ub1024+ by bounding dense QSA scratch rather than moving weights first.
3. QSA-CHUNKED-MASK: chunk only QSA selection-mask construction + masked attention over query-token tiles (start 256) while preserving the outer ubatch for MoE/GDN. This is now the immediate implementation slice.
4. QSA-INDEXER-TILE: independently qualify upstream #29901's 4-head lightning-indexer key/token tiling on HIP; do not conflate indexer compute with the dense-mask memory fix.
5. QSA-SPARSE-PP: only after chunking, consider selected-index attention that eliminates the dense mask entirely; promote only if remaining mask/attention wall time justifies the larger kernel change.
6. MOE-MMQ-PP2: after 1237/1265, tokens-per-expert tile sizing / multi-row expert tiles. Gate on routed MMQ critical-path share.
7. GDN-PP2: verify 1253 is taken; sweep chunk 16/32/64. Gate on GDN wall share.
8. TP-PREFILL-BALANCE: tune -ts against AllReduce arrival times; keep independent attention split.
9. QFP08 draft-prefill overlap. Gate on serialized draft-fill TTFT share.
10. Conditional two-microbatch compute/communication overlap only if collective tail remains material after balancing.

## Detailed Solution & Technical Design

### Immediate slice: bounded QSA query tiling

The 1329 allocation trace changes the priority: ub1024 fails because QSA materializes multiple `[n_kv,n_tokens]` f16 tensors on every Meta rank. At ~240K context each tensor is ~484 MiB at ub1024, versus ~242 MiB at ub512. The fix should reduce the `n_tokens` dimension of the QSA-only subgraph without reducing the model-wide ubatch.

Implement a QSA query-tile loop at graph construction/execution boundary. For outer `n_tokens` of 1024/1536/2048, process query slices of 128/256/512 through indexer score -> top-k/selection mask -> masked attention, then concatenate/write each output slice into the existing attention result. KV remains full-context and read-only. MoE and GDN continue seeing the original outer ubatch, preserving their larger-batch efficiency.

Memory target: peak QSA dense scratch must scale with `n_kv * qsa_tile`, not `n_kv * outer_ubatch`. At 240K, qsa_tile=256 caps each f16 dense mask near ~120 MiB independent of outer ubatch. Compared with ub1024's ~480 MiB mask, this removes ~360 MiB per live mask; with several overlapping graph temporaries the existing allocation trace predicts roughly 1-2 GiB/rank peak reduction.

Do not create a second generic chunking framework. Reuse existing graph/view/split helpers and keep the policy local to Qwen4Exp/QSA until another architecture demonstrates the same shape pathology. QFP17 owns prefill/QSA memory and throughput; QFP07 owns decode attention placement; QFP08 owns speculative overlap; QFP09 owns generic rank-lateness; 1329 owns allocation telemetry only.

### New upstream mechanism: #29901

Upstream llama.cpp PR #29901 (`cuda: tile the lightning indexer over keys and tokens for 4 heads`) is directly relevant to Qwen4Exp prefill. It stages 64 keys once and scores them against an 8-token tile for the 4-head case, retaining the vector path below 8 tokens. Published RTX PRO 6000 data reports the kernel at 6.4 ms versus 16.4 ms at kv=65536, nb=2048 (2.5x), with Qwen4Exp PP 3867 -> 4038 t/s at 128K and TG unchanged; indexer GPU share falls 9.6% -> 4.0%. The implementation is in `ggml/src/ggml-cuda/lightning-indexer.cu` and the CUDA source is also the HIP compilation path, so it warrants exact gfx1100/gfx1201 qualification rather than a CUDA-only assumption.

For HIP, sweep `{kv=10K,80K,240K} x {batch=8,64,256,512,1024,2048}` and record kernel wall, LDS, VGPR/SGPR, spills, occupancy and end-to-end prefill. The upstream tile uses shared `half2 k_shared[64][65]`, `float2 q_shared[8][64]`, 8 warps/block and float accumulation. Verify LDS/resource occupancy on gfx1100 and gfx1201 before carrying it. If HIP compilation or occupancy is poor, test 4 versus 8 warps and 32 versus 64 keys/block behind the existing architecture tuning mechanism; do not fork a standalone dispatcher.

#29901 is complementary to QSA-CHUNKED-MASK: indexer tiling reduces repeated key reads/compute, whereas query tiling bounds the later dense selection/mask lifetime. Measure both independently and combined to avoid attributing a memory-fit win to the indexer kernel.

## Code Samples & Guidance

Pseudo-structure for the QSA-only tile, preserving outer ubatch semantics:

```cpp
for (int64_t q0 = 0; q0 < n_tokens; q0 += qsa_tile) {
    const int64_t nq = std::min<int64_t>(qsa_tile, n_tokens - q0);
    ggml_tensor * q_tile = ggml_view_3d(ctx, q, ..., q0 * q->nb[2]);
    ggml_tensor * sel = build_qsa_selection(ctx, q_tile, k_full, ...);
    ggml_tensor * out = build_qsa_attention(ctx, q_tile, k_full, v_full, sel, ...);
    write_qsa_output_slice(result, out, q0, nq);
}
```

Required invariant: no `[n_kv, outer_n_tokens]` selection/mask tensor may remain live when `outer_n_tokens > qsa_tile`; allocation telemetry must prove the largest QSA mask second dimension is `<= qsa_tile`.

## Files

Expected llama.cpp touch points after pin inspection: Qwen4Exp graph construction in `src/llama-model.cpp` / architecture graph implementation at the current pin; QSA/lightning-indexer op construction; `ggml/src/ggml-cuda/lightning-indexer.cu` only for the separate #29901 qualification. BigCherry plan/recipe changes remain in QFP17 and a new patch package only after the design passes build/mechanics gates.

## Validation

ABBA fixed-prompt prefill at 10K/80K/240K, warm clocks, greedy identity. Sweep outer ubatch 512/1024/1536/2048 and qsa_tile 128/256/512. Record prefill t/s, TTFT, peak/final VRAM per rank, BIGCHERRY_ALLOC_TOP, QSA/indexer/FATTN kernel wall and collective arrival skew. Decode tg128/tg512 must remain unchanged within noise because the tiled path should gate on prefill-sized `n_tokens`.

For #29901 run test-backend-ops lightning-indexer coverage including n_head=4 and the model-level greedy parity lane on gfx1100 and gfx1201.

## Effort & Risk

QSA query tiling: medium implementation risk, high expected memory value. Main risks are graph-lifetime retention defeating the intended peak reduction, incorrect mask/query offsets, and extra launches erasing larger-ubatch throughput gains. #29901 HIP qualification is low implementation cost but architecture-resource risk.

## Standards

Fail closed to the current graph when shape/layout prerequisites are not met. No model-output tolerance widening. Keep measurement and architecture dispatch in existing BigCherry facilities.

## Acceptance Criteria

QSA-CHUNKED-MASK promotes only if ub1024 fits at 240K with >=1 GiB lower peak VRAM on the previously failing rank and end-to-end prefill improves >=5% versus ub512 baseline, with greedy identity and no representative decode regression >1%. ub1536/2048 are follow-on wins, not required for first promotion.

#29901-derived HIP tiling promotes only if it improves lightning-indexer wall >=20% on both gfx1100 and gfx1201 at a representative long-context prefill shape and improves end-to-end prefill >=2% on at least one 80K+ lane, with no decode regression >1% and no spill/occupancy pathology.

## Notes

Rejected from the survey for this lane: large-message CPU-root AllReduce (measured worse), KTransformers CPU-expert offload (experts fit VRAM; PCIe x4/no P2P), MLC-LLM/mistral.rs (no portable RDNA/IQ kernels).

CORRECTION 2026-10-04: 1237 + 1265 (MoE MMQ compact grid) and 1253 (chunked GDN prefill) are already composed through validated-enhancements; step 0 is void.

2026-10-04 ubatch fit: only ub512 fits comfortably at 240K. ub1024/1536/2048 add ~1.25-2.9 GiB/rank compute pressure depending on split; ub768 fits but collapses to ~134 t/s near full R9700 VRAM. Moving tensor split only relocates the OOM.

2026-10-05 allocation attribution (1329): dominant growth is dense QSA/full-attention masks mirrored on every Meta rank. Per QSA layer, selection-mask REPEAT `[247811,n_tokens]` f16 is ~242 MiB at ub512 / ~484 MiB at ub1024; masked indexer_sel ADD and `attn_inp_kq_mask` are each ~240/480 MiB. MoE intermediates are only tens of MiB. This makes QSA-only query tiling the narrowest immediate route to larger outer ubatches.

2026-10-05 upstream scan: llama.cpp #29901 was updated 2026-10-04 and adds 4-head key/token tiled lightning-indexer prefill; upstream reports 2.5x kernel speedup and 3867 -> 4038 t/s Qwen4Exp PP at 128K on RTX PRO 6000. Treat these NVIDIA numbers only as mechanism evidence; qualify HIP independently. Current release scan shows b11386 on 2026-10-04; #29943 selective expert-copy callback and #29927 AMD q1_0 byte permute belong to MET/PKC ownership and are not duplicated here.

2026-10-05 live-at-peak attribution (1331, v6, 240K f16): main graph compute-buffer peak is at indexer_sel of the first QSA layer. ub1024: 1748 MiB live = attn_inp_kq_mask input 480 + mask_all REPEAT 484 + indexer_sel ADD 480 + kpool mask leaf 120 + ~180 other; ub512: 883 MiB (same three, halved). MoE/GDN intermediates are not at the peak. 1330 (in-place sel add) targeted the 480 ADD but the ub1024 reserve stayed 1560.72 MiB and greedy diverged (cause: in-place add or row padding, not the FA strided read); not adopted. Next: (1) peak trace with 1330 on to see where the peak moves; (2) token-chunked QSA mask/attention (build mask_all + sel per 256-token slice, FA per slice) to cut the 484+480 pair ~4x; (3) longer-term avoid the dense kq_mask input via the sparse FA path (compact mask / n_kv_max_query).

2026-10-05 1332 (QSA masks + attention per token chunk; one kq_mask view per chunk shared across QSA layers - per-layer input views were materialised by the meta backend: 3 GiB OOM at ub512). Results (v6, 240K f16, 80K fill): ub512 unchunked 868-895 prefill / 59-63 decode; ub512 chunk256 806-827 / 61.8-61.9 (compute buffer 1020.9 -> 785.6 MiB, ROCm3 616 -> 505); ub1024 chunk256 FITS for the first time: 898-901 prefill / 60.0-60.9 decode (+1-3% prefill vs v6 ub512). Sweep queued: 512:0, 1024:256, 1024:512, 2048:512, 2048:1024 (queue-ubchunk). OPEN CORRECTNESS QUESTION: 1332's 24K greedy text diverges from v6 at byte 41 into exactly the same alternative text as 1330 did ('summarize the entire document they've provided in detail' vs 'summarize in detail the content of the four documents'); two independent rebuilds of the QSA mask agreeing suggests the production mask path differs semantically, not just numerically. Settle with a CPU f32 reference (long-ctx-profile CPU_REF=1) or single-device run at the divergent token before adopting either; check whether the unpatched dense-mask path has a meta-backend split/mirroring issue.

2026-10-05 mask correctness RESOLVED (queue-maskref2, 24K prompt = 38.7K tokens, no MTP, 16 tokens): v6 GPU (no MTP) and the CPU f32 reference produce the same text ('summarize the entire document they've provided in'), matching 1330/1332's output. The divergent text is v6 WITH MTP ('in detail the content of the four documents'): token 9 is a near-tie (top-1 0.479 v6 / 0.566 f32) that MTP verify-batch numerics flip. Not a mask bug; greedy byte-identity vs v6+MTP is too strict at near-ties - use agreement with the f32 reference / top-k prob match for mask-path changes. 1332 single-token decode segfaulted in ggml_backend_meta_graph_compute (whole-mask view of the kq_mask input) -> fixed by chunking only when n_tokens > chunk (decode/MTP verify keep the dense path); queue-chunk3 re-checks.

2026-10-05 queue-chunk3 (1332 with n_tokens > chunk gate): MTP serving at 24K fine (ms/step 41.2 all arms; text = f32-side near-tie). No-MTP single-token decode after a chunked prefill still segfaults in ggml_backend_meta_graph_compute (null write) even though decode itself takes the dense path; chunk 0 decodes fine (38.6 t/s). Likely stale meta-backend bookkeeping for the per-chunk views of the kq_mask INPUT created during prefill. Next fix if resumed: never view the input - copy the chunk's kq_mask rows with ggml_get_rows from a small per-chunk I32 row-index graph input (needs an llm_graph_input set_input hook), or fix the meta view bookkeeping. Parked as experimental: payoff is ub1024 fitting at 240K with only +1.5-6% prefill at an 80K fill; bigger prefill wins need removing the dense kq_mask input (step 3).

2026-10-05 prefill attribution on pin b11402 (build b-defon-b11402, production config 240K f16 ub512, rocprofv3 kernel trace of one uncached fill, tools/lab/flash-next/queue-prefill-profile.sh + prefill-kernel-table.py; runs prefillprof-b11402-d20480 / -d81920). Per-device GPU kernel time, XTX0 / XTX1 / R9700.
31.8K tokens (1021 t/s under profiling, 62 ubatches, ~23.5 s kernel time per device): all-reduce 7.1 / 6.7 / 7.7 s (30 / 29 / 33%); MMQ incl. MoE 6.2 / 6.2 / 6.4 s (26-27%); float matmul 3.6 / 3.6 / 2.9 s (15 / 15 / 13%); flash attention 1.96 / 1.96 / 0 s (8%; was 2.39 s on pin 0504396 for the same prompt); indexer + top-k 0.47 / 0.47 / 0.68 s (2-3%); GDN 0.6 / 0.6 / 1.1 s.
99.3K tokens (828 t/s, 194 ubatches, ~89.5 s kernel time per device, 120 s wall): all-reduce 22.2 / 20.9 / 35.6 s (25 / 23 / 40%); MMQ 19.4 / 19.7 / 20.1 s (22%); flash attention 18.9 / 18.9 / 0 s (21%); float matmul 11.1 / 11.2 / 9.0 s (12 / 13 / 10%); indexer + top-k 4.3 / 4.4 / 6.3 s (5 / 5 / 7%); GDN 1.8 / 1.9 / 3.5 s. Drafter (6900 XT): 12.4 s, 56% flash attention.
Per ubatch: all-reduce on the XTX is constant at ~114 ms (7.1 s / 62 and 22.2 s / 194), MMQ constant at ~100 ms, float matmul ~57 ms; flash attention grows with context, 32 ms average over the first 32K and 97 ms average over 100K, so it passes all-reduce somewhere past 100K; the indexer grows too (7.6 -> 22 ms). The R9700 holds no attention (BIGCHERRY_ATTN_TS 1,1,0), and its extra all-reduce time (35.6 s vs ~21.5 s) matches the XTX attention time it waits for: at long context the R9700 idles inside the collective while the XTX run attention.
Consequences for ordering: (a) all-reduce is the largest block up to ~100K (about 18% of wall at 100K, 21% at 32K); (b) at the long-context end, attention + indexer is 26% of XTX kernel time at 100K and still growing, so QFP25 (sparse prefill attention, cost ~ n_sel instead of n_kv) and the rank imbalance it causes on the R9700 move up with context; (c) MMQ is a constant 22-27%; (d) the upstream tiled indexer (#29901) is worth at most 5-7% of kernel time at 100K. 1332 chunk does not reduce attention work, it only allows ub1024.

## Change Log

- 2026-10-04T07:22:34.987044+00:00 (created-by): Created by agent
- 2026-10-04T07:55:32.606718+00:00 (updated-by): Updated: section:notes
- 2026-10-04T12:10:08.450336+00:00 (updated-by): Updated: section:notes
- 2026-10-04T13:00:59.838288+00:00 (updated-by): Updated: section:notes
- 2026-10-05: Deepened QFP17 around QSA-only query tiling and added independent HIP qualification of upstream #29901.

## Ledger-events

- chg_20261004_161454_found-what-limits-the-larger-p_4462
- 2026-10-04T16:14:57.881449+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-04T16:15:06.578561+00:00 (updated-by): Updated: section:notes
- 2026-10-04T18:12:07.315403+00:00 (updated-by): Updated: section:notes
- 2026-10-04T19:15:06.825550+00:00 (updated-by): Updated: section:notes
- 2026-10-04T19:37:51.020304+00:00 (updated-by): Updated: section:notes
- chg_20261004_235349_long-context-flash-next-candid_9064
- 2026-10-04T23:53:53.025717+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-05T12:52:50.061761+00:00 (updated-by): Updated: section:notes

## Plan Review - 2026-10-05 prefill round

Scope: b11402 `d89651a7b205`, Brutus 3-rank Meta split `0.31,0.27,0.42`, attention/KV on the two XTX, f16 KV, ub512. Shared-branch note: while this review was running, patch `1334_hip_sparse_flash_attn` landed behind `BIGCHERRY_FA_SPARSE=1`; treat QFP25 as an implementation-ready hardware-proof item, not a future design project. MTP look-ahead is out of scope.

### Amdahl / priority order

Use critical-path wall, not summed rank kernel percentages. The current-pin profile gives ~31.1 s wall at 31.8K (`31793/1021`) and 120 s at 99.3K. Bounds below are the optimistic wall fraction removable if that candidate eliminated its owned work completely.

| Rank | Candidate | 31.8K / 99.3K hard wall bound | Expected E2E / effort-risk | Decision |
|---|---|---:|---|---|
| 0 | Existing provider/protocol screen | AR bound 22.8% / 18.5% | 0-5%, S/low | Run before new collective code; may make PGC14/PGC15 unnecessary or change their baseline. |
| 1 | QFP25 / `1334_hip_sparse_flash_attn` | FA 6.3% / 15.8%, rising strongly with context | ~4-12% at 100K+ if sparse WMMA behaves; S-M/medium now that code exists | **First hardware proof.** Also removes the XTX-attention wait seen as inflated R9700 collective time. |
| 2 | 1332 ub1024 + QSA chunk256 | collective-only ceiling 22.8% / 18.5%; whole-MMQ ceiling 20.6% / 16.8%; actual mechanism only amortizes fixed costs/improves batch geometry | **measured +1.5..7%** at 80K; S/medium | **Second hardware proof** on b11402 at 32K/100K/200K. |
| 3 | PGC15 token-tiled compute/AR overlap | all AR 22.8% / 18.5%; phase-1 one-of-two reductions <=11.4% / 9.3% | phase-1 ~3-6%, full ~6-10%; L/high | Implement only after provider screen; phase-1 outline in PGC15. |
| 4 | upstream #29901 tiled lightning indexer | max-rank indexer+top-k 2.2% / 5.3% | ~1-3%; S-M/low | Merged upstream after b11402; exact backport/HIP qualification is cheap. |
| 5 | PGC14 RCCL transport/protocol | AR 22.8% / 18.5% | 0-3% likely; S screen, L if transport code | Continue only if RCCL protocol/topology sweep exposes a real loss. |
| 6 | QFP26 grouped MoE MMQ + gate/up fusion | whole MMQ 20.6% / 16.8%; QFP26 owns only a subset | ~3-6% if grouping/fusion removes repeated quantize/dispatch; L/high | After sparse FA/ub1024/indexer; preserve 1237/1265 expert maps. |
| 7 | QFP24 pinned input ring | only defensible ceiling is the ~30% GPU-idle envelope; input copies are a subset | prior unsafe +3.7/+7.2%, safe staging -2%; expected 0-3%; M/high lifetime risk | Park until timeline proves pageable/pinned input stalls remain exposed after 1326. |
| 8 | PGC12 phase-aware provider | 0 incremental for current prefill: large messages already route RCCL | ~0%; S | Keep as policy owner, not a new optimization. |

Drop/park for this round:

- **PGC16 fewer AllReduces: no foldable Qwen4Exp pair.** The two reductions/layer are not independent. `build_layer_attn()` ends in the tensor-split `wo` projection (`attn_output`); its reduced output immediately feeds `build_hc_combine()` (sigmoid/scale/multiply/add or fused HC post), then `build_hc_mix()`/normalization before the FFN/MoE branch exists. The FFN reduction therefore depends on the first reduction. The Meta backend's existing `get_i_delayed()` already folds the only safe algebraic case (`PARTIAL a`, independent `PARTIAL b`, then MIRRORED `ADD`); this graph cannot satisfy it. Hypothetical half-AR bound is 11.4%/9.3%, **actual eligible-pair bound is 0**.
- **New half-width prefill wire: drop.** b11402 RCCL already converts large F32 reductions to BF16 for `n_backends==3 && ne>=131072`. A 2560x512 reduction has 1,310,720 elements: logical F32 size 5.24 MiB, RCCL payload ~2.62 MiB plus F32<->BF16 conversions. `1272_ar_host_compressed_wire` is the two-GPU internal-provider codec and `1250` is P2P-oriented; neither improves this active 3-rank RCCL path. Q8 would add a new numerical/quality trade for a problem already halved.
- **GDN:** 1253 is already promoted. Hard residual ceiling is only 3.5%/2.9% of wall and any second-order chunk tuning owns a fraction of that; park unless the 200K profile grows materially.

### Gate 0: settle the provider before PGC15

The observed `ncclDevKernel_Generic_4` is RCCL. On Linux b11402 `ggml_backend_cuda_comm_init()` defaults to NCCL/RCCL; the 3-rank 5.2 MiB logical F32 call takes the large BF16 branch in `ggml_backend_cuda_comm_allreduce_nccl()`. Stock `ggml_backend_cuda_comm_allreduce_internal()` asserts `n_backends == 2`, so it is not a 3-rank alternative. `1244_gp11_internal_allreduce_nway_root` is the only existing 3-rank internal/root candidate. The CPU-root large path is already negative evidence: 1291 records a prior 32 MiB large-path prefill change around 1450 -> 1060 t/s.

Run the existing balanced `tools/lab/flash-next/prefill-provider-sweep.sh` at ~32K and ~100K before writing PGC15 transport code:

1. `--allreduce ccl` versus `--allreduce adaptive`; they should converge to the same RCCL large path. A difference means dispatch/policy overhead or wrong phase classification and belongs to PGC12/0840.
2. CPU-root large diagnostic: `--allreduce host BIGCHERRY_AR_CPU_ROOT_LARGE_MAX_BYTES=8388608` and `BIGCHERRY_AR_CPU_ROOT_CHUNK_BYTES={524288,1048576,2097152}`. Expect rejection unless the new pin/topology overturns the existing large-path loss.
3. `--allreduce root3` with the 1244 experiment composition; compare the same 5.24 MiB logical shape. If root3 wins >=3% E2E with parity, promote/repair that provider before PGC15.
4. RCCL protocol proof: `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,GRAPH,COLL`; then diagnostic `NCCL_ALGO=Ring|Tree` and supported `NCCL_PROTO=Simple|LL|LL128`. Do not infer protocol from `ncclDevKernel_Generic_4`. With no P2P, verify the logged SHM/PCIe route and measured bandwidth rather than assuming it.

The 38.9 ms max is not representative transport latency given 0.89 ms median / 2.03 ms p90. Align the same collective ordinal across ranks. If one rank enters ~39 ms before peers and peers have normal kernel duration, the long kernel is arrival skew/wait (the 100K R9700 profile already demonstrates this class: its excess AR time tracks XTX attention). If all ranks enter together and all run long, investigate RCCL/OS/PCIe/protocol stalls. One long rank is not evidence for a slower wire protocol.

### Rank 2 concrete: 1332 ub1024 proof

Use the existing implementation, not a new patch: `BIGCHERRY_QSA_CHUNK=256`, outer `-ub 1024 -b 1024`; keep the existing small-batch gate so decode/MTP-sized batches use the dense path. Compare against `-ub 512`, chunk off, from the same b11402 composition. At fixed prompt length the model still performs 96 reductions per outer ubatch, but ub1024 approximately halves ubatch count while doubling reduction payload; bytes/token are nearly unchanged. The expected gain is therefore launch/conversion amortization plus better MMQ/float-matmul geometry, not a 2x communication reduction.

Mechanism gate: kernel census must show ~half the ubatches and ~half the 96-AR bursts for a fixed fill, with no new QSA launch explosion; record RCCL total time/token, MMQ time/token and peak VRAM. ABBA: 32K/100K/200K, at least 4 arms/config after clocks stabilize. Correctness: CPU-f32/top-k reference at the known near-tie, no-MTP greedy where margin is safe, QFP28 multi-request/cache-reuse state gate, and decode control <=1%. Promote only if current-pin 100K or 200K prefill is >=3% faster (or a depth-specific larger win is reproducible) and 240K reserve remains safe.

### Proof standard for the remaining round

| Candidate | Mechanism proof | Performance proof | Correctness gate |
|---|---|---|---|
| 1334 / QFP25 | rocprof: sparse index compaction + sparse `flash_attn_ext_f16` selected; FA reads/work scale with selected cells, XTX FA wall falls and R9700 collective-wait tail falls | ABBA ~100K/~200K, flag 0/1; then compose with 1332 | `test-backend-ops` sparse FA on gfx1100+gfx1201; logits/top-k vs dense and CPU-f32; multi-request gate |
| 1332 | ubatch/AR census and peak allocator trace | ABBA ub512:c0 vs ub1024:c256 at 32K/100K/200K | CPU-f32/top-k + multi-request/cache reuse; decode <=1% |
| PGC15 | timeline must visibly overlap producer tile `i+1` with range AR tile `i`; trace range offsets/slots/provider | same-build flag 0 vs tile128/256 ABBA 32K/100K/200K; first isolate from 1334, then compose | direct range-vs-whole provider test; model logits/top-k vs CPU-f32 because BF16 collective segmentation may change low bits; multi-request + graph replay |
| #29901 | tiled lightning-indexer kernel selected; kernel wall/resource census on both XTX/R9700 placements | ABBA 100K/200K; require >=2% E2E or retain upstream-only | backend-op lightning indexer + model logits/top-k |
| PGC14 | RCCL debug confirms transport/protocol; per-call latency/bandwidth improves with same count | ABBA 32K/100K; >=2% E2E before code carry | same BF16 reduction/logit reference; no provider-state drift |
| QFP26 | grouped expert map reused; fewer quantize/MMQ launches and gate/up reads with same routed IDs | ABBA 32K/100K/200K; profile MMQ/quantize separately | deterministic expert IDs, backend-op `MUL_MAT_ID`, logits/top-k vs reference |
| QFP24 | timeline proves host-input copy/sync bubbles removed, no hidden staging overwrite | ABBA short + 100K prefill | repeated/multi-request stress; exact input bytes and output reference |

Order of execution: **provider/protocol screen -> 1334 hardware proof -> 1332 current-pin proof -> PGC15 phase-1 microbench/patch -> #29901 -> PGC14 only if screen justifies it -> QFP26 -> QFP24.**
