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

## Change Log

- 2026-10-04T07:22:34.987044+00:00 (created-by): Created by agent
- 2026-10-04T07:55:32.606718+00:00 (updated-by): Updated: section:notes
- 2026-10-04T12:10:08.450336+00:00 (updated-by): Updated: section:notes
- 2026-10-04T13:00:59.838288+00:00 (updated-by): Updated: section:notes
- 2026-10-05: Deepened QFP17 around QSA-only query tiling and added independent HIP qualification of upstream #29901.
