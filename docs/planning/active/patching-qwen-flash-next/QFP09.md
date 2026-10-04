---
id: QFP09
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-03T15:21:02.481672+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: M
---

# Per-rank critical-path census by op class (R9700 slow rank) -> selective per-class skew

## Description

QFP09 owns **cross-rank placement and arrival-skew diagnosis only**. The original hypothesis that a single R9700-heavy tensor family could be moved to remove most skew has now been tested and is not supported: 1305 expert-only placement was neutral/worse, and the late rank flips with context because XTX-only attention becomes dominant at long context. QFP09 therefore keeps the per-AllReduce segment attribution needed to decide placement, but must not grow a second launch/fusion programme.

The remaining short-context R9700 segment gap (~0.66 ms/token) is local launch/dispatch idle. That belongs to QFP13's measured kernel-count/fusion programme; QFP09 consumes QFP13 results and then re-measures whether fewer launches reduce arrival skew. Attention placement remains QFP07. Collective transport remains QFP01/1291. Shared-expert fusion remains RNX10/PRBE owners.

## Steps

1. Keep `ar-segment.py` as the authoritative placement diagnostic, aligning boundaries by timestamp rather than ordinal index.
2. At representative ~10K/~80K/~160K depths, decompose each late-rank interval into compute classes plus local no-kernel gap; record which rank is last and why.
3. Do not add another per-class split unless one movable class alone explains >=0.5 ms/token late-rank excess and a feasible placement has VRAM headroom on the receiving rank(s).
4. Send local launch/dispatch opportunities to QFP13 and its canonical backend owners. After an accepted QFP13 fusion/reduction, rerun the same QFP09 segment census to measure whether arrival skew falls.
5. Send depth-dependent attention imbalance to QFP07. Do not duplicate attention-head rotation/sequence partitioning here.
6. Park 1305 unless a new census materially changes the expert balance point.

## Detailed Solution & Technical Design

Treat each AllReduce boundary as a synchronization observation, not as an optimization seam. For boundary `i`, derive per-rank `arrival_i` from the produce/collective transition and pair boundaries by nearest monotonic timestamp. For each rank, attribute `[arrival_{i-1}, arrival_i)` to kernel classes plus `idle = interval - union(kernel_busy)`. Compare the last-arriving rank against the first-arriving rank for the same boundary. Aggregate **excess** time per class rather than raw GPU time; only excess can explain skew.

Decision hierarchy:
- excess attention at long context -> QFP07 placement;
- excess routed/dense matmul with a stable receiving-rank budget -> existing tensor/per-family placement mechanism;
- excess local no-kernel time or repeated tiny operators -> QFP13 fusion/kernel-count owner;
- collective/resume time after all ranks have arrived -> QFP01/1291, not QFP09.

This prevents double-counting QFP13's ~3-4 us launch gaps as a placement problem. In particular, upstream llama.cpp PR #29948 demonstrates the preferred mechanism class for launch-bound work: fuse gate/up MMQ and GLU into one launch with one accumulator and register write-back instead of adding scheduler machinery. Its reported dense-FFN prompt gains are 1.9-10.7% depending on GPU/quant, but it is not directly applicable to Flash-Next routed `MUL_MAT_ID`; use it as design evidence only unless QFP13 trace proves an equivalent local topology.

## Code Samples & Guidance

`tools/lab/flash-next/ar-segment.py` should emit, per depth and rank: `last_fraction`, `excess_moe_mmvq_ms_tok`, `excess_mmvq_ms_tok`, `excess_mmvf_ms_tok`, `excess_attention_ms_tok`, `excess_idle_ms_tok`, and p50/p90 total arrival skew. Keep timestamp-based alignment; ordinal AR alignment is known unreliable.

Any new placement override must reuse the 1303/1305 split-vector mechanism rather than create a second tensor-placement registry.

## Files

- `tools/lab/flash-next/ar-segment.py` — authoritative skew attribution.
- `docs/planning/active/patching-qwen-flash-next/QFP07.md` — attention placement owner.
- `docs/planning/active/patching-qwen-flash-next/QFP13.md` — launch/fusion census owner.
- Existing 1303/1305 patch sources — only placement mechanism to extend if the gate is met.

## Validation

ABBA at matched context and model composition; report ms/step, effective TG, acceptance, per-rank last-arrival fraction, p50/p90 skew, excess time by class, peak VRAM, and max context. Greedy/logit correctness must remain within the owning patch's gate.

## Effort & Risk

Low for diagnosis; medium for any new placement. Main risk is optimizing raw per-rank work rather than critical-path excess, or moving work onto an XTX that is already attention-bound at long context.

## Standards

One owner per mechanism. QFP09 diagnoses and assigns placement; it does not implement generic fusion, graph scheduling, attention partitioning, or collective transport.

## Acceptance Criteria

A new QFP09 placement patch requires all of: >=0.5 ms/token attributable movable excess before implementation; >=3% end-to-end decode improvement at one representative depth and no >2% regression at another; reduced p50 arrival skew or last-rank excess consistent with the claimed mechanism; no loss of required context/VRAM headroom; correctness gate pass. Otherwise park the placement and advance through QFP13/QFP07 instead.

## Notes

Split screens 2026-10-03 (QFN01 notes): 2,2,3.4 49.3; 2,2,3.8 49.1; 1.9,2.1,3 48.1; 2.1,2.1,2.8 48.0; 2.2,2.2,2.6 47.4; 2.3,2.3,2.4 46.8; 2.4,2.4,2.2 46.8 ms/step vs ~48.3. Global skew loses context (2.3,2.3,2.4 fits only 160K q8). GPT RV4215 rank #4. Tools: tools/lab/flash-next/long-ctx-profile.sh perf/timing/synctrace modes, sync-tracer.c.

2026-10-04 profile v2 decode census (flashnext-v2-profile, rocprofv3, tools/lab/flash-next/rank-census.py; per generated token, profiler-slowed). GPUs are busy only 39-44% of wall at every depth: ~56-61% of each token has no kernel running (host scheduling/sync/launch gaps) - the largest single lever. Slow rank flips with depth because 1303 put all attention on the two XTX: ~10K: R9700 slowest (moe-mmvq 1.42 vs 0.85 ms/tok on XTX at 42% vs 27-31% expert share), XTX wait ~1 ms/tok more in allreduce. ~80K: XTX flash-attn 1.46-1.53 ms/tok, R9700 now waits (allreduce 3.17 vs 2.5-2.75). ~160K: XTX flash-attn 3.4-3.7 ms/tok, R9700 allreduce wait 5.7 vs 2.2-3.0 ms/tok -> ~3 ms/tok of R9700 idle. Kernels/token ~1300-1410 per GPU (elementwise ~370-400, quantize ~180-200, mmvq ~150-166). Draft (6900): 1.2 ms/tok at 10K, 3.1 at 160K (attention 1.5).

2026-10-04 promoted to P0 by the QFP11 boundary decomposition: per-AR arrival skew (median ~80 us, R9700 last in ~70% of ARs; p90 800 us at 80K when XTX attention is slow) costs ~2.4 ms/token, far more than split boundaries (~0.4).

2026-10-04 1305 (BIGCHERRY_FFN_TS) screens on profile v2 placement at -c 65536 (-c 131072 OOMs XTX0 with more expert share): ~24K: 0.34,0.33,0.33 45.8 vs 46.0/45.6 ms/step (neutral); 0.37,0.35,0.28 47.1 vs 45.8/46.0 (+2.6%, worse); ~40K: 0.34,0.33,0.33 49.3 vs 50.1/49.5 (-0.8%, noise). Conclusion: moving the expert family alone off the R9700 does not remove the arrival skew. 1305 parked.

2026-10-04 per-segment attribution (`ar-segment.py`, timestamp-aligned): R9700 last at 76% (~10K) / 85% (~80K) of ARs. Extra time on late GPU vs first: ~10K moe-mmvq 0.68, mmvq 0.52, mmvf 0.17, idle gaps 0.66 ms/token; ~80K moe-mmvq 1.00, mmvf 0.23, mmvq 0.13. R9700 expert matvec per share ~22% slower than XTX. Expert-only balance point ~0.355,0.355,0.29 conflicts with XTX attention load, explaining why 1305 does not win.

2026-10-05 consolidation: the 0.66 ms/token short-context local-idle lead is now explicitly transferred to QFP13 rather than spawning a QFP09 scheduler/fusion path. QFP13 already measured ~3.1-3.7 us inter-kernel gaps and has reduced kernels/token materially with 1307-1313/1327-class work. Upstream PR #29948 independently reinforces operator fusion as the correct mechanism class (single MMQ accumulator + GLU at write-back; dense prompt gains +1.9..10.7%), but Flash-Next routed MoE requires its own trace/proof before adapting that design. QFP09 re-enters only after those local launch reductions to see whether rank arrival skew actually moved.

## Change Log

- 2026-10-03T15:21:02.481672+00:00 (created-by): Created by agent
- 2026-10-03T16:05:27.527362+00:00 (updated-by): Updated: section:notes
- 2026-10-03T16:35:24.607549+00:00 (updated-by): Updated: priority='P0', section:notes
- 2026-10-03T16:59:58.379545+00:00 (updated-by): Updated: section:notes
- 2026-10-03T17:25:54.754317+00:00 (updated-by): Updated: section:notes
- 2026-10-05T06:08:42+11:00 (updated-by): Consolidated local launch-gap work under QFP13; added ownership and promotion gates.
