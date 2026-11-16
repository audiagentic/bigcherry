---
id: RAD40
order: 0
plan: patching-radiance-rdna3
state: superseded
created-at: '2026-11-01T11:26:10.176424+00:00'
breadth: ''
skill: intermediate
created-by: agent
work: S
priority: P1
---

# Host-link transfer harness and historical measurements (remaining tests superseded by HL02)

## Description

**Superseded by HL02 for all remaining cross-GPU transfer-method and contention testing.** RAD40 preserves the 2026-11-01 host-link physics observations: GPU kernel read/write, SDMA, CPU BAR push, isolated coherent/non-coherent/write-combined host reads, asymmetric contention, helper activation rounds and chunked handoff. HL01 completed the diagnosis with a physical slot swap on 2026-11-13. The unfinished all-pair and mode matrix, contended non-coherent probe, CPU-push comparison, duty cycle and TP3 shapes all belong to `docs/planning/active/run-radiance-host-link/HL02.md`.

This supersession does **not** mean the original all-mode/pair acceptance criteria were met. The historical evidence and harness remain reusable; production consumers are separate items (HL03, HL05, RAD42/RAD51 and VTP38).

## Steps

1. **Historical:** Measure solo GPU pulls/writes, SDMA, CPU BAR push and host-memory attributes, plus competing-card transfers (2026-11-01).
2. **Historical:** Measure 5.4 MB activation round and 10 MiB chunked producer/consumer handoff.
3. **Closed elsewhere:** HL01's 2026-11-13 slot swap determined the asymmetry follows the first PEG half, not physical card/VBIOS.
4. **Transferred to HL02:** remaining all-card/pair/method/contended-memory coverage, non-coherent mailbox, GPU-write trigger, CPU-push and TP3-sized overlap tests.
5. No second active RAD40 transfer campaign.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Preserve observed 2026-11-01 figures and the negative 6900 XT device-flag test as History below. The outstanding original all-modes/all-pairs gate was **not passed**; HL02 owns completion. New production decisions must rely on matched contended shapes and the separate RAD50 safety results, not only this serial harness.

## Effort & Risk



## Standards



## Acceptance Criteria

- [x] Keep original physics measurements, chunked/activation timing, and negative device-flag evidence.
- [x] Name HL01 as completed root-port diagnosis.
- [x] Transfer every outstanding cross-GPU transfer method and contention measurement to HL02.
- [x] Preserve separate production ownership for MoE activations/results, TP2 relay and TP3 transport.

**State: superseded** (not 'all original experiments completed').

## Notes

**Current disposition (2026-11-13):** The 2026-11-04 triage correctly rejected RAD41 as the owner of these experiments. HL02 now fills that ownership gap and supersedes the unfinished RAD40 campaign. RAD40's original harness and numbers below remain the historical source, not fresh performance gates. Equal *idle* host reads across coherence modes do not establish equality under contention.

**History — retained original results/review:**

No P2P on this host (memory no-p2p-on-dev-host). Feeds RAD41-RAD43.



BACKLOG TRIAGE 2026-11-04 (revalidated against main 6eb31a708): the proposed disposition "MERGE remaining transfer A/B into RAD41" was NOT applied; this item stays open. Why: PR #496 added stage stamps, and RAD40.md:59-80 records PCIe-mode, contention, BAR and chunk measurements; RAD42.md:22,57 still names RAD40 as a harness dependency. RAD41.md:14-31 is confined to result formats/sparse bytes, not RAD40's all-pair transfer-mode and CPU BAR versus GPU-pull validation, so its remaining acceptance criteria cannot be merged there unchanged. Active items that reference this one: RAD42. Keep the transfer-physics harness and missing mode/pair coverage until the transfer experiment's owner is explicitly designated.

**Ownership clarification (2026-11-13):** HL02 now owns the new matched cross-GPU contention matrix and GPU-pull/CPU-BAR-push crossover for TP3, TP2 and MoE. Keep RAD40 for its historical link-physics receipts, reusable benchmark harness and distinct unfinished all-pair coverage; do not duplicate HL02's contention arms. The RAD40 idle coherent/non-coherent/write-combined equality is not a verified PCIe No Snoop or contended-coherence result. TP3 integration belongs to VTP38, not RAD40.

### History — harness results (2026-11-01)

Alone, 21 MiB a pass (GB/s): kernel pull from coherent, non-coherent and write-combined host memory are identical (XTX
13.3, R9700 6.7, 6900 XT 3.6); SDMA H2D on one or two streams the same; SDMA D2H and GPU writes 14.2 / 6.9-7.1 / 3.6; CPU
BAR push 10.4-10.5 (XTX, 4 threads), 5.6 (R9700), 2.8 (6900 XT), one thread 7.1-8.1 / 4.9 / 2.7.

While XTX 0 pulls at full rate: every GPU-initiated transfer of another CPU-attached card halves -- kernel pulls, SDMA H2D
on any number of streams, and GPU writes to host too (XTX 1 6.8, R9700 3.4-3.5); only CPU BAR pushes keep their rate
(XTX 1 10.3, R9700 5.6). XTX 0 wins whichever card starts first (XTX 1 streaming, XTX 0 joining a second later: XTX 0 13.4).
XTX 1 with the R9700 or the 6900 XT: both full.

Activation round (5.4 MB to each helper at once, p50 finish over 20 rounds): XTX 0 662 us, XTX 1 853, tail 1,601 (its own
Gen3 x4). Staggering XTX 0 by 100-800 us only moves the wait (head makespan 852-1,522 us). A CPU push of XTX 0's copy
instead: XTX 0 683, XTX 1 655 (its uncontended time 651): the heads' makespan 853 -> 683 us. Chunked hand-off of a 10 MiB
result plane, producer XTX 0 or 1, consumer R9700: 128 rows 1.36x, 256 rows 1.27x, 512 rows 1.17x, 1024 rows 1.04x faster
than write-then-read. A 6900 XT producer hung the device-flag variant (consumer never saw the flags): the host-mediated
protocol first, as the review says.

### History — adopted 2026-11-01 review

- One trace contract for every item: per layer-step publish, activation-ready per helper, compute start/end, result-write
  done, combine start/end, engine add start/end, bytes per endpoint; host monotonic clock across processes, HIP events
  for GPU durations. Go gate: >2% on the 42K probe or 8 sessions, no p99 regression >5%.
- Stop/go for a transfer change: >=15% off the transfer-path makespan or >=0.3 ms off a layer-step critical path.
- Contention attribution with root: `lspci -vv` (MPS, MRRS, relaxed ordering, extended tags) for the cards and their root
  ports; swapping XTX 0 and XTX 1 physically separates slot from card.
- Next: a complete per-layer critical-path trace of the engine and the three helpers before kernel or protocol work.
- 2026-11-04T04:24:23.232307+00:00 (updated-by): Updated: section:notes

## Change Log

- 2026-11-01T11:26:10.176424+00:00 (created-by): Created by agent
