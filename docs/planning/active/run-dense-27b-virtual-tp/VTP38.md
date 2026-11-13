---
id: VTP38
order: 0
plan: run-dense-27b-virtual-tp
state: pending
created-at: '2026-11-13T05:14:45.261435+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P1
---

# Host-reduced TP3 prefill all-reduce: the CPU sums the partials and pushes the result, so no card pulls host memory

## Description

In TP3 prefill the all-reduce is 57-65% of device time (int8, 2048 tokens, pipelined). The rad0031 timeline (VTP36, 2026-11-13) shows two costs that the GPU-pulled two-shot cannot remove:

1. The x4 link floor. With message S and R9700 share w, every exact GPU scheme makes the R9700 send S (its partial minus its shard, plus its reduced shard) and receive (1+w)S. At 4,4,1 that is S out and 10/9 S in over ~6.7 GB/s; measured push 1334 us for ~8.9 MB. A one-shot (every rank reads all partials) is worse for prefill: the R9700 would receive 2S.
2. XTX 0's host-read contention (HL01: no configuration cause; CPU-internal). While XTX 0 pulls, XTX 1 reads at ~6.8 GB/s instead of ~13 (reduce 1301 vs 889 us), and every rank waits ~400 us at the second barrier.

A host-reduced all-reduce removes both: each card writes its partial to pinned host memory once (S out), the CPU sums the three partials in a fixed order, and the CPU writes the result straight into each card's VRAM through the BAR / DMA-BUF (S in). No card pulls host memory, so the contention never starts (RAD40: CPU BAR pushes keep full rate under it), and every card moves the minimum S each way. Every rank receives the same bytes, so ranks cannot diverge.

## Steps

1. Gate: HL02 contention arm in the all-reduce shapes (XTX 1 recovers >= 90% of ~13 GB/s while XTX 0 is CPU-fed). Stop if it does not.
2. Gate: R9700 CPU-write path (RAD50 gate 2, inconclusive) passes serial full-word verification at 4 KiB-10 MiB.
3. Host-side microbench (no engine): 3 x S bf16 partials (S = 320 KiB .. 10 MiB) summed with AVX2 in the GPU kernel's order and precision, written to three DMA-BUF targets; measure CPU time, memory bandwidth (the i7-12700KF must sustain ~30 MB in + ~30 MB out per prefill all-reduce) and per-card arrival, in pieces so writes and reads overlap.
4. Engine patch (opt-in, layout key, e.g. `tuning.all_reduce_host = prefill`): a host reducer thread per world, a GPU-side ready/done protocol reusing RAD50's corrected wall_clock64 waiter, and the existing two-shot as fallback. Decode keeps the GPU one-shot (handshake-bound; RAD50 single-row push is slower).
5. Measure: rad0031 timeline, 2x ABBA vs current (prefill 512/2048/6144, four 4096 prompts, long prompt beside 4 streams, 1/4/6/8 streams), then the VTP37 reordering quality gate (identical bits expected if the CPU sum matches the GPU order; otherwise KLD vs BF16).

## Detailed Solution & Technical Design

Per piece of rows: each rank's GEMM epilogue (or a copy kernel) writes its partial to its pinned host slot and raises a flag. A CPU thread (one per piece stream, pinned to P-cores) waits for three flags, sums p0+p1+p2 in the GPU two-shot's order with fp32 accumulation over bf16 inputs, and streams the bf16 result with non-temporal stores into each rank's DMA-BUF-mapped VRAM receive buffer, then writes a ticket. Each rank's next kernel (norm/quant) waits on its ticket in VRAM (local polling, no PCIe reads). Pieces (as rad0032) let card writes, CPU sums and CPU pushes overlap; the expected critical path is the R9700's S/6.7 GB/s each way, ~1.4-2 ms for a 10 MB message vs ~4 ms today.

Alternative without the CPU sum (cheaper fallback if CPU memory bandwidth is the limit): hierarchical exchange. XTX 0 and XTX 1 exchange partials and add them; the R9700 writes its partial once and both XTX read it and finish; the XTX pair publishes XTX0+XTX1 (half each) and the R9700 reads that S and adds its own. R9700 moves S each way; all ranks add (p0+p1)+p2. It still has XTX ranks pulling, so it keeps the HL01 contention.

## Code Samples & Guidance



## Files

engines/radiance/patches (new package), libr4d r4d_ar_entry.hip / r4d_ar_twoshot_Nrank_exact.hip (fallback), arch/qwen35_fp8/qwen35_fp8.cpp (pipelined prefill AR call sites), tools/lab/radiance/rad50-tester (HL02 arm), engines/radiance/layouts/qwen27b.toml.

## Validation

Timeline: no rank waits > 10% of the all-reduce at any barrier; R9700 link time within 20% of S / measured x4 rate per direction. 2x ABBA: prefill 2048 TTFT improves >= 10% with decode within noise. Quality: VTP37 reordering gate (noise floor vs serial, then mean/p99.9 KLD within +10%, top-1 within -0.2 points, PPL within +0.1% vs BF16). Stress: 12 clean runs, no stale tickets.

## Effort & Risk

L. Risks: CPU memory bandwidth (~60 MB of traffic per prefill all-reduce at 2048 tokens); R9700 BAR path unproven; GPU-visible readiness of CPU writes (RAD50 needed a corrected wall_clock64 watchdog on gfx11); one more host thread on the critical path; HL04's no-go was a different case (combine exposed only 0.1-0.2 ms).

## Standards



## Acceptance Criteria

Opt-in layout key, generic (no model or card constants); default unchanged until the ABBA and quality gate pass; fallback to the GPU two-shot on any protocol timeout.

## Notes

Created 2026-11-13 from the VTP36 all-reduce timeline review and the HL01 lspci/IOMMU finding. Depends on HL02 and RAD50 gate 2 (R9700). Related: VTP36 (two-shot pieces/duplex, uneven XTX shares), RAD50/RAD51 (CPU BAR push lab), RAD40 (contention measurement), HL04 (different-layout no-go).

## Change Log

- 2026-11-13T05:14:45.261435+00:00 (created-by): Created by agent
- 2026-11-13T05:15:16.842115+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:acceptance_criteria, section:notes
