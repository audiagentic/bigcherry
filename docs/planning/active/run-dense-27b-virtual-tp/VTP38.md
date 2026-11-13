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

A host-reduced all-reduce is intended to avoid GPU-initiated host reads: each card writes its partial to pinned host memory once (S out), the CPU sums the three partials in a fixed order, and the CPU writes the result straight into each card's VRAM through the BAR / DMA-BUF (S in). If the source-side GPU writes do not trigger HL01 contention and the CPU BAR path remains independent, this eliminates the GPU-pull slowdown and reduces R9700 receive traffic. Rank-consistent delivery is required but does not itself prove bit-exact equivalence to the current GPU reduction; that must be checked separately.

## Steps

1. Gate: HL02 C **GPU-pull recovery** arm (XTX 1 pulls >=90% of its own idle baseline while XTX 0 is CPU-fed, not pulling), HL02 D **victim CPU BAR push under ongoing XTX 0 pull** arm, and HL02 E primary write-only trigger control; record these as distinct questions. Do not use a >=90%-of-idle GPU-pull threshold for CPU BAR push.
2. Gate: R9700 and both XTX CPU-write paths pass full-size (4 KiB to 10 MiB), multi-generation full-word and **GPU-already-waiting** DMA-BUF verification, including cross-process mapping, ordering, timeouts and buffer reuse. Reconcile RAD50's historic R9700 failure with the latest receipts before saying this passes.
3. Host-side microbench (no engine): 3 x S bf16 partials (S = 320 KiB .. 10 MiB) summed with AVX2 in the GPU kernel's order and precision, written to three DMA-BUF targets; measure CPU time, memory bandwidth (record actual 3S GPU-to-host partial writes, 3S CPU input reads, 3S CPU BAR target writes, and any S intermediate output, rather than assuming a fixed ~60 MB memory-traffic total) and per-card arrival, in pieces so writes and reads overlap. Estimate both the serial (~3.35 ms R9700 link-only at 10 MiB from RAD40 rates) and ideal overlapped (~1.87 ms) floors, adding measured reduction and visibility costs.
4. Engine patch (opt-in, layout key, e.g. `tuning.all_reduce_host = prefill`): a host reducer thread per world, a GPU-side ready/done protocol reusing RAD50's corrected wall_clock64 waiter, and the existing two-shot as a **start-of-invocation** alternative; any in-flight timeout requires coordinated all-rank abort/drain before switching. Decode keeps the GPU one-shot (handshake-bound; RAD50 single-row push is slower).
5. Measure: rad0031 timeline, 2x ABBA vs current (prefill 512/2048/6144, four 4096 prompts, long prompt beside 4 streams, 1/4/6/8 streams), then the VTP37 reordering quality gate (identical bits expected if the CPU sum matches the GPU order; otherwise KLD vs BF16).

## Detailed Solution & Technical Design

Per piece of rows: each rank's GEMM epilogue (or a copy kernel) writes its partial to its pinned host slot and raises a flag. A CPU thread (one per piece stream, pinned to P-cores) waits for three flags, sums p0+p1+p2 in the GPU two-shot's order with fp32 accumulation over bf16 inputs, and streams the bf16 result with non-temporal stores into each rank's DMA-BUF-mapped VRAM receive buffer, then writes a ticket. Each rank's next kernel (norm/quant) waits on its ticket in VRAM (local polling, no PCIe reads). Pieces (as rad0032) **may** overlap GPU host writes, CPU sum and BAR pushes. For a 10 MiB message, RAD40's R9700 ~7.1 GB/s GPU write and ~5.6 GB/s CPU BAR push imply ~1.48 + ~1.87 = ~3.35 ms of serial link stages, or a best-case ~1.87 ms *fully overlapped* link floor, before CPU sum, visibility and handshakes. The current ~4 ms is a measured comparator, not proof that ~1.4-2 ms is achievable.

Alternative without the CPU sum (cheaper fallback if CPU memory bandwidth is the limit): hierarchical exchange. XTX 0 and XTX 1 exchange partials and add them; the R9700 writes its partial once and both XTX read it and finish; the XTX pair publishes XTX0+XTX1 (half each) and the R9700 reads that S and adds its own. R9700 moves S each way; all ranks add (p0+p1)+p2. It still has XTX ranks pulling, so it keeps the HL01 contention.


**Refined implementation and feasibility gates (2026-11-13; VTP38 owns TP3 host reduction, not HL02 or RAD50):**

- **Measured vs projected:** RAD40 gives XTX 1 CPU BAR push ~10.3 GB/s and R9700 CPU BAR push ~5.6 GB/s. For a 10 MiB (=10,485,760-byte) message, an R9700 host write at ~7.1 GB/s alone takes ~1.48 ms and a CPU BAR push at ~5.6 GB/s alone takes ~1.87 ms. If serial, those two link stages cost ~3.35 ms **before** reduction, readiness and scheduling. The best ideal *overlapped* lower bound is ~1.87 ms, not a guaranteed ~1.4-2 ms result. Measure PCIe duplex, CPU sum, memory traffic, GPU arrival skew and inter-piece pipeline before relying on any projected 2x reduction. CPU input 3S, output to three VRAMs 3S and potential intermediate output S are **in addition** to three GPUs' S host writes; count actual bytes and cache/reuse rather than treating ~60 MB as a settled total host-memory traffic budget.
- **Two different HL02 gates:** C tests the hypothesis that XTX 1 GPU pulls recover when XTX 0 stops pulling and is instead CPU-fed. D tests whether a victim's CPU BAR push improves real makespan while XTX 0 **continues** pulling. VTP38 consumes both observations but cannot use C's >=90% *GPU-pull* recovery threshold as an acceptance criterion for CPU BAR push (which has a lower standalone rate). Its direct viability gate is the full TP3 push-plus-reduction pipeline.
- **Device support:** Reproduce CPU-accessible GPU-local VMM allocation and export via DMA-BUF/SCM_RIGHTS, persistent mmap, CPU SYNC START/END, per-device identity and 4 KiB/4 MiB/10 MiB full-word verification on *both* XTXs and R9700, including a GPU consumer launched **before** the CPU writes. RAD50's historic R9700 canary failure and later notes must be reconciled with the actual recent full-size live verification; serial results alone are insufficient.
- **Arithmetic correctness:** Reconstruct the current two-shot TP3 accumulation order, intermediate precision and BF16 conversion/rounding; don't assume that FP32 sum of three BF16 values is bit-identical to the GPU's shardwise summation. Byte-identical broadcast across ranks establishes *rank agreement*, not equivalence with the current TP3 output. First run a card-free per-element reference oracle (signed zeros, infinities, NaNs, ties, overflow/subnormals) and exact-wire comparisons. Then use VTP37 KLD/top-1/PPL only where a demonstrably different reduction order is intentionally admitted.
- **Publication and lifetime:** GPU partial-to-host stores must become CPU-visible before reducer reads; the reducer must not publish a chunk until CPU writes to each destination VRAM have become GPU-visible. A CPU release fence, DMA_BUF_IOCTL_SYNC END, or a host-ready word **alone** is not sufficient evidence for the device-live happens-before relation (RAD50's earlier 768-row GPU-live timeout is a warning). Use persistent per-rank GPU-local receive buffers with generation/slot/shape identity, separately tracked producer-ready, reducer-completed, rank-consumed and safe-to-reuse states. Verify delayed posted writes, stale flags, GPU already polling, wraparound, producer/reducer death, teardown/restart and missed ACK without hanging.
- **Scheduling:** Start with an isolated lab of host reduction plus real CPU BAR pushes; measure serial and chunked pipelined alternatives, including 256-row pieces, before fusing the producer GEMM epilogue or modifying production. Keep reduction and transfer on the owning Radiance collective path, never build a parallel runtime engine. CPU affinity and contention with serving threads must be measured. Retain existing GPU two-shot/one-shot paths for controls and unsupported hardware.
- **Safe fallback:** A timeout in a live collective cannot simply rerun the same operation through the old two-shot while other ranks may have consumed different data. Fail closed for that generation, coordinate a whole-world abort/drain or restart, and only choose the original protocol for a subsequent clean invocation. Do not silently switch paths with in-flight buffers. No default enablement until ABBA, exactness/quality, and bounded fault-injection gates pass.

## Code Samples & Guidance



## Files

engines/radiance/patches (new package), libr4d r4d_ar_entry.hip / r4d_ar_twoshot_Nrank_exact.hip (fallback), arch/qwen35_fp8/qwen35_fp8.cpp (pipelined prefill AR call sites), tools/lab/radiance/rad50-tester (HL02 arm), engines/radiance/layouts/qwen27b.toml.

## Validation

Timeline: no rank waits > 10% of the all-reduce at any barrier; R9700 link time within 20% of S / measured x4 rate per direction. 2x ABBA: prefill 2048 TTFT improves >= 10% with decode within noise. Quality: VTP37 reordering gate (noise floor vs serial, then mean/p99.9 KLD within +10%, top-1 within -0.2 points, PPL within +0.1% vs BF16). Stress: 12 clean runs, no stale tickets.

## Effort & Risk

L. Risks: CPU memory bandwidth (~60 MB of traffic per prefill all-reduce at 2048 tokens); R9700 BAR path unproven; GPU-visible readiness of CPU writes (RAD50 needed a corrected wall_clock64 watchdog on gfx11); one more host thread on the critical path; HL04's no-go was a different case (combine exposed only 0.1-0.2 ms).

## Standards



## Acceptance Criteria

Opt-in layout key, generic (no model or card constants); default unchanged until the ABBA and quality gate pass; fail closed on a protocol timeout and only select the GPU two-shot after a coordinated drain/restart or on a later clean invocation.

## Notes

Created 2026-11-13 from the VTP36 all-reduce timeline review and the HL01 lspci/IOMMU finding. This is the sole owner of TP3 CPU-reduced all-reduce integration; VTP36 retains programme-level A/B reporting, HL02 owns cross-GPU contention A/B, RAD40 the historical physical-link harness and RAD50 the generic DMA-BUF/visibility proof. Depends on HL02 and RAD50 gate 2 (R9700). Related: VTP36 (two-shot pieces/duplex, uneven XTX shares), RAD50/RAD51 (CPU BAR push lab), RAD40 (contention measurement), HL04 (different-layout no-go).

## Change Log

- 2026-11-13T05:14:45.261435+00:00 (created-by): Created by agent
- 2026-11-13T05:15:16.842115+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:acceptance_criteria, section:notes
