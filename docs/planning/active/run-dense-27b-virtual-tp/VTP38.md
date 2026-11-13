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

# Production TP3 per-rank host transport: CPU push and optional ordered host reduction

## Description

**Single production owner** for the native Radiance TP3 host-staged all-reduce receive protocol. VTP36 owns programme performance, current GPU collective candidate and split/quality; HL02 owns contention, host-memory and concurrent transfer measurements; RAD50 owns reusable DMA-BUF publication and safety evidence.

The rad0031 weighted two-shot (~10 MiB, 2048-token pipelined prefill) measured ~4 ms per collective, including R9700 x4 link transfers and an XTX reduction skew of ~889 versus ~1301 us due to HL01 root-port contention. For that **specific two-shot protocol**, with message S and R9700 shard fraction w=1/9, it sends ~S and receives ~(1+w)S. A one-shot inbound read is ~2S. These are protocol-specific counts, not lower bounds on *every* exact GPU algorithm.

**Preferred prefill end state:** one consistent CPU-fed receive path for **all** ranks. Every GPU publishes an S partial to pinned host memory; CPU reduces the three partials in the proven arithmetic order, and separate CPU workers push an S result to every GPU-local DMA-BUF destination. No rank GPU-pulls host memory on that collective. **Fallback:** CPU push to the contention-driving receiver only if measured CPU/memory bandwidth constrains all-rank push, with an explicitly compatible world-wide GPU reduction/receive schedule. Decode retains GPU one-shot absent a separate gain.

**Current comparator:** The unmerged VTP36 rad0032 fused-pieces + rad0033 write-once + duplex-weights candidate, `nblocks=4,nthreads=256,ts_pieces=4,ts_fuse=1,ts_once=1,ts_duplex=1`, produced bit-identical logprobs and 2 completed ABBA comparisons: int8/MXFP4 TTFT 2048 -24.4%/-27.2%, 8-stream +13.9%/+16.7%. Stress and 5,5,6 split checks remained unfinished. Host reduction must beat the **validated improved GPU candidate**, not claim those previous gains again.

## Steps

1. Pin the current VTP36 rad0032/rad0033 GPU collective, exact wire/split/tuning and rad0031 phase timing; finish its pending stress before promoting this baseline.
2. HL02 Gate A **HIP attribute NO-GO as of 2026-11-13:** contended mapped-host GPU reads/writes from coherent, non-coherent and write-combined allocations all retained the ~50% victim slowdown; no production benefit from changing those flags alone. PCIe No-Snoop bits were not observed. The preferred next comparison is still HL02's real concurrent CPU-push-all versus push-to-one, with the improved GPU collective as control.
3. HL02 Gate B: at 320 KiB–10 MiB compare all-three CPU push and push-to-one while GPUs concurrently publish partials. Record CPU memory bandwidth, P-core time, per-rank arrival and slowest collective completion; start one push thread per GPU and measure whether two per x8 help.
4. HL02 Gate C: qualify XTX and R9700 actual CPU-write DMA-BUF paths to 10 MiB, repeated live-ready generations, pre-armed GPU full-word checks, fault, timeout and reuse. The original R9700 HSA-pool canary failure was not a DMA-BUF failure.
5. Introduce generic validated per-rank `tuning.all_reduce_recv = ["cpu_push", "pull", "pull"]` with enum `pull|cpu_push|host_reduce`, world-size check, immutable generation protocol and optional measured host-profile selection. Mixed methods require a proven world-wide compatible reduction schedule; no BDF/device literal in engine code.
6. Implement opt-in prefill host transport as a managed Radiance patch: persistent pinned host partials and GPU-local DMA-BUF targets, fixed-order CPU FP32 reduction with actual GPU BF16 rounding comparison, persistent CPU per-card push workers, GPU-local generation/ticket readiness and `wall_clock64()` bounded waiter.
7. Pipeline GPU write piece k+1 / CPU reduce k / CPU push k-1 using existing VTP36 pieces. Evaluate measured all-rank and push-to-one schedules; hierarchical GPU exchange remains an optional tested fallback if CPU sum is limited.
8. Choose fallback before invocation where unsupported. On in-flight timeout, coordinate all-rank abort/drain, invalidate the generation and only then retry the GPU two-shot on a fresh generation; never publish both paths or double-reduce.
9. Run 2x exploratory, 3x final ABBA and VTP37 arithmetic/quality gate, 512/2048/6144 or 8192 TTFT, four 4096 prompts, long prefill beside 4 streams, 1/4/6/8-stream decode, 12-run int8 stress and fault cases.

## Detailed Solution & Technical Design

**Typed per-rank contract.** `pull` consumes a qualified GPU host-pulled receive payload. `cpu_push` consumes CPU-delivered GPU-local data without automatically changing arithmetic. `host_reduce` participates in a globally defined ordered CPU reduction of all rank partials. The array is world-size matched and fixed for the whole generation. Mixed methods are rejected unless all ranks' producer, reduction, delivery, ACK and once-only semantics are proven compatible. The desired *uniform* `host_reduce`/CPU-push path eliminates rank GPU host pulls; push-to-one is a measured protocol alternative, not an arbitrary per-rank pointer substitution. Use a measured topology/profile to propose methods, not machine-specific constants.

**Byte budget:** all-rank host sum requires 3S GPU writes to host, 3S CPU reads and 3S CPU BAR result writes, plus possible S temporary. For S=10 MiB, R9700 measured ~7.1 GB/s GPU write and ~5.6 GB/s CPU push imply ~1.48+1.87=3.35 ms *serial* link components, or ideal fully overlapped ~1.87 ms link-only floor before CPU sum, fences and arrivals. These are estimates, not a TP3 host-reduced measurement. CPU FP32 `(p0+p1)+p2` can differ from GPU two-shot if intermediate conversion/rounding is different: signed zero, NaN, inf, ties and subnormal oracles and VTP37 numerical checks decide whether bit identity is supportable.

**Safety:** GPU partials must become CPU-visible before host read, and CPU DMA-BUF writes must be GPU-visible before local ready-ticket publication; CPU fences or `DMA_BUF_IOCTL_SYNC END` alone do not prove application-level happens-before. Reuse RAD50's device-scaled gfx11 `wall_clock64()` waiter and separate host timeout. Generation, slot, world/run, piece and layout identity remain bound through acknowledgment and retirement. Any failed in-flight generation requires global abort/drain before GPU fallback.

**History — original design and alternatives:**

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

HL02 coherent/non-coherent, true CPU-push and three-way concurrent gates; RAD50 cross-process live-ready full-size proof on all target GPUs. Independent exact-word and generation tests for posted-write completion, pre-armed consumer, ring wrap, faults, timeouts and cleanup. Compare against the validated VTP36 rad0032/rad0033 baseline using rad0031 phases and CPU memory/P-core measurements; report p50/p95/p99. At least 2 preliminary and 3 final matched ABBA trials. Exact logprobs only where the CPU sum reproduces GPU order/rounding; otherwise VTP37 reorder/KLD/top-1/PPL gate. Decode stays GPU one-shot unless separately justified; require 12 clean stress runs.

## Effort & Risk

L. Risks: CPU memory bandwidth (~60 MB of traffic per prefill all-reduce at 2048 tokens); R9700 BAR path unproven; GPU-visible readiness of CPU writes (RAD50 needed a corrected wall_clock64 watchdog on gfx11); one more host thread on the critical path; HL04's no-go was a different case (combine exposed only 0.1-0.2 ms).

## Standards



## Acceptance Criteria

- [ ] Generic per-rank `pull|cpu_push|host_reduce` layout/profile contract with no machine/card/BDF constants.
- [ ] HL02 all-three CPU push and push-to-one concurrent producer/consumer capacity and critical-path matrix complete.
- [ ] XTX/R9700 10 MiB GPU-live DMA-BUF safety, lifetime and bounded failure path passes.
- [ ] Ordered host arithmetic is either bit-exact to current GPU two-shot **or** passes the explicit VTP37 reordering quality gate.
- [ ] >=10% better 2048-token prefill TTFT against the **validated improved VTP36 GPU candidate**, or a measured NO-GO retaining the faster baseline.
- [ ] Decode at GPU one-shot parity, 12 clean stress runs, no stale/duplicate generations.
- [ ] Timeout aborts and drains all ranks before any fresh-generation GPU fallback; no unsafe mid-flight switch.
- [ ] Default unchanged until repeated ABBA, quality, stress and explicit approval.

## Notes

**2026-11-14 host-write safety progress, not TP3 protocol qualification:** Archived RAD50 GPU-local DMA-BUF direct publication was independently repeated with pre-armed, full-word GPU verifier for 100 generations **per card idle** and 100 generations **per affected XTX/R9700 under active primary-XTX host reads at 13.431 GB/s**, all passing for 3.93 MiB payloads. The same 4 MiB serial GPU pull vs CPU BAR push comparison at 30 iterations showed XTX 1.847× and R9700 2.315× advantage under contention. Evidence: `docs/evidence/2026-11-14-hl02-live-verification/README.md`. This does not satisfy 10 MiB, parallel multi-destination CPU sum/push, lifetime/restart, generation-wide abort/drain or matched VTP36 A/B. No VTP38 production method change is authorized.

**New HL02 lab measurement (2026-11-13):** Three-session mapped-host attribute/trigger matrix and 4 MiB RAD50 serial pull/push comparison are recorded in `docs/evidence/2026-11-13-hl02-transfer-matrix/README.md`. Coherent, non-coherent and write-combined HIP allocation flags all yielded ~half victim GPU host-transfer bandwidth under primary GPU reads. Primary GPU write-only controls showed no slowdown. Under primary reads, the serial 4 MiB CPU push retained approximately idle latency while GPU pull slowed (XTX push-vs-pull 1.85x, R9700 2.30x session-median speedup). No triple concurrent CPU BAR push or host sum was tested, so the production admission gate remains closed.

**New partial R9700 gate (2026-11-13):** Existing RAD50 GPU-live tester verified a 3,932,160-byte pre-armed R9700 DMA-BUF payload across 30/30 generations, no mismatches, exit 0; see `docs/evidence/2026-11-13-hl02-vtp38-consolidation/R9700-LIVE-2026-11-13.md`. This does **not** satisfy the all-rank 10 MiB Gate C or the CPU host-sum protocol.

**2026-11-13 status:** The VTP36 rad0032/rad0033 two-ABBA GPU candidate is a measured starting point, *not* a merged production host-reduce improvement; stress and split validation remained. RAD50's apparent 768-row gfx11 `clock64()` live-waiter failure was fixed with `wall_clock64()`, and later R9700 DMA-BUF tests passed smaller sizes; full concurrent 10 MiB remains unproven. Proven/estimated quantities and source run names are indexed in `docs/evidence/2026-11-13-hl02-vtp38-consolidation/README.md`. No host-reduced TP3 engine hardware run has been performed by this item.

**History — original plan and refinements:**

Created 2026-11-13 from the VTP36 all-reduce timeline review and the HL01 lspci/IOMMU finding. This is the sole owner of TP3 CPU-reduced all-reduce integration; VTP36 retains programme-level A/B reporting, HL02 owns cross-GPU contention A/B, RAD40 the historical physical-link harness and RAD50 the generic DMA-BUF/visibility proof. Depends on HL02 and RAD50 gate 2 (R9700). Related: VTP36 (two-shot pieces/duplex, uneven XTX shares), RAD50/RAD51 (CPU BAR push lab), RAD40 (contention measurement), HL04 (different-layout no-go).

## Change Log

- 2026-11-13T05:14:45.261435+00:00 (created-by): Created by agent
- 2026-11-13T05:15:16.842115+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:acceptance_criteria, section:notes
