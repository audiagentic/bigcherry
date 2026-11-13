---
id: RAD50
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-11-08T21:44:12+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P2
---

# Prove CPU BAR-push MoE result transport in isolated, gated harnesses

> Archived 2026-11-12 from branch `exp/rad42-cpu-bar-push` (git tag `archive/exp-rad42-cpu-bar-push`), which never merged. The lab programs (tools/lab/rad42-bar-push) are in that tag; the lab notes and raw A/B logs are in docs/evidence/2026-11-10-rad50-cpu-bar-push.

## Description

**Ownership clarification 2026-11-13:** RAD50 remains pending as the reusable **isolated DMA-BUF/CPU-write proof and safety** owner, not the remaining cross-GPU transfer-method experiment owner. RAD40 is superseded, HL01 diagnosis is completed; **HL02** owns all outstanding contended pull/SDMA/CPU BAR and concurrent three-rank push measurements. **VTP38** owns any production TP3 host transport; RAD42/RAD51 retain MoE result-path integration only. Preserve RAD50's serial/live mapping, arithmetic, fault and lifetime gates as independently required for each consumer. Later R9700 DMA-BUF successes supersede neither the original inconclusive HSA-pool canary nor the unproven full-size TP3 concurrency.

**Historical original scope:**

RAD42 design B proposes CPU-writing the CPU-combined remote-MoE FP16 result
plane into engine VRAM through a CPU-accessible BAR, then letting the GPU
consume locally instead of pulling the result plane from host RAM. RAD40's
host-link measurements motivate this under XTX0 host-read contention, but
the standalone CPU BAR push rate is *lower* than an uncontended GPU pull.
The advantage is not established for inference. This item owns the
**experimental feasibility and safety gates only**, before any Radiance
or moe-remote runtime modification. RAD42 remains the owning production
design item; RAD49 is a future TP2/consumer-rank integration concern.

## Steps

1. Card-free model: generation-scoped payload ownership and distinct
   GPU-produced, CPU-write-completed, GPU-consumed states. Negative cases
   for stale epochs, early publication, early recycle and faults.
2. Hardware discovery: map PCI identity to the engine GPU and enumerate
   GPU-local allocations whose CPU access the HSA runtime permits.
   An allocatable pool is not proof of a functional BAR mapping.
3. Single-process CPU-write -> GPU-read canary on 4 KiB then 4 MiB,
   with fully checked payload, repeated generation patterns, no
   streaming and no interprocess sharing.
4. Before bandwidth tests, validate GPU-local memory sharing across
   a **separate CPU-only helper process** via HSA IPC and confirm that
   the importer can write without enabling GPU peer access. Serial
   verification is only an IPC feasibility gate, not device-live ordering.
5. Measure CPU BAR push and mapped-host GPU pull on the same destination
   and payload, across sizes, thread counts and contention arms.
   Record CPU utilisation, p50/p95 latency, time to first chunk and
   final completion, not bandwidth alone.
6. Prove per-chunk publication and visibility in a bounded, separate
   producer/consumer harness. Use distinct ready and consumed generations;
   timeout and fail closed. Run the 128/256/512/1024-row matrix.
7. Reproduce the real CPU-combine FP16 semantics (overflow sentinel
   fallback and consumer-rank/once-only ownership) on representative
   captured rows, without a serving engine.
8. Only if all gates pass, propose a new, separately reviewed
   opt-in moe-remote transport branch. No production edits,
   default change, push or PR are authorised by this item.

## Detailed Solution & Technical Design

The engine GPU may continue issuing work and request tickets. A helper's
GPU produces output and writes host-visible result planes. The existing
CPU combiner may assemble a complete chunk. The proposed *bulk* transfer
is a CPU write to GPU-local memory accessible via the BAR, followed by
a tested visibility action and a separate ready generation that the GPU
checks before consuming. Request publication is **not** evidence that
the CPU's BAR writes have completed. The consumer must acknowledge before
slot reuse. Neither a CPU fence nor an HSA pool-access flag alone proves
cross-agent visibility or ordering; the hardware canary and chunk
protocol must establish those facts.

Retain the existing mapped-host GPU-read transport as the control and
fallback. Preserve FP16 combine numerical order and overflow fallback.
Do not assume direct GPU-to-GPU peer access on this host. The test must
distinguish engine ranks from helper slots as RAD49 evolves. Never
substitute a CPU-round-tripped transfer for the GPU-driven decode path
without measuring its added wakeup and dispatch latency.

**Stop/go:** Stop on any stale-read, data mismatch, unsupported CPU mapping,
unbounded wait or GPU fault. For production consideration, first require
at least 15% improvement in contended transfer makespan or at least
0.3 ms reduction of the layer-step critical path, then at least 2%
end-to-end improvement and no p99 regression over 5% with quality parity.

## Code Samples & Guidance

The isolated harness and gate log live in
`tools/lab/rad42-bar-push/README.md`. The single-process canary may
use HIP kernels and HSA GPU-local memory allocations, but the destination
must be proven to reside in GPU memory, be CPU-addressable, and match
the selected GPU's PCI identity. Every failure should yield a nonzero
exit status with a reported reason. The actual cross-process export,
mapping, lifetime and visibility method is deliberately deferred until
the single-process gate passes.

## Files

- `tools/lab/rad42-bar-push/chunk_contract.py`
- `tools/lab/rad42-bar-push/test_chunk_contract.py`
- `tools/lab/rad42-bar-push/pool_inventory.cpp`
- `tools/lab/rad42-bar-push/bar_write_canary.hip` (Gate 2)
- `tools/lab/rad42-bar-push/critical_path_model.py` (pure scenario model)
- `tools/lab/rad42-bar-push/cpu_handoff_bench.cpp` (host-only lower bound)
- `tools/lab/rad42-bar-push/bar_ipc_canary.hip` (Gate 2b, no peer GPU)
- `tools/lab/rad42-bar-push/bar_transfer_ab.hip` (Gate 3, not run)
- `tools/lab/rad42-bar-push/bar_live_ready_canary.hip` (Gate 4 single-process canary, not run)
- `tools/lab/rad42-bar-push/combine_reference_test.cpp` (reference parity)
- `tools/lab/rad42-bar-push/PROGRESS.md`, `RUNBOOK.md` and `FINDINGS.md`
- `tools/lab/rad42-bar-push/README.md` (per-gate progress and decisions)
- Subsequent isolated lab harnesses only after the preceding gate passes.

## Validation

Card-free unit tests for lifecycle ordering and fault conditions;
`g++` and `hipcc` compile-only checks in the experimental worktree.
Hardware tests must use an exclusive GPU lease and identify the physical
GPU, ROCm release, source commit, payload size, pass/fail and repeat count.
Record both an uncontended control and the RAD40 XTX0-contention arm.
Benchmark comparisons must keep the same payload, CPU pressure and GPU
link conditions and report critical-path makespan, not just aggregate GB/s.
A wrong result, hang, fault or consumer-gate violation is a failure.

## Effort & Risk

M experimental scope, advanced synchronization/hardware semantics.
Risks: BAR visibility and coherency, CPU/GPU pointer accessibility,
cross-process GPU memory export, VRAM buffer lifetime, CPU-copy overhead,
PCIe posted-write ordering, TP2 consumer-rank identity and disabled
peer-to-peer transport. Keep serving untouched while uncertain.

## Standards

Use project-local experimental worktree, one experiment directory under
`tools/lab/`, documented run identity, and explicit retention or
retirement disposition. Respect GPU leases, agent-owned branches,
the pinned vendor tree and card-free CI. No hardware run without a
lease; do not bypass unavailable lease controls.

## Acceptance Criteria

**Scope update:** Outstanding shared cross-GPU contention S6 is discharged by referenced HL02 receipts, not a duplicate RAD50 benchmark. RAD50 remains open for mapping/API-specific proof, live-ready publication, crash/restart, FD lifetime/reuse, correct MoE FP16 fallback and independent production admission. Do not count the 2026-11-09 synthetic serial gains as model performance, and do not treat the earlier gfx11 `clock64()` false timeouts as a live data-corruption proof.

**Original laboratory acceptance gates (historical wording below):**

1. Contract and GPU-accessibility gates explicitly passed or failed,
   with reproducible commands and evidence.
2. CPU-written GPU-local data verified exactly by a GPU on required
   payload sizes and repeated generations; negative failure behavior
   and device identity documented.
3. A separate push/pull and contention comparison with p50/p95 timing,
   CPU cost, chunk latency and hardware identity.
4. Per-chunk consumer safety and faults tested, no stale or duplicate
   consumption, no buffer reuse before acknowledgment.
5. MoE FP16/fallback and TP2 once-only equivalence verified against
   the existing reference on captured rows.
6. Any proposed serving integration gated on measured end-to-end
   benefit and explicit subsequent review; item remains pending until
   all acceptance criteria are met.

## Notes

**New isolated comparative control (2026-11-13):** Reused the existing RAD50 `dmabuf-ab` 4 MiB, 10-iteration exact GPU verifier under controlled primary-GPU read and write backgrounds in three separately leased sessions on XTX and R9700. Under primary read contention, serial synthetic CPU BAR-push vs mapped-host GPU pull speedups were XTX 1.85x, R9700 2.30x (session medians). Idle 1.38x/1.54x; primary write-only gave 1.35x/1.53x. Full measures in `docs/evidence/2026-11-13-hl02-transfer-matrix/README.md`. This strengthens a *transfer-method* result, not RAD50's unpassed fault/lifetime or VTP38's three-rank/10 MiB/production gates.

**New repeat (2026-11-13):** Archived `rad50-tester` on R9700 passed the pre-armed `dmabuf-live-parallel` 768-row full-word check for 30/30 generations, zero mismatches, exit 0 (3,932,160 bytes), with GPU lease released; new managed evidence at `docs/evidence/2026-11-13-hl02-vtp38-consolidation/R9700-LIVE-2026-11-13.md`. Still not a 10 MiB three-way, crash/lifetime or inference test.

**Latest evidence reconciliation:** The 2026-11-09 GPU-live 768-row failures were diagnosed as a gfx11 `clock64()` false deadline; replacing it with HIP device-rate-scaled `wall_clock64()` enabled 100 live generations with a pre-armed GPU verifier on each XTX. On 2026-11-10 the R9700 *DMA-BUF* mapping, 4 KiB/4 MiB GPU-verification, 30-generation live and 100-iteration serial combine A/B succeeded, although the older R9700 **HSA-pool serial canary** remains inconclusive. Neither proves crash/restart/lifetime or 10 MiB three-way prefill safety. The 768-row R9700 direct serial numbers: 959.0 us GPU pull versus 770.4 us CPU push (~1.25x); one row 74.2 versus 76.7 us (push loses). See `docs/evidence/2026-11-10-rad50-cpu-bar-push/PROGRESS.md` 2026-11-09/10 entries and `docs/evidence/2026-11-13-hl02-vtp38-consolidation/README.md`. HL02 owns new contention S6, not RAD50.

**History — original laboratory notes and gate log:**

Related: RAD40 (link measurements), RAD42 (production design B),
RAD51 (future production integration plan, blocked on RAD50),
RAD49 (TP2 engine and remote experts). Start from the existing
`exp/rad42-cpu-bar-push` lab worktree, not production sources.
Do not claim CPU BAR push is faster without the complete comparison.

**Ownership clarification (2026-11-13):** RAD50 remains the owner of generic CPU-accessible VRAM/DMA-BUF export, cross-process lifetime, GPU-live visibility, fault and MoE FP16 correctness canaries. HL02 owns the remaining matched multi-GPU contention transfer matrix using RAD50 lab assets. VTP38 owns TP3 production host reduction and RAD42/RAD51 own MoE integration. Do not treat a serial, GPU-later or idle result as full-size, GPU-already-waiting, live cross-process proof without exact receipts.

## Change Log

- 2026-11-09 (updated-by): Prepared one fixed-mode RAD50 test executable
  (`rad50-tester`, dual gfx1100/gfx1201) and a source-based Makefile,
  with strict argument parser and card-free self-check. Existing HSA
  inventory, serial canary, IPC, A/B, live-ready, CPU handoff and FP16
  reference implementations are compiled directly into one binary;
  there is no command forwarding to separate test executables. Build
  and link pass; the one `rad50-tester` executable is allowed and
  passes 19/19 CLI self-checks and seven FP16 reference checks. No
  GPU run or serving change made for this update.


- 2026-11-08T21:44:12+00:00 (created-by): Created as an isolated
  proof-first experiment item; initial Gate 0 passed 10 card-free tests.
- 2026-11-08T22:01:00+00:00 (updated-by): Gate 0 extended to 13 card-free tests;
  inventory and single-process GPU canary compiled for two GPU architectures.
  Lease command is now available, but RAD49 holds the exclusive GPU lease.
  Hardware Gates 1 and 2 remain unrun; no performance or coherence claims.
- 2026-11-08T22:27:00+00:00 (updated-by): Added model of already-overlapped CPU combine and GPU pull, 10 more card-free tests, host-only wakeup probe and GPU canary/IPC/transfer microbenchmark binaries. 23 Python tests pass; no GPU run. Actual serving integration is described in new pending RAD51, blocked by these gates.

- 2026-11-08T23:50:00+00:00 (updated-by): Checked all four GPUs idle, acquired the exclusive GPU lease and inspected ROCm devices/pools. The remote command policy rejected running the separately compiled `pool_inventory` executable. No BAR or GPU canary ran. Released the lease promptly. Gate 1 remains unverified; explicit executable allowlisting is required.

- 2026-11-08T23:55:00+00:00 (updated-by): Hardware Gate 1 API discovery PASS under exclusive RAD50 lease after fixing non-global HSA pool enumeration. XTX0/XTX1: two CPU-accessible GPU-local pools each; R9700: one; RX 6900 XT: two. Gate 2 blocked by missing `bar_write_canary-gfx1100` and `bar_write_canary-gfx1201` executable permissions; no canary ran. Lease released. CPU BAR writes, GPU correctness, IPC and latency remain unverified.

- 2026-11-09 (updated-by): Gate 2 serial CPU BAR/GPU read **passed on both RX 7900 XTX cards**: 4 KiB and 4 MiB each, 12 generations each, every word checked, exits 0. R9700 `gfx1201` canary exited 1 without output, leaving its correctness unresolved. Added lab-only early diagnostic markers and recompiled R9700 probe without retesting. Exclusive GPU lease released. Cross-process mapping, live ready signalling and speed are still unverified.

- 2026-11-09 (updated-by): R9700 gfx1201 exit-1 narrowed to the first HIP GPU-verification kernel launch; HSA VRAM CPU allocation/stores and HIP error-counter memset completed, but launch did not return. No GPU result was read and the failure mechanism is unresolved. Compiled a HIP-only smoke diagnostic without rerunning it. XTX IPC executable remains disallowed, and subsequent GPU lease was obtained by HD2701. RAD50 did not retain a lease or run further GPU tests.

- 2026-11-09 (updated-by): Hardware: R9700 HIP-only smoke PASS;
  XTX0/XTX1 GPU-agent HSA IPC import PASS, but CPU-agent import
  fails 4104 OUT_OF_RESOURCES and the GPU-first imported pointer
  has no writable CPU VMA. Guarded CPU-store probe refused writes,
  with no GPU readback. VMM GPU-local DMA-BUF export/map PASS on
  XTX0; CPU get_access fails 4100 INVALID_AGENT. No actual
  cross-process CPU BAR payload verified; Gate 2b blocked.
  Added explicit diagnostics and fixed-mode VMM capability probe
  to the unified tester; no model/serving changes.

- 2026-11-09 (updated-by): **New cross-process positive result** through Linux DMA-BUF (not HSA IPC attach): both XTXs' GPU-local ROCr VMM allocations exported and passed to separate CPU-only helper via SCM_RIGHTS; mmap and DMA_BUF_IOCTL_SYNC start/end succeed; CPU writes **4 KiB** and **4 MiB**, and parent GPU verifies every 32-bit word for each, exit 0. One generation per size/card, **Gate 2b partial pass** pending repeated generations, fault/lifetime behavior and concurrent readiness. R9700 remains unverified. No speed or serving claim. GPU lease released after runs; no new worktree.

- 2026-11-09 (updated-by): Gate 3 performance comparison now compiled into the *existing* RAD50 tester as `dmabuf-ab`; persistent CPU-only child and matched mapped-host GPU-read baseline versus DMA-BUF VRAM push, 5 payload sizes, 10..200 repetitions, direct/staged CPU store variants, full GPU word verification, stage costs and serial p50/p95/p99. Companion card-free log analyzer added; 34 CLI checks and 32 Python tests pass. **No hardware speed run or speedup result** because RAD49 holds exclusive GPU lease and cards are active. Refer to `tools/lab/rad42-bar-push/PERFORMANCE.md`; keep RAD51 blocked until concurrency, real combine and paired inference evidence.

- 2026-11-09 (updated-by): Gate 3 `dmabuf-ab` now supports an optional bounded per-size `--bytes` selector for short safe runs under the remote shell time limit; 4 KiB/64 KiB/256 KiB/1 MiB/4 MiB allowed, with explicit lease, device and iteration controls. Unified benchmark rebuilt (OPT=-O1), card-free CLI tests 39/39, Python tests 32/32. RAD49 continues to hold the GPU lease; **no actual A/B timing result** yet. Once free, OPT=-O2 build then XTX0/XTX1 4 KiB smoke before expanding per-size measured pairs.

- 2026-11-09 (updated-by): **Gate 3 first completed measured serial A/B** under exclusive GPU lease: two XTXs each pass all GPU full-word checks for 5 payload sizes, 30 measured iterations plus 3 warmups, direct CPU output vs staged CPU memcpy. At 4 MiB direct CPU VRAM-push vs mapped-host GPU-pull: 1.414x on XTX0 (1075.16 vs 760.30 us p50), 1.362x on XTX1 (1139.85 vs 836.90 us). At 4 KiB CPU-push is slightly slower; staged 4 MiB benefits only ~1.11x. DMA-BUF SYNC per 4 MiB chunk ~0.5–0.7 us combined median. **Only isolated serial synthetic producer and GPU verifier; no actual MoE/serving speedup, contention, concurrent ready or TP2 evidence.** Raw logs and full analysis in tools/lab/rad42-bar-push/GATE3_RESULTS.md. User-requested partial logs preserved and --provisional diagnostic mode added; final parser still requires successful completion. Released GPU lease; branch remains lab-only.

- 2026-11-09 (updated-by): **Gate 3b: production `combine_row_f16` direct into GPU-local DMA-BUF demonstrated on both XTXs.** Persistent CPU-only helper performs real AVX2/F16C 2-helper FP16 combine, including masked/overflow fallback/NaN BF16 test profiles, while a separate GPU oracle checks every FP16 word. 1–768 rows × 30 generations and 768 rows × 100 generations pass per GPU with reused buffers. 100-iteration 768-row direct versus host-mapped GPU pull: XTX0 700.87 vs 621.96 us (1.127x); XTX1 703.44 vs 622.16 us (1.131x). Actual combine to GPU VRAM costs ~50 us more CPU work, but GPU consumption saves ~128 us, net 79–81 us serial. Short 30-sample large cases vary and are preserved. Single-row push loses ~1–2 us. Synthetic BF16 inputs, full-word verifier rather than Radiance consumer, **not an inference speedup**. See `tools/lab/rad42-bar-push/GATE3_COMBINE_RESULTS.md`; Gate 4 concurrent readiness, helper crash/restart and TP2 inference remain required. Existing worktree only; no serving edits.

- 2026-11-09 (updated-by): **Gate 4 GPU-live PARTIAL PASS, blocked at 768 rows.** The CPU-only helper computes production FP16 into GPU-local DMA-BUF while a GPU verifier, launched before CPU writes, waits for a generation-ready flag and checks every FP16 word before ACK/reuse. Both XTXs passed 10 generations each at 1, 16 and 256 rows. XTX0 failed on the first 768-row generation twice: initially with 256 GPU host-control polling lanes and again with single-lane polling; CPU observed ready=868001 but GPU observed ready=0. This is a ready-visibility failure, not a confirmed FP16 mismatch. A separate hipExtHostRegisterUncached control-plane candidate (`dmabuf-live-fg`) is compiled but **not GPU-tested**. Gate 4 is not passed, no inference gain or memory-order guarantee established. Complete successes and both failures retained under `tools/lab/rad42-bar-push/evidence/2026-11-09/`, see `GATE4_LIVE_RESULTS.md`. No serving changes authorised.

- 2026-11-09 (updated-by): **Gate 4 LIVE cross-process GPU-ready BLOCKER.** New fixed `dmabuf-live` launches GPU verifier before separately spawned CPU helper writes real production AVX2/F16C FP16 output into GPU-local DMA-BUF, calls SYNC END, publishes generation via shared HIP-registered host memfd and awaits GPU done. Both XTXs pass 10 generations at 1/16/256 rows. XTX0 768-row **FAILED twice** on first generation with GPU timeout, not reported FP16 mismatch. First 256 GPU pollers, then one poller with diagnostics; in latter case CPU ready=868001 but GPU seen=0 (helper ACK at 872.71 us). Direct hipHostRegisterMapped host control cannot be trusted for all sizes. DO NOT enable RAD51. Prepared and compiled fixed `dmabuf-live-fg` mode using hipExtHostRegisterUncached extended fine-grained registration (AMD Linux except Navi4X), **not hardware tested**. If that fails, consider parent HIP-coherent relay or DMA-BUF-local flag after independent visibility proof. Full logs and alternative designs at `tools/lab/rad42-bar-push/GATE4_LIVE_RESULTS.md`. Released both RAD50 GPU leases; RAD49 subsequently took over. No new worktree/production changes.

## 2026-11-09 — Measured RAD50 optimization slices

The earlier 768-row live failures were false GPU timeouts: the gfx11 `clock64()` deadline fired even though CPU data had been published. A `wall_clock64()` deadline scaled using the HIP device wall-clock rate restored live correctness. Both XTXs passed 100 repeated 768-row generations with direct child publication and parent coherent relay, with a GPU reader armed before CPU writes, exact FP16 checks and per-generation acknowledgement. The parallel 256-block verifier cuts the 768-row diagnostic from ~7.4 ms to ~0.464 ms. CPU combine now dominates (~363 us); launch/arm (~24 us) and GPU post-ACK completion (~71 us) are next. Detailed stage budgets, optimization order, safety gates and source logs: `tools/lab/rad42-bar-push/SLICE_OPTIMIZATION.md`. These are lab-only results, not a general memory-model guarantee, failure-recovery proof or a measured Radiance inference speedup. RAD51 remains blocked on reliability, contention and paired TP2 model tests.

- 2026-11-09 (updated-by): **RAD50 Gate 4 correction and slice optimisation plan.** GFX11 `clock64()` was an unreliable GPU watchdog in the live canary. Device-local diagnostics showed false deadline exit; `wall_clock64()` scaled by HIP device wall-clock rate resolved 768-row false failures. Both XTXs now pass 100 generations at 768 rows with live waiter armed before CPU writes, direct CPU-child ready publication after DMA-BUF SYNC_END and independent parallel GPU FP16 verification; coherent parent relay also passes. Direct p50/p95 XTX0 463.87/596.41 us, XTX1 463.71/590.44 us; relay p50 469.06/471.06 us. The old single-block verifier spent ~7.38ms at 768 rows: a lab-consumer bottleneck, not the VRAM transport rate. CPU production FP16 combine ~363us dominates ~464us live path; GPU arm ~24us and host completion after ACK ~70us. Keep both control modes and mapped-host fallback; 100-generation lab correctness is **not** an engine memory-model, fault-lifetime or TP2 inference PASS. Deep optimisation design: `RAD50_SLICE_OPTIMISATION.md`, detailed evidence: `tools/lab/rad42-bar-push/SLICE_OPTIMIZATION.md`, `GATE4_LIVE_RESULTS.md`; all early failure/partial logs retained.
