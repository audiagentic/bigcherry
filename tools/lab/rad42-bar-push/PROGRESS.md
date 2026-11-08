# RAD50 — progress, gates and evidence

This is the **in-worktree experiment journal**, not an inference-performance
claim. The authoritative plan item is
`docs/planning/active/patching-radiance-rdna3/RAD50.md`; production design
remains RAD42, and RAD49 is the TP2 consumer context.

## Current gate table

| Gate | State | Evidence / decision |
| --- | --- | --- |
| 0: lifecycle and critical-path model, card-free | **PASS** | 23 tests: 10 lifecycle, 3 multi-chunk stress, 10 critical-path/overlap model; no GPU visibility implied |
| 1: host-visible GPU-local pool discovery | **BUILD PASS, RUN BLOCKED** | `pool_inventory.cpp` compiled with g++; lease command available but GPU lease held by RAD49, so no HSA hardware discovery performed |
| 2: CPU-write/GPU-read canary | **BUILD PASS, RUN BLOCKED** | `bar_write_canary.hip` compiled for gfx1100 and gfx1201; hardware correctness is unverified. Requires Gate 1 PASS and an exclusive GPU lease before launch |
| 2b: separate CPU-only IPC import | **BUILT, RUN BLOCKED** | `bar_ipc_canary.hip` compiled gfx1100/gfx1201; import/CPU write/GPU full verification not run. Requires Gates 1 and 2 |
| 3: isolated transfer A/B | **BUILT, RUN BLOCKED** | `bar_transfer_ab.hip` compiled for gfx1100/gfx1201. No actual bandwidth, CPU cost or latency result. Requires Gate 2b PASS |
| 4: chunked producer/consumer | **PARTIAL BUILD, RUN BLOCKED** | `bar_live_ready_canary.hip` compiled gfx1100/gfx1201: GPU already running before CPU writes. Multi-process chunk pipeline still not implemented. Requires Gate 3 |
| 5: captured MoE data / TP2 exact-once | **PREP ONLY** | `combine_reference_test.cpp` compiled against BigCherry; execution blocked by remote policy. Captured KLD/TP2 replay still pending |
| 6: opt-in engine integration | **NOT AUTHORISED** | Separate future review after Gates 0–5 and end-to-end A/B |

## Chronology

### 2026-11-09 — isolated scaffolding

- Created branch `exp/rad42-cpu-bar-push` and worktree under the
  project worktree root, based on `6f7de471a`.
- Commit `8b0e1da24`: lifecycle model, 10 passing unit tests,
  HSA pool-discovery source and phased README.
- Pool inventory built in a separate project work-root build directory;
  it was **not run** against the GPUs.
- No model or engine work, no GPU run, no push, no PR.

### 2026-11-09 — plan and next small harnesses

- Added pending `RAD50` to the existing
  `patching-radiance-rdna3` plan group, without rewriting RAD42.
- Added `bar_write_canary.hip` for **single-process, serial**
  CPU-write -> GPU-read exact-pattern verification at 4 KiB and 4 MiB
  (12 generations each); it matches HIP and HSA devices by physical PCI ID,
  requires an HSA GPU-local allocation accessible to a CPU, and fails
  rather than falling back to host RAM.
- Compiled canary for gfx1100 and gfx1201. No card was touched.
- remote shell returned `Command 'agent-lease' not in allowlist` when
  checking `agent-lease status`. This is a **hard run gate**; do not
  bypass it with a Python/subprocess wrapper or run hardware probes
  without a legitimate GPU lease.
- No conclusions about BAR mappability, coherence, speed or real
  concurrent generation publication can be drawn from offline builds.

### 2026-11-09 — lease access restored, RAD49 hardware work in progress

- The remote command policy now permits `agent-lease status`; the earlier command-permission blocker is resolved.
- At 2026-11-08 22:01 UTC, the exclusive `gpu` lease was held by
  another RAD49 measurement (expiry 2026-11-09 03:48 UTC).
- `amd-smi` remains disallowed; the permitted `rocm-smi` reported GPU
  0 and 1 at 0% use, and the R9700 / RX 6900 XT at 13% / 11% use,
  with approximately 94% / 95% VRAM allocated on the latter pair.
  Radiance and remote-MoE worker processes were visible.
- No hardware testing was started for RAD50. The lease covers all GPUs,
  even those currently at zero utilisation. Gates 1/2 remain build-only.
- Recheck GPU processes and acquire our own lease after RAD49 releases
  its lease before any HSA or HIP hardware probe.

### 2026-11-09 — next-stage preparation and production risk review

- Read the merged RAD49 v25 protocol from origin/main, without merging into this worktree.
- Added pending RAD51 for future opt-in BigCherry integration; no production edit.
- 23 card-free Python tests pass; the overlap model distinguishes serial CPU copy from direct VRAM output.
- Built host-only CPU wake timing and FP16 reference test binaries, but did not execute them.
- Built gfx1100/gfx1201 serial IPC import and CPU-BAR/GPU-pull A/B harnesses; NO GPU runs.
- See FINDINGS.md for ranked risks and a clearly labelled unmeasured illustrative scenario.
- See RUNBOOK.md for lease-safe ordered testing and required trace fields.
- A single-process GPU-live waiter correctness canary (`bar_live_ready_canary.hip`) also built for gfx1100/gfx1201; its test and multiprocess extension remain blocked.
- The GPU lease remains owned by the RAD49 measurement; Gate 1 onward is hardware blocked.

## Next action when a GPU lease is available

1. Acquire/check the project's exclusive GPU lease by its supported tool,
   identify physical PCI devices and verify there is no live serving.
2. Run `pool_inventory` **first**; preserve output with ROCm/kernel versions,
   PCI identities and selected GPU-local pool. If CPU access is not
   supported, STOP and record the negative result.
3. Only if the selected devices have an eligible pool, run the
   architecture-matched `bar_write_canary` for each XTX destination
   (and separately for the R9700 if useful). Demand 0 mismatches at
   4 KiB and 4 MiB on every generation.
4. Review CPU read/write ordering, buffer placement, and timing evidence.
   Only then design the standalone bandwidth A/B gate.

## What is still unknown

- Whether the CPU-accessible HSA allocation is actually backed by the
  desired VRAM aperture on this host and can be safely accessed.
- Whether CPU posted stores need a stronger cache-flush/read-back step.
- Whether HIP can use an HSA-allocated pointer in the ROCm version on
  the target host. The canary deliberately fails on unsupported combinations.
- Whether cross-process GPU-memory IPC preserves CPU-mapped access.
- Whether a CPU callback plus BAR write is faster on the **real critical
  path**, not just under a synthetic contention benchmark.
