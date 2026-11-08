# RAD50 — progress, gates and evidence

This is the **in-worktree experiment journal**, not an inference-performance
claim. The authoritative plan item is
`docs/planning/active/patching-radiance-rdna3/RAD50.md`; production design
remains RAD42, and RAD49 is the TP2 consumer context.

## Current gate table

| Gate | State | Evidence / decision |
| --- | --- | --- |
| 0: lifecycle contract, card-free | **PASS** | `test_chunk_contract.py` (10 tests) and deterministic multi-chunk stress (`test_chunk_stress.py`, 3 tests); no real GPU visibility implied |
| 1: host-visible GPU-local pool discovery | **BUILD PASS, RUN BLOCKED** | `pool_inventory.cpp` compiled with g++; GPU lease command not available via current remote command allowlist, so no HSA hardware discovery performed |
| 2: CPU-write/GPU-read canary | **BUILD PASS, RUN BLOCKED** | `bar_write_canary.hip` compiled for gfx1100 and gfx1201; hardware correctness is unverified. Requires Gate 1 PASS and GPU lease before launch |
| 3: isolated transfer A/B | **NOT STARTED** | Do not implement a benchmark or claim throughput until Gate 2 verifies CPU-writable GPU VRAM |
| 4: chunked producer/consumer | **NOT STARTED** | Requires Gate 3 throughput case and dedicated CPU-write-drained flag semantics |
| 5: captured MoE data / TP2 exact-once | **NOT STARTED** | Requires proven chunk visibility and explicit quality parity |
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
