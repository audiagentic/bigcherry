# RAD50 — progress, gates and evidence

This is the **in-worktree experiment journal**, not an inference-performance
claim. The authoritative plan item is
`docs/planning/active/patching-radiance-rdna3/RAD50.md`; production design
remains RAD42, and RAD49 is the TP2 consumer context.

## Current gate table

| Gate | State | Evidence / decision |
| --- | --- | --- |
| 0: lifecycle and critical-path model, card-free | **PASS** | 23 tests: 10 lifecycle, 3 multi-chunk stress, 10 critical-path/overlap model; no GPU visibility implied |
| 1: host-visible GPU-local pool discovery | **PASS (API discovery)** | Under exclusive RAD50 lease, HSA enumerated 7 CPU-accessible GPU-local allocatable pools: XTX0 2, XTX1 2, R9700 1, 6900XT 2. Not proof of BAR writes |
| 2: CPU-write/GPU-read canary | **PASS XTX0/XTX1; R9700 INCONCLUSIVE** | Both gfx1100 GPUs: 4 KiB and 4 MiB, 12 generations each, zero mismatches, exit 0 (48 fully verified buffers total). gfx1201 R9700 (HIP ordinal 2): executable exit 1 with no stdout/stderr; no memory correctness established. Diagnosing, no retry yet |
| 2b: separate CPU-only IPC import | **BUILT, PENDING LEASE** | XTX Gate 2 passed. The allowlisted combined `rad50-tester --test ipc` contains the IPC implementation. No IPC attach or GPU verification performed; another task holds the GPU lease |
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

## Next hardware steps

1. Confirm the GPU lease is free, no KFD processes or serving remains, and acquire a new exclusive GPU lease.
2. Diagnose R9700's exit-1/empty-output failure using the newly instrumented `bar_write_canary-gfx1201 --device 2` probe. Record whether process entry, HIP initialization, HSA match or allocation is reached. Do not classify it as GPU memory corruption without evidence.
3. After resolving or explicitly isolating that failure, proceed on the two XTXs with a separate-process CPU-only HSA IPC canary (Gate 2b), then the transfer A/B and live-GPU-ready experiments. Those executable names require remote command-policy permission.
4. Preserve the GPU-pull baseline, obtain measured latency/CPU overhead, and require all later safety gates before considering RAD51 runtime changes.

## What is still unknown

- Whether the CPU-accessible HSA allocation is actually backed by the
  desired VRAM aperture on this host and can be safely accessed.
- Whether CPU posted stores need a stronger cache-flush/read-back step.
- Whether HIP can use an HSA-allocated pointer in the ROCm version on
  the target host. The canary deliberately fails on unsupported combinations.
- Whether cross-process GPU-memory IPC preserves CPU-mapped access.
- Whether a CPU callback plus BAR write is faster on the **real critical
  path**, not just under a synthetic contention benchmark.

### 2026-11-09 — hardware lease obtained; Gate 1 executable blocked by command policy

- Preflight: `agent-lease status` initially reported no leases; `rocm-smi --showuse --showmemuse --showpids` reported all four GPUs at 0% utilization and 0% VRAM allocation, with no KFD PIDs at that instant.
- Acquired the exclusive `gpu` lease for RAD50 (`agent-lease acquire gpu --ttl 2h --purpose ...`); the command was accepted.
- `rocminfo` confirms gfx1100 and gfx1201 devices and their GPU-local HSA memory pools. `Coherent Host Access: FALSE` is reported for the XTX. `Accessible by all: FALSE` does **not** answer whether a selected CPU agent can access an HSA GPU-local allocation. Gate 1 still requires its own access query.
- Attempting the compiled `pool_inventory` executable was denied with `Command 'pool_inventory' not in allowlist`. This was **not** an HSA test failure; no experimental GPU kernels, allocations or BAR writes were launched.
- After observing this command-policy block, released the GPU lease; no lease left held by RAD50. No workaround through another allowed interpreter/build/debugger was attempted.
- Next: user/host administrator must explicitly permit the experiment executables in the remote command policy (start with `pool_inventory`, then `bar_write_canary-gfx1100` / `bar_write_canary-gfx1201`; only authorize later executables as earlier gates pass). Recheck all GPUs and obtain a fresh lease before any run. Do not call subsequent gates passing based on `rocminfo`.

### 2026-11-09 — Gate 1 API discovery PASS; Gate 2 executable policy BLOCKED

- Preflight `agent-lease status` showed no leases and `rocm-smi` reported 0% GPU use, 0% VRAM allocation, no KFD processes. Acquired exclusive RAD50 GPU lease.
- First `pool_inventory` run enumerated XTX0's two pools, then failed on a GROUP (non-global) memory pool. Fixed the lab-only HSA inventory and shared selector to skip non-global segments before querying global-only properties. No engine sources changed.
- Rebuilt and reran inventory under the lease: exit 0, seven GPU-local allocatable CPU-accessible pools: XTX0 (gfx1100, PCI 0000:03:00.0) 2, XTX1 (gfx1100, 0000:06:00.0) 2, R9700 (gfx1201, 0000:09:00.0) 1, 6900 XT (gfx1030, 0000:17:00.0) 2. This is HSA **capability discovery only**, NOT a CPU BAR write or a GPU verification.
- Attempting `bar_write_canary-gfx1100 --device 0` was rejected by the remote command allowlist, without running the executable. The policy allows `pool_inventory` but not `bar_write_canary-gfx1100` or `bar_write_canary-gfx1201`.
- Released our GPU lease immediately; confirmed no RAD50 lease retained. No experimental GPU allocations, kernels, IPC, BAR stores or transfer timings. All later gates remain pending.
- Required authorisation before Gate 2: permit those **exact two executable basenames**. Then recheck GPU users, acquire fresh lease, run XTX0 then XTX1 (and optionally R9700); stop on any wrong bytes, kernel error or unsupported mapping.

### 2026-11-09 — Gate 2 XTX correctness confirmed, R9700 inconclusive

- Confirmed all four GPUs idle and no other lease; acquired RAD50 exclusive `gpu` lease.
- `bar_write_canary-gfx1100 --device 0`: exit 0; HIP PCI `0000:03:00.0`; HSA GPU-local 4,194,304-byte allocation with CPU access; every word matched for **4 KiB × 12** and **4 MiB × 12** generations.
- `bar_write_canary-gfx1100 --device 1`: exit 0; HIP PCI `0000:06:00.0`; same allocation and both payload/generation sets passed. **Both XTXs serial CPU BAR write → GPU verification PASS.**
- `bar_write_canary-gfx1201 --device 2` (R9700): executable returned exit code **1** with empty stdout and stderr. Root cause unknown; no GPU verification can be claimed. Do not interpret this as measured data corruption or a valid rejection of CPU BAR.
- GPU state afterwards: 0% usage, 0% rounded VRAM allocation and no KFD PIDs. Released the lease (`no leases` verified).
- Added early unbuffered diagnostic markers to the lab canary for a future R9700 reproduction; recompiled gfx1201 **but did not rerun**. First investigate the exit stage, then repeat under a fresh lease. No IPC, transfer benchmark, GPU-live waiter or production changes.

### 2026-11-09 — R9700 HIP launch boundary; new GPU lease held by HD2701

- Under RAD50 exclusive lease, retried `bar_write_canary-gfx1201 --device 2`. Exit 1, no HIP error. Diagnostics reached HSA/HIP enumeration, matching R9700 PCI identity, HSA GPU-local allocation, CPU access, 4 KiB CPU stores and `hipMemset` of error counter.
- **Last marker:** `HIP memset completed, launching verification`; the marker *after* `hipLaunchKernelGGL(verify_words)` never appeared. Cause unknown: launch hang/fault or command execution limit. No readback, no R9700 Gate 2 pass, and no verified mismatches.
- Attempted XTX Gate 2b `bar_ipc_canary-gfx1100 --device 0` but the executable was **denied by allowlist**, before launch.
- Checked all GPUs idle with no KFD PIDs, released RAD50 lease.
- Added `--device 2 --probe-hip-only` to the existing canary: HIP `hipMalloc`, tiny kernel and readback, **no HSA allocation or BAR writes**. Compiled gfx1201, not run.
- Tried reacquiring a lease; HD2701 had acquired the exclusive GPU lease, so no further GPU work was attempted or foreign lease touched.
- Next: obtain fresh lease after HD2701, run HIP-only diagnostic. For XTXs, authorise `bar_ipc_canary-gfx1100`, acquire lease, then run cross-process mapping before transfer A/B.

### 2026-11-09 — single tester built; one allowlist entry still needed

- Added `rad50-tester` from `rad50_tester.cpp`, `tester_selfcheck.inc`
  and `Makefile`. HIP fat binary for gfx1100/gfx1201 **builds and links**.
- Named modes: `pools`, `hip-smoke`, `serial`, `ipc`,
  `transfer`, `live-ready`, `combine-ref`, `cpu-handoff`.
  This binary links all existing tests directly; no generic command
  execution or shell wrapper. IPC spawns only this same binary's fixed
  private child, using an inherited Unix socket.
- The remote policy already allows `rad50-tester`: `--list` ran,
  `--self-check` passed 19/19 strict CLI cases, and
  `--test combine-ref` passed all seven FP16 checks (2,560 elements).
  No GPU mode ran; HD2701 holds the GPU lease.
- The `--lease-confirmed` flag is an explicit operator acknowledgement,
  **not** automatic verification/acquisition. See [TESTER.md](TESTER.md).
- At preparation time HD2701 held the GPU lease and a foreign
  server was observed; no GPU testing or serving changes occurred.
  Recheck lease/processes immediately before any later GPU run.

- `llvm-objdump --offloading` verified four HIP kernel bundles with both
  `gfx1100` and `gfx1201` device images inside this one ELF. The current
  proof build SHA256 is `95a40d6c390970c9c04dc3dadb45ea54480762d080374c2b58a49a819fa8d202`
  (changes on rebuild). Its build default is `OPT=-O1`; rebuild with
  `OPT=-O2` for actual transfer-performance evidence, after the other
  GPU work has finished.
