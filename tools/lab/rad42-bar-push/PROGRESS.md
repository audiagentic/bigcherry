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
| 2: CPU-write/GPU-read canary | **PASS XTX0/XTX1; R9700 INCONCLUSIVE** | Both gfx1100 GPUs: 4 KiB and 4 MiB, 12 generations each, zero mismatches, exit 0 (48 buffers). R9700 HIP-only kernel passes, but its HSA BAR verification stops during GPU verify launch; no serial result established |
| 2b: separate CPU-only IPC import | **PARTIAL PASS (DMA-BUF cross-process)** | HSA IPC CPU-agent attach failed, but VMM-exported DMA-BUF + separate CPU child mmap and SYNC START/END passed on XTX0/1; child CPU wrote 4 KiB and 4 MiB, parent GPU verified every 32-bit word in each (one generation/size). Reuse, faults, GPU-live visibility and p99 still unverified |
| 3: isolated transfer A/B | **BENCHMARK BUILT; HARDWARE PENDING LEASE** | New `rad50-tester --test dmabuf-ab` measures matched mapped-host GPU reads vs cross-process DMA-BUF CPU pushes with identical GPU verification, 5 sizes, p50/p95/p99, direct and staged writer variants. No speed numbers yet. Gate 2b serial correctness PASS, concurrency/lifetime still pending |
| 4: chunked producer/consumer | **PARTIAL BUILD, RUN BLOCKED** | `bar_live_ready_canary.hip` compiled gfx1100/gfx1201: GPU already running before CPU writes. Multi-process chunk pipeline still not implemented. Requires Gate 3 |
| 5: captured MoE data / TP2 exact-once | **PARTIAL REFERENCE PASS** | Unified `rad50-tester --test combine-ref` ran the actual BigCherry FP16/NaN/fallback checks, 7/7 PASS. Captured KLD/TP2 replay remains blocked by Gate 2b |
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

### 2026-11-09 — unified GPU tests and cross-process IPC diagnosis

- Preflight: four GPUs idle, no KFD process or lease. Acquired/released
  exclusive RAD50 leases for each staged hardware attempt.
- `rad50-tester --test hip-smoke --device 2`: **PASS**, R9700 basic
  HIP kernel launch and readback. R9700 HSA BAR verification remains unresolved.
- `--test ipc --device 0`: **FAIL**, HSA CPU-agent IPC attach status
  4104 = OUT_OF_RESOURCES, even after granting CPU access before export.
- `--test ipc-map --device 0` and `--device 1`: **PASS GPU-ONLY
  IMPORT** to the owning GPU agent. No CPU stores or GPU data check.
- XTX0 imported pointer reports hostBaseAddress and CPU agent grant
  succeeds; neither proves a writable CPU virtual address.
- `--test ipc-gpu-cpu --device 0`: **FAIL**. Guarded 4 KiB attempt
  refused to write: imported hostBaseAddress is NOT in a writable
  `/proc/self/maps` CPU VMA. No GPU verification occurred. The
  helper signal 15 was caused by parent aborting the failed protocol;
  do not interpret as a GPU memory fault or proven SIGSEGV.
- Parser self-check **21/21 PASS**; all modes remain in the single
  allowlisted `rad50-tester`. No production code changed.
- **Gate 2b remains FAILED for CPU writes**. Do not run bandwidth,
  live-ready, or model integration. Next: explore GPU-local VMM
  dma-buf export/import with CPU-access gate, or change architecture.

### 2026-11-09 — VMM DMA-BUF capability alternative

- Added fixed `rad50-tester --test vmem-cap --device 0`, a new
  lab-only ROCr VMM probe. Compiled for gfx1100+gfx1201.
- Under an exclusive RAD50 lease on XTX0, GPU-local VMM handle
  creation, DMA-BUF export, address reservation/map and combined
  CPU/GPU `set_access` all returned **HSA SUCCESS**.
- GPU `get_access` returned RW (status 0, permission 3).
  CPU `get_access` returned **4100 INVALID_AGENT**, not success.
  `/proc/self/maps` reported writable host VA, but that is only
  a virtual reservation, not proof of CPU GPU-local access.
- **VMM capability probe FAIL (exit 3)** on CPU access gate.
  Did not attempt CPU stores, cross-process DMA-BUF import or a GPU
  verification kernel. Unmapped, released handles, and released lease.
- Strict CLI self-check now **22/22 PASS**. Do not claim VMM
  is a viable alternative until an actual CPU mapping and parent
  GPU payload verification pass in separate processes.

### 2026-11-09 — Linux DMA-BUF cross-process path works on both XTXs

- New fixed unified-tester modes `dmabuf-map`, `dmabuf-write`, `dmabuf-full` use HSA GPU-local VMM handle export, SCM_RIGHTS to an independent CPU-only child, Linux `mmap(PROT_READ|PROT_WRITE)`, and `DMA_BUF_IOCTL_SYNC START/END`. No helper GPU calls/P2P.
- Under exclusive GPU lease, both XTX0 (PCI 0000:03:00.0) and XTX1 (0000:06:00.0) passed the no-store mapping capability probe. `mmap_errno=0`, `sync_start_errno=0`, `sync_end_errno=0` on each. No data proof from these probes alone.
- Both XTXs also passed separate-process CPU-written **4 KiB**, verified by a parent GPU kernel checking every 32-bit word (1,024 words).
- Both XTXs passed separate-process CPU-written **4 MiB**, verified by a parent GPU kernel checking every 32-bit word (1,048,576 words). Exit 0, no mismatches reported. One generation per payload/card; no repeated reuse or concurrency claim yet.
- The first `dmabuf-full` attempt failed due to a lab-mode dispatch bug (`op==2` not entering the CPU-write branch; `wrote=0`); fixed by allowing any nonzero op and reran successfully. This was not observed data corruption.
- Existing HSA CPU-agent IPC and HSA VMM CPU `get_access` failures remain valid negative results for those APIs, **not** evidence that DMA-BUF CPU mmap is impossible.
- **Gate 2b partial pass**, not final: repeated generations/consumer ACK, stale buffers, close/restart, concurrent GPU-visible publication and end-to-end latency are outstanding. `DMA_BUF_IOCTL_SYNC` may add significant syscall overhead; do not enable RAD51 or claim a speedup.
- Released RAD50 exclusive lease after results; all GPUs idle and no KFD processes. Single existing worktree, no vendor/production files altered.

### 2026-11-09 — Gate 3 paired timing harness prepared (card-free)

- New `bar_dmabuf_ab.hip` is integrated as fixed `rad50-tester --test dmabuf-ab` with architecture fat binaries (gfx1100, gfx1201). Built and linked, and parser self-check passes 34/34; the 32 Python card-free tests include nine strict `ab_analyze.py` pairing/invalid-log cases.
- Production-like baseline: a separate persistent CPU-only worker writes a HIP-registered shared host memfd, which the GPU **reads directly** with the same full-word verification kernel used for the proposed GPU-local VMM DMA-BUF CPU-push path. No artificial GPU copy on the baseline.
- Benchmark source provides size cases 4 KiB, 64 KiB, 256 KiB, 1 MiB, 4 MiB; alternating A/B per iteration; 3 warmups plus 10..200 measured samples; `direct` synthetic CPU compute/store and `staged` compute plus memcpy variants. Every generation and word is checked by GPU. Stage-level CPU preparation/write, DMA-BUF START/END, socket roundtrip, GPU kernel and end-to-end serial latency p50/p95/p99 are reported.
- The A/B program uses a persistent child and persistent mappings, not per-iteration spawn/mmap. It tests serial handoff; it **does not prove** GPU-live readiness, contention behavior, actual MoE combine or end-to-end inference performance.
- **No GPU speed run performed:** RAD49 held exclusive GPU lease and the host reported three heavily occupied GPUs. Next when the lease is free: rebuild benchmark at OPT=-O2, run same-session direct/staged modes on both XTXs, collect raw logs and analyze. Do not interrupt RAD49; no live speed claim.
- Full protocol, method and stop/go criteria are documented in [PERFORMANCE.md](PERFORMANCE.md). Same worktree; no production/vendor edits.

### 2026-11-09 — bounded per-size A/B readiness; hardware still occupied

- Added bounded optional `--bytes` to the existing `dmabuf-ab` mode (4 KiB, 64 KiB, 256 KiB, 1 MiB, 4 MiB only). This avoids packing all 5 sizes into a single short remote-shell invocation and permits 10-iteration first smoke, then 30+ iteration speed runs. Other modes reject the option; new parser tests verify duplicate/unsupported sizes and lease requirements.
- The unified gfx1100/gfx1201 benchmark **rebuilds at OPT=-O1**, CLI self-check **39/39 PASS**, existing Python card-free tests **32/32 PASS**. The initial speed evidence will require rebuild at OPT=-O2 under a quiet lease.
- No hardware A/B run or speed tuning result: RAD49 owns the exclusive GPU lease, and Radiance / remote-MoE worker processes have large VRAM allocations. No leases or GPU processes touched. See PERFORMANCE.md.
