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
| 3: isolated transfer A/B | **SYNTHETIC + REAL FP16 COMBINE SERIAL PASS; ENGINE PENDING** | Synthetic transfer A/B 4 MiB: 1.36–1.41x (both XTX). New production AVX2/F16C combine directly into GPU-local VRAM passed full-word GPU checking for 1–768 rows; 100-iteration 768-row result: 1.127x XTX0, 1.131x XTX1, ~79–81 us serial saving. Single-row push slower. See GATE3_RESULTS.md and GATE3_COMBINE_RESULTS.md. Concurrent GPU, crash/restart and engine TP2 remain unproven |
| 4: chunked producer/consumer | **PARTIAL BUILD, RUN BLOCKED** | `bar_live_ready_canary.hip` compiled gfx1100/gfx1201: GPU already running before CPU writes. Multi-process chunk pipeline still not implemented. Requires Gate 3 |
| 5: captured MoE data / TP2 exact-once | **REAL CPU COMBINE + GPU BIT CHECK PARTIAL PASS** | Production combine_row_f16 now computes direct into DMA-BUF/host across deterministic BF16 helper-mask, overflow sentinel, NaN profiles; both XTXs pass all GPU words over 100 repeated 768-row generations. Captured KLD/TP2 replay, ranks and concurrent once-only pending |
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

### 2026-11-09 — Gate 3 measured serial XTX A/B

- After RAD49 released the lease and four GPUs were idle/no KFD PIDs, acquired exclusive RAD50 lease. Recompiled benchmark-relevant tester and DMA-BUF A/B with -O2, gfx1100+gfx1201, HIP 7.2.53211 / ROCm 7.2.4.
- Both XTXs passed 30 measured iterations + 3 warmups for 4 KiB, 64 KiB, 256 KiB, 1 MiB, 4 MiB, both direct synthetic stores and staged prepare+memcpy. Full GPU word-check every iteration, all exits 0 and post-cleanup success markers.
- Direct 4 MiB XTX0: mapped-host GPU pull 1075.16 us vs DMA-BUF CPU push 760.30 us (1.414x); XTX1: 1139.85 us vs 836.90 us (1.362x). At 4 KiB CPU-push loses ~1.4–1.6 us; staged 4 MiB yields only ~1.11x. Stage-level details in [GATE3_RESULTS.md](GATE3_RESULTS.md).

- At 4 MiB direct, CPU writes cost ~540–560 us on either destination, and DMA-BUF SYNC START+END ~0.5–0.7 us p50. The GPU verification kernel is ~331–333 us reading mapped-host memory vs ~25–27 us local VRAM. These are synthetic serial stage times, not Radiance inference timings.
- Preserved raw logs for earlier 10-iteration and new 30-iteration trials at ${BC_WORK_ROOT}/builds/rad42-bar-push/rad50-ab-*.log. Analysis saved as rad50-gate3-o2-analyzed.txt in the same directory.
- Analyzer is strict for final results (requires successful post-cleanup completion), but new --provisional mode exposes interim or failed-run observations with clear warnings instead of discarding useful partial logs. 50/50 Python card-free tests pass.
- Confirmed all GPUs idle and no KFD PIDs afterward. Released GPU lease; no production/vendor changes, push, PR or new worktree.
- Next: repeat-generation/lifetime safety, GPU-live ready, contention, real FP16 CPU combine directly into GPU VRAM, paired TP2 inference. No inference speedup claim.

### 2026-11-09 — Gate 3b production FP16 combine into DMA-BUF

- Added fixed `rad50-tester --test combine-ab --device N --iterations 10..200 --mode direct|staged [--rows 1|16|64|256|768] --lease-confirmed`. Reuses the persistent separate CPU-only helper, mmap, HSA VMM DMA-BUF and GPU verification from Gate 3. CPU worker now actually calls production `combine_row_f16` (AVX2/F16C, two BF16 helpers, 2,560-wide row). Four deterministic source profiles vary with generation/row: normal sum, helper mask, overflow sentinel, canonical NaN. Independent GPU kernel bit-checks every FP16 word. Staged mode computes into RAM then memcpy; direct mode outputs straight into GPU VRAM or baseline host mapped RAM.
- Under exclusive RAD50 GPU lease with idle/empty GPU preflight, both XTXs passed 1-row and 768-row 10-sample canaries, full 30-sample 5-size direct/staged sweeps and 768-row 100-iteration repeats; all device full-word checks, helper exit and HSA cleanup passed. Serial in-place reuse across generations proven; GPU-already-reading concurrency and helper crash/restart remain untested.

- In the **100-iteration 768-row** runs, direct combine into GPU-local VRAM versus host-mapped GPU pull: XTX0 621.96 vs 700.87 us (1.127x); XTX1 622.16 vs 703.44 us (1.131x). +49–50 us CPU output cost to GPU VRAM offset by ~128 us GPU consume completion saving, ~79–81 us net serial benefit. 1 row is ~1–2 us worse. 30-sample full sweeps vary substantially at 768 rows (XTX0 1.267x; XTX1 0.973x), so short-run variation is reported rather than hidden.
- Partial and completed raw logs preserved under `evidence/2026-11-09/rad50-combine-*.log` and original build-root captures; strict and provisional analyzers now support new combine modes and per-style size families. Card-free tests 58/58; CLI 47/47. Full findings in [GATE3_COMBINE_RESULTS.md](GATE3_COMBINE_RESULTS.md).
- Released exclusive lease after all four GPUs returned idle/zero KFD PIDs. No new worktree, vendor/production code, push or PR.
- Next gates: cross-process GPU-already-waiting ready-generation publication and ack, crash/restart/FD lifetime, controlled PCIe contention, captured model BF16 data and actual Radiance TP2 prefill/decode/KL A/B. Do not attribute synthetic serial speedup to inference.

### 2026-11-09 — Gate 4 GPU-already-waiting interprocess test, partial pass / large-shape failure

- Added fixed `rad50-tester --test dmabuf-live` using an independent CPU-only helper writing the real production FP16 combine straight to exported GPU-local DMA-BUF, with a GPU verification kernel launched *before* CPU writes. GPU checks every word, publishes done/ACK, and next generation uses the same slot only after done + GPU drain; previous ready flag deliberately retained.
- Under a RAD50 exclusive lease, both XTXs passed 10 generations at 1, 16, and 256 rows. Full per-generation stdout/stderr, exit codes and timings saved to `evidence/2026-11-09/rad50-live-xtx*.log`. The 16/256-row timings reflect a diagnostic GPU kernel and are NOT inference latency.
- At 768 rows, XTX0 **FAILED at generation 1**: GPU timeout sentinel errors=4294967295. Stopped remaining trials and released lease. A single-poller GPU rewrite, instrumented for CPU/GPU observed ready values, was rebuilt and tested under a separate exclusive lease: **still failed** at generation 868001; CPU observed ready=868001, GPU observed ready=0, helper ACK arrived 872.71 us after arming. Not an FP16 mismatch; evidence of unsolved concurrent ready visibility. Stopped and released lease.

- Official ROCm HIP documents distinguish fine-grained host memory; installed hipExtHostRegisterUncached maps extended fine-grained host memory on AMD Linux except Navi4X. Prepared a separate, fixed `dmabuf-live-fg` variant that uses that flag on the shared host memfd. **Compiled for gfx1100/gfx1201, not yet hardware tested**; do not assume R9700/Navi4X support.
- Added `live_analyze.py` strict and --provisional log analysis; the latter keeps partial/failing hardware results. Both failure logs are saved in Git evidence. 72 Python tests and 60 unified CLI tests pass card-free. Detailed result and next design alternatives: `GATE4_LIVE_RESULTS.md`.
- Current state: **Gate 4 BLOCKED**, no serving integration. RAD49 subsequently acquired exclusive GPU lease; no more RAD50 hardware work. Existing worktree, no push/PR/production edits.

### 2026-11-09 — Slice optimization: gfx11 timer fix and GPU-live correctness

- GPU-local failure diagnostics isolated the false 768-row deadline: `exit_reason=2` from the old gfx11 `clock64()` check, not verified FP16 corruption. The HIP-rate-scaled `wall_clock64()` deadline plus independent host watchdog fixes the hardware canary. Original one-block and extended uncached mode passed 768 rows (10 generations, XTX0). Direct and coherent-relay parallel variants passed 100 generations × 768 rows on **both XTXs** with GPU armed before CPU writes, exact ready ticket and full FP16 GPU verification before buffer reuse.
- Replacing the original single-block full-plane verifier (~7.4 ms) with separate GPU waiter, 256-block verifier and done kernel yields **463.87/463.71 us p50** on XTX0/XTX1 for 768-row direct publication (100 generations). Parent coherent relay: 469.06/471.06 us p50. These are not same-session randomized pairs and not Radiance inference gains.

- Live direct 768-row stage budget: CPU FP16 combine 363.40/363.62 us p50; GPU arm 24.14/24.06 us; helper ACK after arm 368.25/368.99 us; GPU completion after ACK 71.05/70.52 us. Warm DMA-BUF SYNC ~0.3–0.4 us. CPU combine and GPU dispatch are next priority; DMA SYNC is not a measurable primary bottleneck.
- Direct parallel live at 1/16/64/256 rows also passed 30 generations on each XTX; see `evidence/2026-11-09/rad50-slices-validated-summary.txt`. Compact long-run logs retain first three, each tenth, last, all failures, summary over all generations and post-cleanup completion. The strict `slice_analyze.py` checks expected samples and completion; `--provisional` preserves partial/failing output. Card-free suite 91/91 passed including gfx11 clock guard.
- Follow `SLICE_OPTIMIZATION.md` for slice-level optimization matrix, numerical and lifetime gates, and final paired TP2 inference. RAD49 now holds the GPU lease; no further GPU tests, no serving/vendor edits or new worktree.

### 2026-11-09 — RAD50 GFX11 watchdog correction and slice isolation

- **Root cause:** HIP documents unreliable `clock64()` on gfx11. Device-local diagnostics showed false clock deadline (exit_reason=2 after ~528 polls) despite the CPU-ready ticket. Both original and new parallel GPU waiters now use `wall_clock64()` with HIP device wall-clock rate; separate host-side deadlines remain. Original, extended-uncached, direct parallel and coherent-parent-relay GPU-live 768-row tests now pass observed hardware canaries on XTX0. The parallel direct and relay modes pass XTX1 too.
- **GPU consumer isolation:** old one-block GPU verifier ~7381us 768-row diagnostic p50 vs ~464us with separately launched GPU waiter, 256-block verifier and done kernel, while still running real production CPU FP16 combine and exact every-word GPU checks. No inference speedup established.
- **100-generation full live correctness, BOTH XTXs**: direct CPU ready publication XTX0 p50/p95 463.87/596.41us, XTX1 463.71/590.44us; coherent parent relay XTX0 469.06/607.11us, XTX1 471.06/532.53us. All post-cleanup complete/exit0. Compact 100-run stdout prints first 3, every tenth and final generation; every generation is GPU checked and included in summary. Full logs, partial logs and historical failures saved in `evidence/2026-11-09/rad50-slices-*.log`.
- 15-iteration untruncated sweeps across 1/16/64/256/768 rows on both XTXs also pass. Strict `slice_analyze.py` checks the compact sample schedule, FP16 mismatch, observed GPU ticket, failure and postcleanup completion; `--provisional` retains partial records. GFX11 clock regression unit tests prevent reintroduction.
- Slice priorities: CPU FP16 combine ~363us for 768 rows; GPU arm ~24us; post-ACK completion ~70us; socket work ~5–6us beyond combine; DMA-BUF SYNC typically sub-us. Component medians overlap and must NOT simply be added. Direct vs parent relay 5–7us median difference in sequential runs is not a reliable paired speedup. Documented in `SLICE_OPTIMIZATION.md` and `docs/planning/active/patching-radiance-rdna3/RAD50_SLICE_OPTIMISATION.md`.
- Next: helper crash/restart and FD lifetime, duplicate/stale/fault injection, GPU fused consumer, captured BF16 heat and TP2 rank once-only, controlled PCIe contention, R9700 and full model prefill/decode/KL A/B. GPU lease released and no KFD PIDs afterward; RAD49 subsequently acquired lease. Lab-only edits, same worktree, no production/vendor change.

### 2026-11-09 - S4 GPU verifier grid isolation

- The existing `rad50-tester` accepts bounded `--verifier-blocks 1|32|64|128|256` only for live parallel/relay modes; default remains 256. GPU waiter, full FP16 oracle, DMA-BUF SYNC, exact ticketing and post-cleanup proof remain unchanged. Strict analyzer checks grid metadata.
- **22 complete GPU-live S4 captures passed strict verification** across both XTXs and 1/768-row cases; one locally interrupted capture is preserved as PARTIAL, not promoted. OPT=-O2 gfx1100+gfx1201 build passed; CLI 75/75, Python 95/95 card-free passed.
- At 768 rows XTX0 (100 generations), 128 blocks 479.48 us median / 1698.41 us p95 vs 256 blocks 464.23 / 520.50 us. On XTX1 the 256-block GPU post-ACK stage was faster, but total timing was obscured by higher CPU combine costs; one-block large-row verification took 7.50 ms. One-row results did not establish a consistent advantage for smaller grids.
- **Keep 256 blocks**; no inference speedup or shape-dependent tuning claim. Concurrent Radiance build makes CPU/tail latency noisy. Next: isolate ~363-us large-row CPU FP16 combine and BAR stores, fault/restart scenarios, fused Radiance consumer and full TP2 overlap.
- Full evidence and caveats: [S4_VERIFIER_SWEEP.md](S4_VERIFIER_SWEEP.md). RAD50 released its GPU lease; RAD49 subsequently acquired it. Same worktree, lab only, no production/vendor edits.

### 2026-11-09   S1 CPU-only production-combine isolation checkpoint

- Reused the **single** allowlisted rad50-tester; fixed mode combine-cpu with bounded 1/16/64/256/768 rows, 1/2/4 helpers, 10 200 generations and hot/unique BF16 source layouts. This is a host-only production AVX2/F16C FP16 combine plus staged host memcpy comparison, NOT a GPU/BAR benchmark.
- All rows in both paths match an independent scalar bitwise oracle including masks, overflow fallback, canonical NaN, signed zero, BF16 subnormal and helper-order cases. Stage timings and all generation summaries have a strict/provisional parser; CPU CLI checks **87/87**, Python card-free **106/106**, OPT=-O2 fat binary link pass. Four current-format complete CPU captures strictly validated.
- Diagnostic 768-row p50: 2-helper hot 335.86 us, unique 785.45 us, 4-helper unique 1284.64 us. But earlier 2-helper unique was ~382.96 us and a concurrent CPU check showed **98.55% nice utilization** during independent HIP builds. **Reject speed rankings under this load**; correctness only. Full method, table and next gates [S1_CPU_COMBINE.md](S1_CPU_COMBINE.md).
- RAD50 Gate 0 offline handoff and overlap models already merged to main in PR #785. All later hardware and S1 optimization remain in this existing experimental worktree; no production changes or new worktrees.
