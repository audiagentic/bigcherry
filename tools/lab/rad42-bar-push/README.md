# RAD50 — isolated CPU BAR-push transport experiment

**Status (2026-11-09):** Gate 0 **PASS** (23 card-free tests), Gate 1 **PASS** (7 HSA GPU-local CPU-accessible pools); Gate 2 **PASS on both XTXs**, R9700 **INCONCLUSIVE** (exit 1 during first HIP verification launch). IPC, concurrent readiness and performance remain unverified.
Gate 2 ran under an exclusive RAD50 GPU lease: XTX0/XTX1 each passed 12 serial generations at 4 KiB and 4 MiB (every word checked). The R9700 probe reached HSA VRAM allocation, CPU stores and HIP memset but did not return from the first verification launch; a HIP-only smoke mode is compiled, not run. GPU lease released; later gates have not run.

**Plan:** `docs/planning/active/patching-radiance-rdna3/RAD50.md`
(existing `patching-radiance-rdna3` group). Production owner RAD42 design B;
context RAD40 (PCIe contention) and RAD49 (multi-rank consumer).
**Journal:** [PROGRESS.md](PROGRESS.md) tracks gate progress. [FINDINGS.md](FINDINGS.md) explains risks, [RUNBOOK.md](RUNBOOK.md) governs hardware runs, and RAD51 is the future production plan.
**Boundary:** All files are lab-only, never linked into the serving engine.

## Question and existing evidence

Can CPU-produced remote-MoE FP16 result chunks be pushed straight into
CPU-addressable GPU-local VRAM, instead of the consuming GPU pulling
a mapped-host result plane, while leaving GPU model orchestration intact?

RAD40 (2026-11-01) measured that, while XTX0 pulls from host memory,
XTX1's GPU pull falls from 13.3 to 6.8 GB/s, R9700's from 6.7 to
3.4 GB/s, but CPU BAR pushes remain at 10.3 GB/s (XTX1) and
5.6 GB/s (R9700). These are **existing microbenchmarks**, not evidence
of a live inference gain. Uncontended GPU pulls are faster than BAR
pushes, so the design must win on *contended critical-path makespan*.
RAD42's chunked combine is shipped; its BAR-push design is not.

## Progressive stop/go gates

| Gate | Harness | Acceptance / current state |
| --- | --- | --- |
| 0. CPU model, card-free | `chunk_contract.py`, `test_chunk_contract.py`, `test_chunk_stress.py`, `critical_path_model.py`, `test_critical_path_model.py` | **PASS**: no early ready, stale epoch, reuse while busy, duplicate consume, or recovery without quiescence; 23 tests, 12,800 multi-chunk consumptions and 10 critical-path model tests |
| 1. VRAM reachability | `pool_inventory.cpp` | **PASS (API discovery)**: 7 GPU-local pools reported CPU-accessible across four GPUs; find an allocatable HSA GPU-local pool the host CPU is allowed to access; map physical PCI identity; pool availability alone is *not* proof of BAR usability |
| 2. CPU→GPU exact canary | `bar_write_canary.hip` | **XTX0 + XTX1 PASS; R9700 INCONCLUSIVE**: 4 KiB and 4 MiB, 12 generations each, GPU checks every word of CPU-written GPU-local storage; refuse unsupported allocations and PCI mismatch |
| 2b. Cross-process CPU-only IPC | `bar_ipc_canary.hip` (built, not run) | **BLOCKED on Gate 2**: prove a separate CPU-only process can map/write engine-owned GPU-local VRAM through HSA IPC, with no peer-GPU mapping |
| 3. Bandwidth and contention | `bar_transfer_ab.hip` (built, not run) | **BLOCKED on Gates 2/2b**: measure BAR write vs mapped-host GPU pull on both XTXs/R9700, 4 KiB–21 MiB, 1/2/4 CPU threads, p50/p95 and CPU load, alone and under GPU0 host-pull contention |
| 4. Live chunks | `bar_live_ready_canary.hip` built (live single-process canary); full multi-process chunk pipeline not built | **BLOCKED on Gate 3**: separate CPU producer/GPU consumer, independently published ready generations, 128/256/512/1024 rows, bounded timeouts, stale-data and fault cases |
| 5. MoE reference | `combine_reference_test.cpp` (compiled, not run) | **BLOCKED on Gate 4**: actual FP16 CPU combine, overflow sentinel fallback, TP2 consumer rank, once-only rows, captured replay and latency |
| 6. Opt-in integration | Not authorised | Only a future reviewed branch, after hardware A/B, numerical parity and explicit go decision |

**Critical ordering boundary:** the engine GPU's request publication is
not evidence that a CPU BAR push has finished. A prospective protocol
needs an independent *CPU-write-drained and GPU-visible* chunk-ready
generation and a consumer acknowledgment before buffer reuse.
A Python state model proves only state-transition rules; it cannot
prove PCIe posted-write ordering, cache-coherence fences or HSA IPC.
The single-process Gate 2 probe tests a **serial launch after CPU
stores**, not a concurrent GPU/CPU handoff.

## Commands and requirements

From the experiment worktree root, card-free:

```sh
python3 -m unittest discover -s tools/lab/rad42-bar-push -p 'test_*.py' -v
```

Gate 1/2 compile-only (does not run HSA/GPU operations):

```sh
source tools/env/bigcherry-env.sh
mkdir -p "$BC_WORK_ROOT/builds/rad42-bar-push"
g++ -std=c++17 -O2 -Wall -Wextra \
  -I"$ROCM_PATH/include" \
  tools/lab/rad42-bar-push/pool_inventory.cpp \
  -L"$ROCM_PATH/lib" -Wl,-rpath,"$ROCM_PATH/lib" -lhsa-runtime64 \
  -o "$BC_WORK_ROOT/builds/rad42-bar-push/pool_inventory"
for arch in gfx1100 gfx1201; do
  hipcc -std=c++17 -O2 --offload-arch="$arch" \
    tools/lab/rad42-bar-push/bar_write_canary.hip -lhsa-runtime64 \
    -o "$BC_WORK_ROOT/builds/rad42-bar-push/bar_write_canary-$arch"
done
```

**Hardware run preconditions:** acquire/check the project's exclusive GPU
lease by its approved mechanism; confirm no active serving and log
hardware identity, ROCm/kernel versions and PCI topology. A missing or
held lease is a **hard stop**; don't use a subprocess wrapper to evade
the remote tool allowlist.

With lease held, run the pool inventory first. Only if it reports
eligible GPU-local memory for the intended destination, run the matching
canary binary with `--device HIP_VISIBLE_ORDINAL`. Do not assume
HIP and HSA enumerations match; the canary compares PCI identity.
Require zero mismatches on every generation. Any unsupported mapping,
crash, hang, failed HIP API, or mismatch is a stop condition.

## Hygiene and disposition

- No vendor, serving engine, helper, patch, manifest or production
  configuration changes in this experiment.
- Never infer transfer benefit or correctness from a compile alone.
- Retain the existing mapped-host GPU-pull transport as control/fallback.
- Store raw hardware logs, once authorized, in a named project work-root
  evidence run with source commit, device and command provenance.
- No push, PR or merge. Continue only after the previous gate succeeds.
- Current disposition: exploratory and isolated; read [PROGRESS.md](PROGRESS.md).
