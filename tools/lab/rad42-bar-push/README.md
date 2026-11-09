# RAD50 — isolated CPU BAR-push transport experiment

**Status (2026-11-09):** Gate 0 **PASS** (23 card-free tests), Gate 1 **PASS** (7 HSA GPU-local CPU-accessible pools); Gate 2 **PASS on both XTXs**, R9700 **INCONCLUSIVE** (exit 1 during first HIP verification launch). R9700 HIP-only smoke passed, but its serial BAR verification remains inconclusive. **DMA-BUF cross-process CPU mapping and actual 4 KiB/4 MiB writes now pass on both XTXs**, although direct HSA IPC CPU-agent mapping fails. Serial repeated-generation reuse and serial performance now have hardware evidence (including actual production FP16 combine); GPU-live readiness passes 1/16/256 rows but **FAILED at 768 rows** on XTX0; FD lifetime/restart and inference performance remain unverified. See GATE4_LIVE_RESULTS.md.
Gate 2 ran under an exclusive RAD50 GPU lease: XTX0/XTX1 each passed 12 serial generations at 4 KiB and 4 MiB (every word checked). The R9700 probe reached HSA VRAM allocation, CPU stores and HIP memset but did not return from the first verification launch; the HIP-only smoke mode now passes. Direct HSA IPC CPU mapping failed, but Linux DMA-BUF mmap in a separate CPU-only process succeeded, and the parent GPU verified the 4 KiB and 4 MiB writes. GPU lease released; **isolated synthetic Gate 3 serial performance measurements now completed** on both XTXs (see GATE3_RESULTS.md), while concurrent readiness and real inference A/B remain unrun.

**Plan:** `docs/planning/active/patching-radiance-rdna3/RAD50.md`
(existing `patching-radiance-rdna3` group). Production owner RAD42 design B;
context RAD40 (PCIe contention) and RAD49 (multi-rank consumer).
**Journal:** [PROGRESS.md](PROGRESS.md) tracks gate progress. [FINDINGS.md](FINDINGS.md) explains risks, [RUNBOOK.md](RUNBOOK.md) governs hardware runs, and RAD51 is the future production plan.
**Boundary:** All files are lab-only, never linked into the serving engine.

**Preferred interface:** [TESTER.md](TESTER.md) documents the new single
`rad50-tester` executable with fixed `--test` cases and both GPU architectures
in one binary. It is **built and allowlisted**: `--list`, 60/60 CLI checks and
seven FP16 reference checks have run successfully, without a GPU.
One stable executable name replaces per-test allowlist entries.
The old standalone probes remain as source/reference, not subprocess targets.

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
| 2b. Cross-process CPU-only IPC | `rad50-tester --test dmabuf-map`, `dmabuf-write`, `dmabuf-full`, `combine-ab` | **PARTIAL PASS**: both XTXs pass separate-process DMA-BUF 4 KiB/4 MiB correctness and real FP16-combine repeated-generation serial reuse (100 cycles). Direct HSA CPU-agent IPC failed; live readiness and crash/restart still pending |
| 3. Bandwidth and contention | `rad50-tester --test dmabuf-ab` completed serial benchmark, plus historical `bar_transfer_ab.hip` | **ISOLATED SERIAL A/B MEASURED**: same CPU-only child and GPU verification kernel over 4 KiB–4 MiB, 30 iterations, direct/staged paths. 4 MiB direct push 1.36–1.41x faster than mapped-host GPU-pull. Controlled contention and inference replay still pending; see GATE3_RESULTS.md |
| 4. Live chunks | `rad50-tester --test dmabuf-live` and `dmabuf-live-fg` candidate | **BLOCKED: 768-row GPU visibility FAIL**: persistent CPU helper and GPU-already-waiting verification pass 1/16/256 rows on both XTXs, but XTX0 768-row failed twice (CPU saw generation, GPU saw stale 0). Fine-grained host registration candidate compiled but untested; see GATE4_LIVE_RESULTS.md |
| 5. MoE reference | `combine_reference_test.cpp`, `rad50-tester --test combine-ab` | **PARTIAL PASS**: actual CPU AVX2/F16C combine into DMA-BUF/host tested across helper masks, overflow sentinel, NaN and repeated generations, verified on GPU. Captured MoE replay, TP2 rank exact-once and live GPU-ready semantics still pending |
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

### 2026-11-09 updated Gate 4 status

The historical 768-row GPU-ready timeouts were caused by an invalid GFX11 `clock64()` deadline in the test canary. With HIP's `wall_clock64()` and the device wall-clock rate, 768-row direct CPU-published and coherent-parent-relay paths now pass **100 concurrent-ready generations per XTX**, including production CPU FP16 combine into GPU VRAM and every-word GPU verification. Isolating the GPU waiter and using a 256-block verifier reduced the 768-row diagnostic from ~7.38 ms p50 (single block) to ~0.464 ms p50 (direct publish, 100 generations). These are synthetic input and diagnostic consumer timings, **not inferred inference speedups**. See `SLICE_OPTIMIZATION.md`, `GATE4_LIVE_RESULTS.md`, and `docs/planning/active/patching-radiance-rdna3/RAD50_SLICE_OPTIMISATION.md`. Real model ranks, fault/restart, GPU fused consumers, PCIe contention and TP2 inference remain blocked. Old failures remain retained as historical diagnostic records.

### Latest RAD50 Gate 4 slice result (2026-11-09)

The earlier 768-row GPU live failure was traced to the unreliable gfx11 `clock64()` deadline, not demonstrated FP16 data corruption. The corrected `wall_clock64()` GPU waiter and 256-block verifier pass 100 repeated 768-row generations on **both XTXs**. Direct CPU-published ready p50: 463.87 us (XTX0), 463.71 us (XTX1). CPU combine (~363 us) dominates; DMA-BUF SYNC is sub-microsecond after warmup. See `SLICE_OPTIMIZATION.md` and `evidence/2026-11-09/rad50-slices-validated-summary.txt`. These are experimental lab results, not production memory-model guarantees or inference speedups. Host pull remains the fallback; end-to-end integration remains gated.

## Promotion boundary (2026-11-09)

The card-free Gate 0 handoff contract and critical-path model were merged into `main` by PR #785 (`c4da56706`). This full experimental worktree retains the expanded README and later hardware/optimization gates; they remain unmerged and do not enable production transport.
