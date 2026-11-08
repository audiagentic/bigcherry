# RAD42 — CPU BAR push transport experiment (isolated)

**Status:** Stage 0 card-free tests passed (10/10); Stage 1 C++ syntax check passed. No GPU run or engine integration.
**Owner / question:** RAD40 / RAD42, with future RAD49 TP2 applicability.
**Code boundary:** This folder is **lab-only**, never linked into the serving engine.
**What to test:** Whether CPU-produced remote-MoE result chunks written into GPU-local,
CPU-addressable VRAM are faster than the current mapped-host GPU pull, without changing
the GPU's model scheduling.

## Existing evidence (not new measurements)

RAD40 measured 2026-11-01: under XTX0 host reads, the R9700's GPU pull fell
from 6.7 to 3.4 GB/s and XTX1's from 13.3 to 6.8 GB/s; CPU BAR pushes
remained at 5.6 GB/s (R9700) and 10.3 GB/s (XTX1). These are transfer microbench
rates, **not** proven end-to-end inference improvements. RAD42 delivered chunked
GPU consumption but deliberately left CPU BAR push unimplemented. RAD49's
multi-rank engine makes correct consumer rank and exactly-once results essential.

## Small gates — do not skip ahead

| Gate | Harness | Only proceed if |
| --- | --- | --- |
| 0. Contract/lifetime (card-free) | `chunk_contract.py`, `test_chunk_contract.py` | Old generations rejected; publication cannot precede write drain; no buffer reuse while in-flight; fault requires quiescence |
| 1. VRAM reachability (GPU lease; discovery only) | `pool_inventory.cpp` | Each prospective destination has an HSA GPU-local allocatable pool that the host CPU *can* access; record GPU PCI identity (not ordinal-only). A candidate is not proof of a usable BAR mapping |
| 2. Minimal correctness canary (not implemented yet) | Dedicated HSA/HIP CPU-write → GPU-read probe | One GPU and one CPU, 4 KiB then 4 MiB, checked bytes; explicit allocation/access, GPU-visible publication with independent generation; no crashes, stale reads or bad data |
| 3. Standalone transfer cost (not implemented yet) | Add checksum-preserving CPU BAR vs mapped-host GPU-pull modes | 4 KiB–21 MiB sweep, 1/2/4 CPU threads, median/p95 and CPU cost, both XTX and R9700, with and without GPU0 host-pull contention |
| 4. Chunk pipeline (not implemented yet) | Two-process producer CPU + consumer GPU, 128/256/512/1024 rows | GPU waits on *separate readiness*, every chunk exactly once under generation recycling/fault; p50/p95 step makespan beats baseline |
| 5. Harnessed MoE behavior (not implemented yet) | Replayed real result planes, CPU combine and BAR writes, no engine changes | FP16 numerical fidelity; overflow fallback; TP2 rank ownership; repeated steps; no hang; measured critical-path gain |
| 6. Optional separate implementation branch | RAD42/RAD49 runtime behind opt-in transport setting | Only after above gates, A/B, KLD, and GPU lease qualification. No default change in this experiment |

The GPU producer's existing `request_sequence` cannot be reused to claim that
the CPU has finished writing BAR memory. Two independent events are required:
the GPU's result generation and a **CPU write-drained** payload-ready generation.
Buffer recycling must wait for the consumer's observed completion. Cross-process
VRAM mapping, IPC lifetime, cache coherence and fence semantics remain hardware
questions; the card-free model asserts ordering *requirements*, not a real fence.

### Card-free gate 0

From this directory:

```sh
python3 -m unittest discover -s . -p 'test_*.py' -v
```

No GPUs or ROCm dependencies.

### Build gate 1 (compile only; does NOT use a GPU)

From the repository root, output into a scoped build directory outside the worktree:

```sh
mkdir -p "$BC_WORK_ROOT/builds/rad42-bar-push"
g++ -std=c++17 -O2 -Wall -Wextra tools/lab/rad42-bar-push/pool_inventory.cpp \
    -I"$ROCM_PATH/include" -L"$ROCM_PATH/lib" -Wl,-rpath,"$ROCM_PATH/lib" \
    -lhsa-runtime64 -o "$BC_WORK_ROOT/builds/rad42-bar-push/pool_inventory"
```

To **run** gate 1: obtain the normal exclusive `agent-lease gpu` first and
check GPU processes / PCIe links. Do not run under a running server or while
another measurement owns the GPU lease. The probe only enumerates HSA pools;
it does **not** allocate VRAM or establish transfer correctness.

### Scope and safety

- No production engine / helper / patch / manifest edits.
- No live server restarts, model loads, GPU probes, leases taken on another owner's behalf, or GPU benchmarks without the lease.
- No assumptions about resizable BAR access from HSA capability alone.
- Experiments must refuse to proceed on unsupported mappings or failed checks.
- No absolute device BDF addresses or local environment values in committed files.
- Store hardware evidence (once permitted) under a named RAD42 run in
  `${BC_WORK_ROOT}/runs/` (outside the checkout), with run identity and logs.
- Preserve both the existing GPU-pull path and bit-identical/quality-tested
  fallback in any future integration. Decode and prefill must be evaluated separately.

**Disposition:** Explore locally. Neither promotion nor PR is authorized here.
