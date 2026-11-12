# RAD50 S1   CPU FP16 combine and host destination isolation

**Checkpoint:** 2026-11-09, experimental RAD50 worktree. **Status:** host-only S1 correctness and benchmarking harness ready; CPU/VRAM performance optimization remains OPEN. No production/vendor/serving changes.

## Scope and controls

Gate 3b previously measured the real AVX2/F16C production combine for 768 rows of 2,560 FP16 values: 313 314 us in mapped host RAM versus 363 us in GPU-local DMA-BUF, about 49 50 us additional CPU-side destination overhead. This is synthetic-input serial GPU-verified evidence, not serving throughput.

This S1 sub-slice adds a CPU-only mode to the same single allowlisted RAD50 tester. The fixed option is:

    rad50-tester --test combine-cpu --rows 768 --iterations 100 --helpers 2 --mode unique

Selectors: rows 1/16/64/256/768, helpers 1/2/4, iterations 10..200, input mode hot or unique. No GPU lease, GPU allocations, HIP kernels, subprocesses or new binary required. The real production combine_row_f16 writes directly into ordinary host output RAM, or writes to a host staging plane followed by memcpy. Each iteration alternates the A/B order and measures combine, copy and total durations separately. Hot reuses six deterministic BF16 source row profiles; unique gives each output row its own source allocation.

Both paths undergo independent full-word scalar reference comparison on every generation, with helper masks, overflow sentinel, canonical NaN, signed zero, BF16 subnormals, ordered sums and 1/2/4 helpers represented. Full-bit parity is mandatory. This is an ordinary CPU RAM test: it does not prove GPU DMA-BUF coherence, ordering or performance.

## Validation completed

- C++ syntax and unified OPT=-O2 HIP gfx1100+gfx1201 link succeeded; this mode launches NO GPU work.
- Runner CLI argument self-check 87/87 passed.
- 106 Python card-free tests passed, including 11 new strict-log negative cases.
- Strict cpu_analyze.py validates exact context, sample schedule, success envelope, full verified generation count, emitted count, quantiles, stage arithmetic and completion; provisional mode retains incomplete logs without asserting a pass.
- Four full-word-verified trials, 10 or 100 iterations, preserved in evidence/2026-11-09/rad50-s1-cpu-strict-*.log. Older precontract log captures are retained separately for diagnosis.

## Timing samples   correctness validated, NOT a stable optimization ranking

| Input | Rows | Generations | Direct host p50 us | Host stage+copy p50 us |
|---|---:|---:|---:|---:|
| 1 helper, hot | 768 | 10 | 372.36 | 625.57 |
| 2 helpers, hot | 768 | 100 | 335.86 | 482.31 |
| 2 helpers, unique | 768 | 100 | 785.45 | 1186.24 |
| 4 helpers, unique | 768 | 100 | 1284.64 | 1694.55 |

An earlier 100-generation precontract trial of the same 2-helper profiles measured hot 325.49 us and unique 382.96 us for direct host combine (and stage totals 466.10/523.89 us). The later unique-source result more than doubled; staged copy also slowed sharply. At the later sample time, mpstat reported 98.55% nice CPU activity with concurrent independent ROCm/Radiance builds. Core pinning, clocks, memory pressure, NUMA and session order were uncontrolled. Therefore do NOT conclude that unique source locality costs a specific amount, or select any performance optimization from these runs. The negative performance reproducibility result is itself important; correctness remains independently established.

## Ordered next S1 gates   still OPEN

1. With CPUs quiet, record affinity, NUMA, clocks, build activity and temperature. Interleave repeated hot/unique and direct/staged trials across multiple sessions, with exact output checks and retained partial logs.
2. Replace synthetic source patterns with captured model-representative helper masks and row heat. Evaluate 1/2/4 helper counts, cache residency, worker affinity, batch size and data alignment. Preserve ordered FP32 accumulation, canonical NaN, signed-zero/subnormal rounding and overflow fallback.
3. Under an exclusive GPU lease and no foreign KFD processes, compare real CPU combine into mapped host RAM versus GPU-local DMA-BUF, separately timing arithmetic, stores, DMA_BUF_IOCTL_SYNC and CPU/GPU publication. Do not remove cache-coherency fences.
4. Advance to production-equivalent fused Radiance GPU consumer and overlapped TP2 prefill/decode with exact-once semantics, quality parity and p95/p99 controls. Keep the existing mapped-host GPU-pull fallback; do not enable RAD51.

Only the S1 **CPU-only correctness/measurement capability** is ready for review; the S1 performance optimization is not done.
