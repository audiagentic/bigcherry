---
id: BCOP332
order: 332
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-20'
created-by: agent
priority: P3
work: S
---

# QFP44: gfx1201 MMQ k01 compiler-spill admission

**Discovery:** [llama.cpp #29536](https://github.com/ggml-org/llama.cpp/pull/29536), open and updated 2026-10-19 18:30 UTC, reports stock LLVM 23 RDNA4 MMA loop spilling (5/64 IQ4_XS instantiations; reported 2,387 VGPR spills before, zero after) and external R9700 Q4_K_M pp ~298→~1,250 t/s when paired with other compiler compatibility fixes. Not BigCherry, not isolated IQ4_XS gain. The pinned b11552 shared `ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma` K loop is a testable compiler-specific target; four K fragments and all work remain required.

**Owner:** Existing QFP44 (one loop/codegen sub-slice); no new technical plan. `0300` J selection, `1237/1265` compact grid, QFP37 Stream-K, PRBE110 Q6_K and PRBE111 IQ MMVQ remain separate. BCOP300 compact-grid review is not repeated. No code, patch, selector, allocator, scheduler, telemetry or hardware queue changed.

**12-hour/novelty:** Selected `feat/rad14-fp8-gemv` from independent Audumla `d5b1202a2584` (2026-10-19 19:08:19 UTC), newest genuine development branch; `main` at `62f6139f`. Repository-visible 12-hour commits/PRs advance RAD14 FP8 GEMV, RAD13 FP8 grouped GEMM and RCS09/RCS05 runtime helper geometry; protected and untouched. QFP44 last independent plan edit 2026-10-08 06:47 UTC, 0300 last source edit 2026-10-12 01:49 UTC. No independent matching MMQ k01 codegen/ISA experiment, PR or recorded queue in preceding 12 hours. Off-repository hardware queues unverified. Novelty is *new stock LLVM 23 measured spill/codegen evidence*, distinct from prior MMQ compact-grid work.

**Completed:** Inspected b11552 MMA loop, shared IQ4_XS/Q8_0 dispatch, RDNA4 J table, 0300 preallocation J gate, #29536 diff/review, FireBurn fork scope, related plan ownership. Host-only 8,192 modeled J-selection cases and 32/32 K coverage passed; Python compilation passed. No HIP build, profiler, backend-op test or GPU benchmark.

**Action/terminal:** First read-only compiler/version, executed IQ4_XS MMQ/J and spill/critical-path census. `NO_IQ4XS_MMQ_ROUTE`, `NO_SPILL`, `NO_CODEGEN_WIN`, `BELOW_CEILING`, `PIN_ABSORBED` or `TOOLCHAIN_BLOCKED` → close without patch. Only if eligible: one gfx1201-only one-loop candidate, compile/ISA and full-work/correctness gate, then paired E2E CI95-low >=3% with <=1% unaffected regression; otherwise `NUMERIC_FAIL`, `REGRESSION` or `NO_GAIN`. Full technical discriminator lives in QFP44, not this ledger.
