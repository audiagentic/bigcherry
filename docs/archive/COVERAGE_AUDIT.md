# Testable-option audit

What the tuner could choose between, what it could not, and why. See
[FAMILY_MODEL.md](../reference/architecture/FAMILY_MODEL.md) for why these
families exist, their identity rules, and what was rejected.

> **Historical coverage snapshot.** This document records the tuner surface
> observed for the cited HI34/2026-08-17 work. It is not current acceptance
> policy or a canonical selector inventory. Before acting, verify current
> candidates and selectors against repository code, patch metadata, and
> configuration.

---

## Collected — the tuner can choose between these

| Family | Dimensions | Source |
| --- | --- | --- |
| MMQ | `(kernel_type, J, impl_fallback)` and the whole config row it implies | parsed from the architecture's `CASE` table |
| MMVF | `(kernel_type, width, block_size, accumulator)` | block sizes 32..256 step 32; accumulator F32/F16 for F16 sources only |
| MMF | `(kernel_type, width, nwarps)` | widths 1..16, nwarps 1..8 |
| MMVQ | `(kernel_type, width, nwarps, rows_per_block, small_k)` | generated instances, bounded matrix |
| BLAS | native fallback plus one structured forced-native plan; provider/API remain observations | HI17 BLAS-1 |
| **family itself** | MMQ vs MMVQ vs MMVF vs MMF vs BLAS | **unfused signatures only** — see below |

The per-type token in these names is legitimate: these kernels are compile-time
specialised, so `q8_0` names a distinct compiled artifact rather than the
incoming tensor. Read it as `kernel_type`. The same reasoning does **not** hold
for BLAS, which is one library call — see
[FAMILY_MODEL.md](../reference/architecture/FAMILY_MODEL.md).

**Family choice is the biggest lever.** Upstream selects family from a
heuristic ladder (`ggml_cuda_should_use_mmq` and friends) tuned on other
hardware and other shapes. Measuring whether that heuristic is right for a
given signature on a given GPU is worth more than any within-family knob.

For **fused** signatures the restriction is correct and stays: standards 11.1
makes a fused pattern a distinct semantic operation, and comparing fused MMVQ
against an unfused MMQ decomposition is a graph-level question the matmul
tuner is not entitled to answer.

## Deliberately bundled, not separately tunable

MMQ's `nthreads`, `occupancy`, `I`, `sram_layout`, `k_vram` and `stream_k` are
not independent choices — they arrive together as one `CASE` row keyed by
`(type, J, fallback)`. Forcing J selects all of them, which is why the stable
name carries the whole resolved config (standards 2.2) rather than J alone.

MMF's `rows_per_block` is architecture-determined (`MMF_ROWS_PER_BLOCK` vs
`MMF_ROWS_PER_BLOCK_CDNA`), not a free choice.

MMVQ fusion is a runtime branch inside `mul_mat_vec_q_switch_fusion`; one
compiled instance serves both. It belongs to the signature, not the candidate.

## NOT collected — real gaps, each with a reason

**hipBLAS internal algorithms and additional BLAS arms.** The initial BLAS-1
catalog carries only the native fallback and one structured forced-native plan.
Provider/API alternatives remain unenumerated until a runtime apply/execute
seam proves they are executable; enumerating hipBLASLt solutions will also
require namespacing results by exact library version (standards 16.2). Three
of the hidden choices are llama.cpp's own heuristics, not library internals —
see [FAMILY_MODEL.md](../reference/architecture/FAMILY_MODEL.md#blas-decomposition):

- **`compute_type`** (F32/F16/BF16) chosen by `fast_fp16_hardware_available`
  and gated by `GGML_PREC_F32`. Upstream already ships a global env override.
- **`output_conversion`**: `prefer_f32_output` grants RDNA4/CDNA/Volta direct
  F32 output and denies it to RDNA3.
- **`api_strategy`**: four `cublas*` entry points selected by hard if/else
  chain, above a comment admitting the heuristic is a guess about "some old"
  GPUs.
- **`provider_policy`**: `ROCBLAS_USE_HIPBLASLT` — genuinely library-internal,
  and the only one of the four that needs version namespacing.

**Multi-device allreduce.** Three implementations exist (`nccl`, `internal`,
`butterfly`); one is chosen at backend construction and never revisited. On
tensor-split configurations this is on the path of every split matmul. **HI18**
makes it a family.

**FlashAttention.** A different operation, not a matmul family — it needs its
own signature space and collection points. Not currently tuned, and out of
scope for the matmul tuner. (WMMA is rejected as a family — see
[FAMILY_MODEL.md](../reference/architecture/FAMILY_MODEL.md).)

**Signature refinements.** `alignment_class`, `occupancy_bucket` and
`offset_modulo` are defined in the ABI but never populated — `has_refinements`
is always 0. That follows standards 5.5 (promote a refinement only once
measurement proves it changes the winner), but it is circular: nothing can
prove it while nothing populates them. Breaking the circle needs a deliberate
experiment that populates one refinement and compares winners either side of it.

**The transposed-vector MMVF path.** `ggml_cuda_mul_mat` swaps operands and
synthesises a `dst` for `ne01 == 1 && ne11 > MMVF_MAX_BATCH_SIZE`. HI04
declines it because its launch context is not the one the signature describes,
so the shape is never tuned. Needs either explicit modelling or a recorded
decision not to.

**MoE occupancy bucketing.** `MUL_MAT_ID` is collected, but routing density
does not reach the key, so two very differently-loaded expert dispatches share
a signature and therefore a winner.

**Non-latency objectives.** The dispatch key carries an `objective` field and
everything is measured against `latency`. Throughput and
workspace-constrained objectives are expressible but unused.

## How to tell, at a glance, whether a run measured enough

The tuner records `generated / eligible / measured` per signature plus a reason
for every rejection. Those three numbers being far apart is the signal:

- `generated` high, `eligible` low → an eligibility predicate is too strict, or
  the candidate set never matched the observed shapes
- `eligible` high, `measured` low → launches or correctness checks are failing
- both close and small → the catalog genuinely has little to offer here

Check them before believing a winner. "We tuned it" without these is an
assumption.

## Measurement context — cache residency (HI34 B1, decided 2026-08-17)

The question: does measuring candidates back-to-back on the same operands (hot
L2/Infinity Cache) distort *which* candidate wins, compared to a cold-cache
measurement? HI34 step 4 ran the locked three-arm experiment on Brutus (7900
XTX, gfx1101, ROCm 7.2.4, unblocked): H1 (flush=0) -> C (flush=1, 256 MB
eviction between samples, outside the timed window) -> H2 (flush=0), all at
lps=1, full MUL_MAT matrix, 1156 signatures per arm.

**Answer: measurement context changes the selected winner on this hardware.**
Gate 1 passed (H1 vs H2: 3 winner flips in 1156, 0.26%). Gate 2 failed: four
signatures show true material median crossovers — the hot-cache winner
(`blas:native`) beats the cold-cache winner (`mmvf:native`) on hot context by
+5.7%..+24.4%, and the ordering inverts on cold context, where the cold winner
beats the hot winner by +28.7%..+50.6%. One caveat limits attribution: the
absolute shifts are not uniformly in the eviction direction — on two of the
four, mmvf is much FASTER in the cold arm (102 -> 59 us), and on one, blas is
slower (26 -> 36 us) while on another it is flat. Sustained 256 MB eviction
writes before every sample plausibly precondition GPU clock/power/memory-fabric
state, so the intervention perturbs more than the cache. B1 therefore
establishes **measurement-context sensitivity**, not yet "L2 residency caused
this"; attribution needs the cause-validation sweep before any re-tuning or
promotion decision (HI65). Four further replicated winner flips do NOT cross
on medians — in the cold context the hot winner stays faster or ties — so they
are non-crossover selection flips, consistent with the elevated cold-arm noise
(MAD inflation under flush). They are NOT diagnosed as confirmation-noise
artifacts: the production winner is not selected from raw GPU median alone
(effective_us = max(gpu_median, host_median - sync_overhead), then a fresh
confirmation holdout can change the final winner), and the evaluator did not
check the confirmation stage for these four. A secondary effect: under flush,
63 signatures rejected fail-closed as "native timing unstable" (MAD inflation
on small kernels), so the cold arm also changes the noise regime, not just the
means.

**Attribution caveat (does not change the verdict).** The absolute shifts are
not uniformly in the cache-eviction direction: on some crossover signatures
the cold arm is *faster* for both candidates (e.g. `38ee4d70...`: blas
83.4->76.4, mmvf 102.0->59.3 us; `f33c9997...`: mmvf 102.1->59.8 us), while
on others the blas path is slower under flush (`ee8a2c18...`: 25.96->36.2 us,
+39%). Sustained 256 MB eviction writes before every sample plausibly
precondition GPU clock/power state, so the intervention perturbs more than the
cache. B1 therefore establishes **measurement-context sensitivity** (rankings
move with the measurement context) but does NOT yet attribute the effect to
L2/Infinity-Cache eviction per se.

**Consequence.** Recorded winners of those four crossover signatures from hot
tuning are suspect in the cold direction, but which direction is
"more production-true" is exactly what the attribution work must settle. The
experiment measures a bound, not a correction — production sits between the
two extremes and depends on the graph, so the flush is NOT promoted to
calibration default on this evidence. `flush=1` stays OFF by default and
remains a diagnostic knob. Before any promotion or re-tuning decision, the
eviction must be validated as the cause: size-saturation sweep (128/256/512 MB)
on a diagnostic subset plus an evict-then-rewarm control and DVFS/thermal/
memory-fabric preconditioning ruled out — tracked as HI65, not a flag flip.

Evidence: `artifacts/b1/{h1,cold,h2}.jsonl.measurements.jsonl` (arm headers
carry `flush_l2`/`flush_evict_mb` provenance); strict gate evaluator at
`tools/residency_gates.py` with unit tests in
`tools/tests/test_residency_gates.py` (promoted from
`tmp/b1-gate2-strict.py`, 2026-08-17). Plan item HI34 steps 4-5, notes of
2026-08-17; follow-on attribution item HI65.

