# Testable-option audit

What the tuner can actually choose between, what it cannot, and why. Written
2026-08-05 after finding that the tuner was silently restricted to a single
family.

The question this answers is not "does tuning work" but "when it reports a
winner, what was that winner chosen *from*". A tuner that measured four options
reports its result exactly as confidently as one that measured forty.

---

## Collected — the tuner can choose between these

| Family | Dimensions | Source |
| --- | --- | --- |
| MMQ | `(kernel_type, J, impl_fallback)` and the whole config row it implies | parsed from the architecture's `CASE` table |
| MMVF | `(kernel_type, width, block_size, accumulator)` | block sizes 32..256 step 32; accumulator F32/F16 for F16 sources only |
| MMF | `(kernel_type, width, nwarps)` | widths 1..16, nwarps 1..8 |
| MMVQ | `(kernel_type, width, nwarps, rows_per_block, small_k)` | generated instances, bounded matrix — **enumerated but dispatch unwired**, see below |
| BLAS | one opaque `hipblas-auto` | standards 16.1 — **decomposed by HI17**, see below |
| **family itself** | MMQ vs MMVQ vs MMVF vs MMF vs BLAS | **unfused signatures only** — see below |

The per-type token in these names is legitimate: these kernels are compile-time
specialised, so `q8_0` names a distinct compiled artifact rather than the
incoming tensor. Read it as `kernel_type`. The same reasoning does **not** hold
for BLAS, which is one library call — hence its separate treatment in
[FAMILY_MODEL.md](FAMILY_MODEL.md).

**MMVQ candidates are currently unreachable.** The entry point is gated
(`return false` in `ggml_hip_mmvq_can_execute` for non-native), so a signature
where MMVQ would win records "MMQ won". That reads as a measurement and is an
artifact of what was compiled and reachable — worse than a missing candidate,
because it is indistinguishable from a real result afterwards. HI09 wires it;
HI19 makes the condition visible as `dispatch_status = unavailable` in the
meantime.

**Family choice is the biggest lever.** Upstream selects family from a heuristic
ladder (`ggml_cuda_should_use_mmq` and friends) tuned on other hardware and
other shapes. Measuring whether that heuristic is right for a given signature on
a given GPU is worth more than any within-family knob. Plan 11.3 items 4 and 5
require it; an earlier version of the tuner restricted every signature to
native's family and would have thrown it away silently.

For **fused** signatures the restriction is correct and stays: standards 11.1
makes a fused pattern a distinct semantic operation, and comparing fused MMVQ
against an unfused MMQ decomposition is a graph-level question the matmul tuner
is not entitled to answer.

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

**hipBLAS internal algorithms.** One opaque candidate by design (standards
16.1); enumerating hipBLASLt solutions requires namespacing results by exact
library version (16.2). Quantified cost: on the 27B MTP profile, **19 of 92
signatures** resolve to BLAS, so roughly a fifth of the workload's distinct
operations can only ever pick "whatever hipBLAS decides".

Reviewing the source showed this is not one gap but four, and three of them are
not library-internal at all — they are llama.cpp's own heuristics, which the
opaque candidate was hiding behind hipBLAS's opacity. **HI17** addresses them:

- **`compute_type`** (F32/F16/BF16) is chosen by `fast_fp16_hardware_available`
  and gated by `GGML_PREC_F32`. Upstream already ships a global env override for
  it, so it is unambiguously a free choice — just not a measured one.
- **`output_conversion`**: `prefer_f32_output` grants RDNA4/CDNA/Volta direct
  F32 output and denies it to RDNA3. On the gfx1100 rig, every F16-compute BLAS
  call therefore runs an extra full-output conversion pass.
- **`api_strategy`**: the four `cublas*` entry points are selected by a hard
  if/else chain, above the comment "Theoretically cublasGemmStridedBatchedEx
  would always work … probably because the internal kernel selection logic is
  suboptimal". An admitted guess deciding every single-matrix call.
- **`provider_policy`**: `ROCBLAS_USE_HIPBLASLT` — genuinely library-internal,
  and the only one of the four that needs version namespacing.

**Multi-device allreduce.** Three implementations exist (`nccl`, `internal`,
`butterfly`); one is chosen at backend construction and never revisited. On the
2× XTX tensor-split configuration this is on the path of every split matmul and
none of its cost is attributed. **HI18** makes it a family.

**FlashAttention.** A different operation, not a matmul family — it needs its own
signature space and collection points. Not currently tuned, and out of scope for
the matmul tuner. (rocWMMA appears nowhere in the ggml sources; `amd_wmma_available`
selects code paths *inside* MMQ, MMF and FA, so WMMA is not a tunable family.)

**Signature refinements.** `alignment_class`, `occupancy_bucket` and
`offset_modulo` are defined in the ABI but never populated —
`has_refinements` is always 0. That follows standards 5.5 (promote a refinement
only once measurement proves it changes the winner), but it is circular: nothing
can prove it while nothing populates them. Breaking the circle needs a
deliberate experiment that populates one refinement and compares winners either
side of it.

**The transposed-vector MMVF path.** `ggml_cuda_mul_mat` swaps operands and
synthesises a `dst` for `ne01 == 1 && ne11 > MMVF_MAX_BATCH_SIZE`. HI04 declines
it because its launch context is not the one the signature describes, so the
shape is never tuned. Needs either explicit modelling or a recorded decision not
to.

**MoE occupancy bucketing.** `MUL_MAT_ID` is collected, but routing density does
not reach the key, so two very differently-loaded expert dispatches share a
signature and therefore a winner.

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
