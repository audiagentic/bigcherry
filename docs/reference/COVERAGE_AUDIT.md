# Testable-option audit

What the tuner can actually choose between, what it cannot, and why. See [FAMILY_MODEL.md](FAMILY_MODEL.md) for why these families exist, their identity rules, and what was rejected.

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
for BLAS, which is one library call — see [FAMILY_MODEL.md](FAMILY_MODEL.md).

**Family choice is the biggest lever.** Upstream selects family from a heuristic
ladder (`ggml_cuda_should_use_mmq` and friends) tuned on other hardware and
other shapes. Measuring whether that heuristic is right for a given signature on
a given GPU is worth more than any within-family knob.

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

**hipBLAS internal algorithms and additional BLAS arms.** The initial BLAS-1
catalog carries only the native fallback and one structured forced-native plan.
Provider/API alternatives remain unenumerated until a runtime apply/execute
seam proves they are executable; enumerating hipBLASLt solutions will also
require namespacing results by exact library version (standards 16.2). Three
of the hidden choices are llama.cpp's own heuristics, not library internals —
see [FAMILY_MODEL.md](FAMILY_MODEL.md#blas-decomposition):

- **`compute_type`** (F32/F16/BF16) chosen by `fast_fp16_hardware_available`
  and gated by `GGML_PREC_F32`. Upstream already ships a global env override.
- **`output_conversion`**: `prefer_f32_output` grants RDNA4/CDNA/Volta direct
  F32 output and denies it to RDNA3.
- **`api_strategy`**: four `cublas*` entry points selected by hard if/else chain,
  above a comment admitting the heuristic is a guess about "some old" GPUs.
- **`provider_policy`**: `ROCBLAS_USE_HIPBLASLT` — genuinely library-internal,
  and the only one of the four that needs version namespacing.

**Multi-device allreduce.** Three implementations exist (`nccl`, `internal`,
`butterfly`); one is chosen at backend construction and never revisited. On
tensor-split configurations this is on the path of every split matmul. **HI18** makes it a family.

**FlashAttention.** A different operation, not a matmul family — it needs its own
signature space and collection points. Not currently tuned, and out of scope for
the matmul tuner. (WMMA is rejected as a family — see [FAMILY_MODEL.md](FAMILY_MODEL.md).)

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
shape is never tuned. Needs either explicit modelling or a recorded decision not to.

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
