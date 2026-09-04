# 1245: Widen the MMVQ MUL_MAT+GLU fusion gate beyond ncols_dst==1

**Status:** untested
**Group:** gpu-collectives
**Plan item:** GP11

Not built, not run, not measured on real hardware. Origin is local
(bigcherry-original): this relaxes an upstream restriction, it is not a port of
any upstream commit, so there is no external commit to cite as provenance.

## What it changes

Three edits across two files, all in the eligibility/instantiation layer — none
in the fused kernel's arithmetic:

| file | edit | change |
|---|---|---|
| `mmvq.cu` | `instantiate-fusion-for-all-ncols` | `if constexpr (c_ncols_dst == 1)` → `>= 1` in `mul_mat_vec_q_switch_fusion`, so the fused template is emitted for the whole batch range instead of only single-column |
| `mmvq.cu` | `widen-dense-fusion-prologue-assert` | `GGML_ASSERT(ids \|\| dst->ne[1] == 1)` → `<= MMVQ_MAX_BATCH_SIZE` |
| `ggml-cuda.cu` | `widen-mmvq-fusion-eligibility` | `ggml_cuda_should_fuse_mul_mat_vec_q`: `dst->ne[1] != 1` → `dst->ne[1] > MMVQ_MAX_BATCH_SIZE` |

The `GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1")` that
follows the first edit is deliberately **left in place**: with the guard now
always true, the fused branch returns before reaching it whenever fusion is
active, so it only ever evaluates with `!has_fusion` and passes trivially.
Leaving it also avoids anchoring through a string literal, which this project's
patcher blanks before matching.

## Why — measured, not assumed

rocprofv3 decode capture, 2026-09-04, dual gfx1100, Qwen3.8-27B-Q8_0,
`-sm tensor`, MTP `spec_draft_n_max=5`. Real 5.9s decode window, 632,620
dispatches, union-of-timestamp-spans (never summed durations).

MMVQ template is `mul_mat_vec_q<type, ncols_dst, has_fusion, small_k, halve_iters>`:

| dispatches | ncols | has_fusion |
|---|---|---|
| 112,110 (91%) | 6 | false |
| 7,848 | 1 | false |
| 2,000 | 2 | false |
| **1,120** | 1 | **true** |
| 1,010 | 3 | false |
| 1,008 | 4 | false |

**Fusion fires on 1,120 of 123,090 dispatches — 0.9% — and only ever at ncols=1.**

The workload sits at ncols=6 because MTP verifies 5 drafted tokens + 1, so
upstream's fusion is structurally disabled on 91% of decode. The same capture
shows the consequences directly:

- all **20,561** same-shape consecutive-MMVQ pairs per GPU carry a
  `quantize_q8_1` between them (`quantize → MMVQ → quantize → MMVQ`) rather
  than one fused op;
- **7,345** of those pairs are immediately followed by `unary_gated_op_kernel`
  — FFN gate/up SwiGLU, the textbook fusable shape;
- **30,500** separate `unary_gated_op_kernel` dispatches (38.2 ms) exist that
  fusion would have absorbed.

The same `ncols_dst==1` assumption is why patch 1241 (RD33) measured a null
result — its own gate is `ncols_dst==1`, so it never fired on ~93% of MMVQ work
either. One root cause, several stalled items.

## Why widening is believed safe

The fused kernel body is **already generic over `ncols_dst`**: `x_biases[ncols_dst]`
and `gate_biases[ncols_dst]` are sized by the template parameter, and the bias
prefetch, per-column accumulation and GLU combine all iterate
`for (int j = 0; j < ncols_dst; ++j)`. The kernel's own comment reads *"Block:
(warp_size, ncols_dst) — each warp handles one token independently."* Upstream
states the restriction flatly with no rationale; the generic body indicates
conservatism rather than a correctness constraint.

## Scope limits (deliberate)

- **Dense `MUL_MAT` only.** The `MUL_MAT_ID` (MoE) gate is left at `!= 1` — the
  target model is dense, the evidence above is dense-only, and expert routing
  has indexing this change has no evidence for.
- **MMVQ only.** The identical gate on the MMVF path
  (`ggml_cuda_should_fuse_mul_mat_vec_f`, ggml-cuda.cu:1774-1777) is untouched;
  this model is Q8_0 so MMVQ is the path that matters.
- **Bounded by `MMVQ_MAX_BATCH_SIZE` (8)**, which already bounds MMVQ
  eligibility on the line directly above the changed gate, so no ncols_dst
  value becomes reachable that MMVQ did not already handle.

## Validation status and required gates

**Nothing below has been done yet.**

1. **Build** on real gfx1100 via the isolated `gp11-fusion-ncols` experiment.
2. **Compile cost — must be measured, not assumed.** Dropping the `if constexpr`
   guard instantiates the fused kernel for `c_ncols_dst` 1..8 instead of 1, i.e.
   up to 7 extra instantiations per quantization type. Record compile time and
   binary size delta against the control.
3. **Correctness is tolerance-based, not bit-identity.** Fusion changes
   accumulation/rounding versus two matmuls plus a separate GLU, exactly as
   patch 1241 found. Run `test-backend-ops` MUL_MAT for the affected quant types
   against the CPU reference, explicitly including ncols_dst > 1 shapes.
4. **Confirm the path actually executes** (do not infer it from timing): verify
   `has_fusion=true` dispatches now appear at ncols>1 in a rocprofv3 kernel
   trace, and that the `unary_gated_op_kernel` count drops correspondingly.
5. **Confirm MUL_MAT_ID and MMVF are unaffected** — both gates were left intact
   on purpose.
6. **Only then** time it, paired/interleaved against the measured 0.5–0.9%
   repetition noise floor. A single non-interleaved run cannot resolve an effect
   of the expected size — RD33 was misled by exactly that before converging to
   null.

## Expected size

Order **~1% of decode wall** on its own (removing ~7,345 MMVQ + ~7,345
`quantize_q8_1` + a share of 30,500 `unary_gated_op_kernel` dispatches per GPU,
at the measured ~2.85 µs dispatch floor, plus avoided quantize compute and
shared-activation reuse). Modest. Its value is being **one change that unblocks
several stalled items** rather than three separate efforts — not a
transformative win, and it should be judged against that expectation rather than
a hoped-for one.
