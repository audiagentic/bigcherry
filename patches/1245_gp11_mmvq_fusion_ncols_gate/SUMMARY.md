# 1245: Widen the MMVQ MUL_MAT+GLU fusion gate beyond ncols_dst==1

**Status:** untested — measured NEGATIVE on real hardware, do not promote
**Group:** gpu-collectives
**Plan item:** GP11

Origin is local (bigcherry-original): this relaxes an upstream restriction, it
is not a port of any upstream commit, so there is no external commit to cite as
provenance.

## What it does

Adds a fused MMVQ specialization for exactly `(ncols_dst == 6 && type ==
GGML_TYPE_Q8_0)` alongside the existing `ncols_dst == 1` case, and lets the
host-side selector choose it on gfx1100. Three edits, all in the
eligibility/instantiation layer — none in the fused kernel's arithmetic:

| file | edit | change |
|---|---|---|
| `mmvq.cu` | `instantiate-fusion-q8_0-ncols6` | `if constexpr (c_ncols_dst == 1)` → `c_ncols_dst == 1 \|\| (c_ncols_dst == 6 && type == GGML_TYPE_Q8_0)` in `mul_mat_vec_q_switch_fusion` |
| `mmvq.cu` | `widen-dense-fusion-prologue-assert` | `GGML_ASSERT(ids \|\| dst->ne[1] == 1)` → `<= MMVQ_MAX_BATCH_SIZE` |
| `ggml-cuda.cu` | `widen-mmvq-fusion-eligibility` | `ggml_cuda_should_fuse_mul_mat_vec_q`: `dst->ne[1] != 1` → `dst->ne[1] == 1 \|\| (dst->ne[1] == 6 && src0->type == GGML_TYPE_Q8_0 && GGML_CUDA_CC_IS_RDNA3_0(cc))` |

The `GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1")` after
the first edit is left in place: the fused branch returns before reaching it
whenever fusion is active, so it only evaluates with `!has_fusion` and passes.
(gpt suggested replacing it with a bare `GGML_ASSERT(!has_fusion)` since the
message is now stale documentation; not done because this project's patcher
blanks string literals before anchor matching, making that line awkward to
target. Worth doing if this patch is ever revived.)

MoE (`MUL_MAT_ID`) and the MMVF path keep their `ncols_dst == 1` gates.

## Why it was tried

rocprofv3 decode capture, dual gfx1100, Qwen3.8-27B-Q8_0, `-sm tensor`, MTP
`spec_draft_n_max=5`; 5.9 s decode window, 632,620 dispatches, union-of-spans.

MMVQ template is `mul_mat_vec_q<type, ncols_dst, has_fusion, small_k, halve_iters>`:

| dispatches | ncols | has_fusion |
|---|---|---|
| 112,110 (91%) | 6 | false |
| 7,848 | 1 | false |
| **1,120** | 1 | **true** |

Fusion fired on **0.9%** of MMVQ dispatches, only ever at ncols=1, while the
workload sits at ncols=6 (MTP verifies 5 drafted tokens + 1). Corroborating
detail: all 20,561 same-shape consecutive-MMVQ pairs per GPU carried a
`quantize_q8_1` between them rather than being one fused op, 7,345 of them were
immediately followed by `unary_gated_op_kernel` (FFN gate/up SwiGLU), and 30,500
separate `unary_gated_op_kernel` dispatches existed that fusion would absorb.

The same `ncols_dst == 1` assumption is why patch 1241 (RD33) measured null —
its own gate is `ncols_dst == 1`, so it never fired on ~93% of MMVQ work either.

## RESULT (2026-09-04, real hardware): NEGATIVE

Both arms built and profiled on brutus with an identical workload (same model,
flags, prompt, seed=42, temp=0, n_predict=200), both under rocprofv3.

Build and instantiation gates all **passed**:

| gate | result |
|---|---|
| compiles | yes |
| code size vs control | +38,512 B (+0.022%) |
| `mul_mat_vec_q<Q8_0,6,true>` instantiated | 35 in patched, **0** in control |
| pre-existing ncols=1 fused kernel | unchanged (105 in both) |
| VGPR / SGPR / **scratch** | 60 / 50 / **0** (unfused ncols=6: 44 / 30 / 0) |

No spill — the explicit abandon signal did not trigger.

The structural prediction was confirmed **exactly**:

| metric | control | experiment | delta |
|---|---|---|---|
| MMVQ dispatches | 75,836 | 67,386 | −8,450 (−11.1%) |
| quantize_q8_1 dispatches | 75,836 | 67,386 | −8,450 |
| unary_gated_op dispatches | 19,082 | 10,632 | −8,450 (−44.3%) |
| quantize_q8_1 time | 100.4 ms | 89.5 ms | −10.9 ms |
| unary_gated_op time | 30.1 ms | 18.6 ms | −11.5 ms |
| **MMVQ kernel time** | 3,470.4 ms | **4,090.8 ms** | **+620.4 ms (+17.9%)** |
| **throughput** | **51.54 tps** | **46.70 tps** | **−9.4%** |

8,450 real fused dispatches fired (zero in control); 25,350 total dispatches
removed. **And it still lost** — the fused kernel's added cost swamps the ~22 ms
saved across quantize and SwiGLU combined.

## Why this matters beyond this patch

1. **Removing dispatches is not sufficient.** ~8% of all decode dispatches were
   eliminated and throughput got *worse*. Any future "reduce kernel count"
   argument on this workload must show the replacement kernel's cost, not just
   the dispatch delta.
2. **This patch's own occupancy reasoning was wrong in practice.** An earlier
   revision argued 60 VGPRs < the ~96 needed for full occupancy on gfx1100
   wave32, so 44→60 should be free. The +17.9% MMVQ time says otherwise.
   Candidate mechanisms, not distinguished here: real occupancy loss, weight-cache
   locality lost by streaming two weight matrices in one kernel, or the
   per-column `tmp_gate[6]` accumulators. Spec-sheet register arithmetic was not
   a substitute for measurement.
3. An earlier revision also cited the kernel comment *"Block: (warp_size,
   ncols_dst) — each warp handles one token independently"* as evidence of
   ncols-safety. **That comment belongs to the separate MoE kernel, not dense
   MMVQ**, and the citation was wrong (caught in gpt review, req_6d9ac5777c094cea).
4. Upstream's `ncols_dst == 1` restriction now looks like a defensible
   performance choice on this architecture, not merely unvalidated conservatism.

## Confidence and limits

Throughput is a **single sample per arm**, which this project's own notes warn is
unreliable alone. But the kernel-time evidence (+620.4 ms summed MMVQ across
67k–76k real dispatches per arm) is not a single-sample artifact and points the
same way, and −9.4% is an order of magnitude above the established 0.5–0.9%
repetition noise floor. A paired/interleaved re-run would tighten the figure but
is unlikely to flip the sign; it was not run because the direction is already
clear and unfavourable.

Both arms ran on `bigcherry-native` (no patch 0840), so `GGML_CUDA_ALLREDUCE=hybrid`
was inert. Fine for an A/B of this patch — both arms share that base — but these
absolute tps values are **not** comparable to production numbers.

Correctness (`test-backend-ops` tolerance comparison at ncols>1) was **not** run:
the performance result made it moot.

**Disposition:** keep at `state=untested`, do not promote, do not add to any
patch-set. Retained as a documented negative result so the `ncols_dst == 1` gate
is not "discovered" and re-attempted a fourth time.
