# 0840_hybrid_allreduce_dispatch: size-adaptive internal/RCCL AllReduce provider dispatch

**Status:** untested
**Group:** core
**Plan item:** GP03

## What it does

Adds `GGML_CUDA_ALLREDUCE=hybrid`: a fourth provider that brings up both
RCCL and the internal AllReduce pipeline (patch 1001) simultaneously, then
picks per call based on `ggml_nbytes(tensors[0])` against the internal
pipeline's own real copy-engine threshold -- below it, tries internal
first (falling through to RCCL on failure); at or above it, goes straight
to RCCL. `comm_ctx->provider_name` is set per call right before each
sub-provider runs, so the existing 0830 telemetry seam attributes every
call correctly (`effective_provider`: "internal" or "rccl") with no
separate labeling change needed. Init also forces the internal pipeline
to exact F32 (`ggml_cuda_ar_pipeline_force_exact_f32`) rather than
trusting `GGML_CUDA_AR_BF16_THRESHOLD`'s own default (1, BF16 for every
nonzero reduction) -- hybrid's internal side must never silently degrade
to the same lossy wire encoding 1001's evidence shows is a net loss.

## Why

Patch 1001 (validated) is a large win for decode (+17.33% TPS,
MTP completion-bench) but a severe regression for prefill (-32% to -34%,
real llama-bench pp512/pp2048/pp4096) -- `GGML_CUDA_ALLREDUCE` can only
pick one provider for a whole server session, so neither `internal` nor
`rccl` alone is safe to ship as a blanket default. Real HI155-1 telemetry
(0830's new `reduction_bytes` field) captured on real traffic found a
clean, 10x, zero-overlap separation: MTP decode tops out at 1,044,480
bytes, pp2048 prefill is a flat 10,485,760 bytes -- and decode's max sits
almost exactly at `allreduce.cu`'s own default copy-engine threshold
(1,048,576 bytes), meaning decode never reaches the large-message strategy
that the prefill regression is entirely attributable to. Dispatching on
the pipeline's own real threshold (not a second, independently-tunable
constant) keeps this policy from ever routing a call into exactly the
regime the regression evidence implicates.

## Fixes applied 2026-09-02 (GP03 consolidation, gpt-dev-agent review)

This patch is the consolidation target superseding `1243_gp03_size_adaptive_allreduce_dispatch`
(same problem, independently implemented, retired once this patch passes
validation). Four real bugs found by gpt-dev-agent adversarial review, all
fixed:

1. **Missing `requires` on 0830.** `comm_ctx->provider_name` is an
   0830-provided field; `patch.toml` only declared `requires =
   ["1001_hip_internal_allreduce"]`. Fixed: `requires =
   ["0830_split_reduce_telemetry", "1001_hip_internal_allreduce"]`.
2. **Large calls fell to META when RCCL was unavailable, instead of
   internal.** The original dispatcher only tried internal when the
   reduction was below the copy-engine threshold; at/above it, if RCCL
   also wasn't available (NCCL init failed, virtual devices), the call
   fell straight through to `return false` (META) even though internal
   WAS available and is strictly better than META for any size. Fixed:
   internal is now always the last-resort fallback before META, not only
   the below-threshold path.
3. **`GGML_CUDA_AR_COPY_THRESHOLD=0` sentinel inverted.** The internal
   pipeline treats `copy_threshold=0` as "never use the copy-engine, the
   chunked kernel handles every size." The dispatcher's naive `bytes <
   threshold` comparison made that sentinel mean the opposite (never
   route to internal) -- an operator explicitly forcing internal-always
   got the reverse of what they asked for. Fixed with an explicit
   `below_copy_threshold` check treating `threshold==0` as always-eligible.
4. **Explicit-override telemetry mislabel.** `GGML_HIP_REDUCE_PLAN=rccl`
   bypasses this patch's own per-call dispatcher entirely via 0830's
   shared `try_reduce_plan()` rccl branch, which called
   `ggml_backend_cuda_comm_allreduce_nccl()` directly without updating
   `comm_ctx->provider_name` -- in hybrid mode that field was last set at
   init, so an explicit-rccl-forced call could genuinely run RCCL while
   telemetry still reported "internal". Fixed by setting
   `provider_name = "rccl"` in that shared branch immediately before the
   call.

**Still not fixed / not yet safe to ship, even experimentally**: this
patch's `ggml_backend_cuda_comm_init_hybrid()` brings up its own secondary
`ncclCommInitAll()` with zero awareness of GP02's (not yet landed) RCCL
admission predicate or patch 1225's device-3 guard -- confirmed via the
identical test that found this gap in the now-retired 1243: a topology
including physical device 3 will report spurious `ncclCommInitAll` success
and then hard-crash on the first real collective. Do not enable
`GGML_CUDA_ALLREDUCE=hybrid` on any device-3-inclusive topology until GP02
lands and this patch consults it.

## Real hardware validation (2026-09-02) -- CONSOLIDATION COMPLETE

Built from a clean, zero-BigCherry-patches vendor/llama.cpp checkout (pin
b10705, 2578138397d7) in an isolated scratch clone, patches applied and
verified to apply cleanly + idempotently against the true pinned source
(0100_cmake_options, 0830, 1001, this patch). Compared against 5 arms on
{0,1} (2x RX 7900 XTX): native llama.cpp RCCL (zero BigCherry patches --
confirmed the GGML_CUDA_ALLREDUCE selector itself is genuine unpatched
upstream code, a legitimate baseline), native META (`-sm tensor`,
butterfly), layer-split (`-sm layer`), this patch's `hybrid` provider, and
`internal`-only.

**pp/tg synthetic** (llama-bench, `-b 2048 -ub 512`, r=3): hybrid matches
or slightly exceeds native RCCL at pp512 (1478 vs 1465) and tg128 (37.25
vs 34.62), roughly ties native RCCL at pp2048/pp4096. Layer-split beats
BOTH native RCCL and hybrid at pp2048/pp4096 by 10-19% -- confirmed NOT a
defect in this patch (native RCCL has the identical shortfall against
layer-split at the same sizes; this is a real tensor-split-prefill
characteristic of the hardware, present in upstream).

**Real MTP completion-bench** (the metric that resolves the above --
production flags, real `mtp-27b-v1` corpus, 48 requests/arm,
`predicted_tps` mean):

| arm | predicted_tps | vs layer-split | vs native RCCL |
|---|---|---|---|
| layer-split | 44.99 | -- | -31% |
| native-meta | 53.04 | +18% | -19% |
| native-rccl (baseline) | 65.50 | +46% | -- |
| **hybrid (this patch)** | **68.03** | **+51%** | **+3.9%** |
| internal-only | 68.02 | +51% | +3.9% |

Draft acceptance (0.487-0.491) and mean accepted length (2.93-2.95) are
consistent across every arm -- no correctness/behavioral regression in
speculative decoding from any provider choice. Real MTP serving is
decode-dominated (many small verification-reduction calls, not large
prefill-sized ones), which is why layer-split -- the pp-synthetic winner
at large sizes -- is actually the WORST arm under the workload that
matters. Under real MTP serving, this patch beats every baseline including
layer-split by the largest margin of any arm, and beats native RCCL by
3.9% -- a genuine, validated improvement, not a wash.

Full evidence: `artifacts/gp03-validation/` on Brutus (bc-native, bc-hybrid
scratch builds; `mtp-results/*.completion.jsonl` + server/bench logs for
all 5 arms). See GP03's plan notes for the complete writeup.

**Consolidation outcome**: this patch supersedes and replaces
`1243_gp03_size_adaptive_allreduce_dispatch` (deleted 2026-09-02) -- same
problem, independently implemented, retired once this patch reproduced and
exceeded its real hardware numbers under a proper baseline comparison.
This is now the sole implementation for size-adaptive AllReduce dispatch.

## Upstream / provenance

Local, BigCherry-authored (not an upstream port) -- plan item GP03,
opened following gpt-dev-agent's explicit design guidance (dev-gpt-agent
gateway session `ses_5307d9c58ec645cb`) after the real prefill-regression
and reduction-byte-histogram evidence in
`patches/1001_hip_internal_allreduce/SUMMARY.md` and project memory.
Requires 1001 (the internal pipeline this dispatches into) and 0830 (the
`provider_name`/telemetry field this patch reads and writes). Real hardware
consolidation validated 2026-09-02 (see above) -- remaining before
STATE=validated / default patch-set promotion: GP02's admission predicate
must land and be consulted by this patch's secondary `ncclCommInitAll()`.
