# 0840_hybrid_allreduce_dispatch: HI155 size-adaptive internal/RCCL AllReduce provider dispatch

**Status:** untested
**Group:** core
**Plan item:** HI155

## What it does

Adds `GGML_CUDA_ALLREDUCE=hybrid`: a fourth provider that brings up both
RCCL and the internal AllReduce pipeline (patch 1001) simultaneously, then
picks per call based on `ggml_nbytes(tensors[0])` against the internal
pipeline's own real copy-engine threshold -- below it, tries internal
first (falling through to RCCL on failure); at or above it, goes straight
to RCCL. `comm_ctx->provider_name` is set per call right before each
sub-provider runs, so the existing 0830 telemetry seam attributes every
call correctly (`effective_provider`: "internal" or "rccl") with no
separate labeling change needed.

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

## Upstream / provenance

Local, BigCherry-authored (not an upstream port) -- plan item HI155,
opened following gpt-dev-agent's explicit design guidance (dev-gpt-agent
gateway session `ses_5307d9c58ec645cb`) after the real prefill-regression
and reduction-byte-histogram evidence in
`patches/1001_hip_internal_allreduce/SUMMARY.md` and project memory.
Requires 1001 (the internal pipeline this dispatches into). Not yet
validated on real hardware -- see HI155 for the remaining slices
(correctness probe across both sides of the threshold, mixed prefill+decode
end-to-end A/B).
