# 1272 — host AllReduce compressed wire

PGC07 tests explicit wire formats for llama.cpp b11233's existing two-GPU internal pinned-host AllReduce without changing provider selection or routing decode traffic to the copy engine.

## Behaviour

`GGML_CUDA_AR_WIRE` unset is byte-for-byte control-flow compatible with pristine b11233: F32 inputs continue to use `GGML_CUDA_AR_BF16_THRESHOLD` (default 1). Explicit `f32`, `bf16`, `f16`, or `q8_0` overrides the wire type on both internal paths. The copy-engine decision is made from the selected wire payload size; decode-sized reductions remain on the mapped-host chunked kernel.

Q8_0 uses upstream `block_q8_0`: 32 values, fp16 scale, int8 payload. Each rank quantizes its own contribution; the receiver dequantizes both local and peer blocks and performs the sum in F32 before casting to the tensor destination type. Tail elements are zero-padded inside the final block.

Under `BIGCHERRY_PATCH_TRACE`, the first explicit-wire execution in a process emits `BIGCHERRY_PATCH_HIT patch=1272_ar_wire path=ar_wire_<fmt>`.

## Composition

This patch changes `ggml/src/ggml-cuda/allreduce.cu` only. It does not modify 1244's N=3 root3 branch: when 1244 is composed, that branch returns before 1272's N=2 override dispatch. It conflicts with `1250_nro01_allreduce_q8_wire`, which owns overlapping AllReduce wire-format machinery.

## Validation

State is **untested**. Required evidence is declared in `validation.toml`: deterministic apply, HIP build, explicit-wire activation, correctness, and paired performance on gfx1100. The checked-in offline mechanics test applies against the pristine b11233 fixtures, covers all four format branches and both Q8 transport paths, verifies idempotence, and checks anchor mutation fails closed.
