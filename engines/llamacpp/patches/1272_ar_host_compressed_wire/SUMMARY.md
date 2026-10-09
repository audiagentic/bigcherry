# 1272_ar_host_compressed_wire

**Status:** untested
**Plan item:** PGC07

Selectable host-AllReduce wire format for the upstream b11233 two-GPU internal provider.
Unset `GGML_CUDA_AR_WIRE` preserves pristine BF16-threshold selection; explicit
`f32|bf16|f16|q8_0` applies to both mapped-host chunked and copy-engine paths.
Q8_0 uses 32-element blocks with fp16 scale and fused receiver dequantization + F32 accumulation.

Validation required: apply, HIP build on gfx1100, activation marker, correctness, and paired performance A/B.
Conflicts with `1250_nro01_allreduce_q8_wire`; 1244 root3 is intentionally untouched.
