# 1272_ar_host_compressed_wire

**Status:** untested
**Plan item:** PGC07

Selectable host-AllReduce wire format for the upstream b11233 two-GPU internal provider.
Unset `GGML_CUDA_AR_WIRE` preserves pristine BF16-threshold selection; explicit
`f32|bf16|f16|q8_0` applies to both mapped-host chunked and copy-engine paths.
Q8_0 uses 32-element blocks with fp16 scale and fused receiver dequantization + F32 accumulation.

Validation required: apply, HIP build on gfx1100, activation marker, correctness, and paired performance A/B.
1250 depends on 1272 (and rejected 1252); current patch metadata declares no 1250/1272 conflict. The Q8-only 1250 fused path is not adopted and does not provide exact-F32 CPU-root fusion. 1244 root3 remains untouched. See PNRO02/PGC11/BCOP212.
