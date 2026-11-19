# Retain the helper’s E4M3 MoE rows

Flash-Next’s W4 helper looks up `had_quant_act_fp8` at startup. The 35B helper and the gfx1201 decode plugin use `moe_gemm_fp8a8` and its block-scaled row-major layout. Upstream removed the two row declarations in its 2.0 release, while retaining the templated launches, constraints and operand descriptions. This patch registers the E4M3 forms on those implementations. The layout validates the exact code and scale planes for each expert class before serving the container’s existing bytes.

The native RDNA3 provider has separate row coverage. Hardware numerics, selection and throughput on the new upstream revision require separate qualification.
