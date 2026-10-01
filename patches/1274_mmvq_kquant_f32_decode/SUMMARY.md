# 1274_mmvq_kquant_f32_decode

**Status:** validated
**Plan item:** RD33

Extends the validated 1241 dense single-token F32-activation MMVQ path from Q8_0 to Q6_K on gfx1100 (Q4_K was measured and dropped: -1.7% decode on one XTX, flat on two; Q6_K +1.6%..+1.7% decode). It requires 1241 and reuses its `f32_act` kernel/launcher seam, so Q8_0 remains owned by 1241 while 1274 adds only the K-quant helpers and dispatch gate.

For eligible dense `ncols_dst == 1` calls, Q6_K keeps the native MMVQ weight-lane unpacking but dot directly against the original F32 activation. This removes the per-call Q8_1 activation allocation, quantization kernel, and quantized activation write/read. The tradeoff is extra per-weight-block unpack plus F32 FMAs. Q4_K/Q6_K are the first probes because their unpack is relatively direct; Q5_K is deferred because its additional high-bit plane increases unpack/register cost without increasing the fixed activation-quantization saving.

Validation required: compose after 1241, HIP build on gfx1100, Q6_K CPU-reference correctness, marker activation only for dense width-1 decode, forced-candidate/non-width-1 negative controls, and paired performance A/B before promotion.
