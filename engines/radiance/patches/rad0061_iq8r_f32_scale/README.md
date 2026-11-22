# rad0061 — diagnostic IQ8R F32 scale fold (RUM15)

Opt-in native Radiance 2.2 source patch for evaluating precision, memory admission
and GPU throughput. This patch is **not selected by serving-core**. It widens
only the IQ8R group-256 scale layout to F32 and updates the existing decode
(K-parallel and narrow) and prefill paths in libr4d. The registered gfx1100
libr3 component rewrites this same source into its native WMMA implementation;
no second kernel or loader is introduced.

Requires complete validation on gfx1100 and gfx1201, all 1/16/64-row kernel
variants, source-backed C06 numerical parity and GPU admission before production
selection. ABI/layout fingerprint changes; existing serialized layout plans must
not be reused. libref remains the F32 reference.
