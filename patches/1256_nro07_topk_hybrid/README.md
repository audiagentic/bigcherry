# 1256_nro07_topk_hybrid

Plan `NRO07`; state `untested`; HIP/gfx1100 target.

`b10705` uses full argsort+copy on non-CUB HIP. Nasone provides dedicated TOP_K n-ary/radix/TOP-1 kernels. This first package does not paste the large kernel family directly into live dispatch. It adds the exact ordered-float key conversion and an explicit opt-in policy scaffold (`GGML_CUDA_TOPK_HYBRID`) so source applicability and semantics can be reviewed and fixture infrastructure can land first.

The next implementation stage will port selection kernels and route only fixture-proven shapes, retaining stock bitonic fallback. NRO08 owns the later wave32-native reductions and must remain a separate causal increment.
