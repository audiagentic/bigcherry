# 1269 PRBE55 Vulkan small-N DMMV

Untested, opt-in AMD Vulkan dispatch experiment. `BIGCHERRY_VK_SMALLN_DMMV=1` can force baseline-MMVQ N=2..8 vector matmuls through the existing DMMV fallback; default routing is unchanged.

## Validation

Contract `PRBE55-VK-SMALLN-DMMV`. Execution target is Brutus RADV on gfx1100 and gfx1201. Require subject activation, no control activation, greedy token parity, and `improvement_no_regression_v1`; gfx1030 is a non-gating compatibility check unless separately declared.
