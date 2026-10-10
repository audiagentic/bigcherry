# 1273: IQ4_XS / IQ3_XXS single-token MMVQ launch tuning (RDNA)

## Scope

`ncols_dst == 1` MMVQ for IQ4_XS and IQ3_XXS on gfx1100 and gfx1201 only. Two independent, default-off
env gates: `BIGCHERRY_IQ_MMVQ_VDR=1` (lower-VDR vec-dot split) and `BIGCHERRY_IQ_MMVQ_NWARPS=1`
(per-architecture block width). With both unset the pinned launch geometry and vec-dot entry points are
unchanged. Quant math is unchanged in every arm.

## Activation

`BIGCHERRY_PATCH_HIT patch=1273_iq_mmvq path=<vdr|nwarps|vdr_nwarps> type=<iq4_xs|iq3_xxs> arch=<gfx1100|gfx1201>`
(once per process with `BIGCHERRY_PATCH_TRACE=1`).

## State

`evaluated`: run on hardware with results recorded, not qualified. No experiment contract is bound yet,
so there is no validation adapter (one returns together with a contract) and no promotion claim is possible.
See SUMMARY.md for the candidate matrix and the rocprofv3 capture plan that decides the next arm.

## 2026-10-10 BCOP111 numerical warning

IQ3_XXS VDR1 rounds each integer half before reduction; pristine VDR2 rounds the combined integer sum once. These are not algebraically equivalent: a compiled 100,000-tuple synthetic C++ fixture differed 37,433 times (max integer delta 1). **Not a packed-quant or GPU mismatch measurement.** This evaluated patch has no auditable numerical hardware receipt. Require packed-IQ3/Q8_1 parity before any VDR speed/promotion run. Qualify IQ4 VDR2 and nwarps-only independently. The once-per-process trace marker is not a per-request launch count. See PRBE111/BCOP111.
