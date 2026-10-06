---
id: BCOP45
order: 45
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-07T03:13:00+11:00'
created-by: agent
priority: P1
---

# RNX02 quantized-KV vector attention disposition

## Audit result

RNX02 is the authoritative owner. Existing patch 1298 already rejected the broad Q8-vector MTP-verify override (~6% slower at ~30K); patch 1300 remains only an untested n=1 selector experiment. Current upstream gfx1201 evidence makes vector-kernel parallelism the cheaper discriminator before another selector E2E lane.

## Unresolved action

Run RNX02's bounded gfx1100/gfx1201 Q8 D=256 n=1 profiler + legal `nthreads_KQ_q` sweep. Do not create another FA dispatcher, KV cache mechanism, or tuning registry.

## Terminal disposition

- **Pass:** >=5% vector-kernel improvement on either target with <=2% cross-architecture regression, or prove avoidable tile Q8->F16 staging >=3% decode wall; then qualify 1300 with MTP-off ABBA before MTP interaction testing.
- **Fail:** retire 1300 and close RNX02 dense-attention extraction; preserve 1298 as rejected evidence and leave QSA/1301 independent.

## Dependencies / blockers

No hardware result is claimed by this audit. The next step requires gfx1100/gfx1201 profiling; current active QFP30/FMTP03/MET01 work is outside this item and must not be modified.
