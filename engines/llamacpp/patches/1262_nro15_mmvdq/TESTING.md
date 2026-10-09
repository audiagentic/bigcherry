# Testing — NRO15 MMVDQ

## Lane applicability
Adds dequant-float single-token matvec for contiguous, non-batched Q4_K/Q5_K/Q6_K weights, bypassing MMVQ's Q8_1 activation quantization. It is opt-in on gfx1100. Qwen3.8-27B-Q8_0 uses Q8_0 weights, outside this patch's routed K-quant types; MTP verify batch is 5 columns rather than the single-token route. **Applicable to this lane: no — not applicable to this lane.** Existing status is rejected; testing below is the qualification shape if reconsidered on a matching K-quant lane.

## Hardware-free
- `PYTHONPATH=tools python -m bigcherry patch-rebase-check --source bigcherry --focal-overlay 1262_nro15_mmvdq`
- Unit/backend expectations: apply/idempotence; missing anchor fails closed; Q4_K/Q5_K/Q6_K single-token route only; batched/prefill and Q8_0 remain control route; environment gates and row values behave fail-safe.

## Activation
With `BIGCHERRY_PATCH_TRACE=1`, require exact marker: `BIGCHERRY_PATCH_HIT patch=1262_nro15 path=mmvdq`. On this Q8_0/MTP lane the marker should not fire; firing would be an applicability/routing defect.

## Correctness
For a matching K-quant qualification lane, require control/subject MTP logprob max abs diff <= `5e-4`, report any greedy divergence, and require draft-acceptance parity to standard control `0.90101`. For this Q8_0 lane, record N/A rather than treating non-activation as parity evidence.

## Performance
Only on a matching routed lane: order-balanced `bigcherry ab-benchmark --server-config`, one variable, `pp1024`, `pp4096`, `tg512`, `tg2048`, Mann-Whitney. Prefill is expected control/non-routed. Do not use this Q8_0 lane to promote or re-evaluate the rejected patch.

## Promotion
Any reconsideration requires a fresh `NRO15-MMVDQ-KQUANT-DECODE` contract campaign and `patch-verify-evidence` validated-evidence on a matching K-quant workload; ad-hoc A/B is insufficient.
