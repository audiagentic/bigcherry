# Testing — RD12 paired MMVQ dual output

Target candidate path: Qwen3.8-27B-Q8_0, dual 7900 XTX/gfx1100, `-sm tensor`, MTP `n_max=4`.

## Activation
Require subject hit/control miss for `BIGCHERRY_PATCH_HIT patch=1205_rd12 path=dual_output_mmvq_fusion` under `BIGCHERRY_PATCH_TRACE=1`.

## Correctness
Keep the existing exact-pattern bit-identical producer. For this target MTP lane, additionally compare pre-sampling full-vocabulary logprobs control vs subject with max absolute difference <= `5e-4` where the validation contract demands it, plus greedy-token parity. Cover near-miss/fallback graphs and safe view/data-interval gates before any promotion argument.

## Work equivalence
Record drafted/accepted counts and acceptance ratio for every arm; require MTP draft-acceptance parity before attributing server throughput.

## Performance
Run fixed-work `llama-bench` first, then the applicable order-balanced `tools/lab/native-vs-patched/server-ab-*.json` single-patch arm, alternating order and retaining raw pairs. Do not combine with conflicting 1207. Existing correctness/activation evidence does not constitute performance validation or sign-off.
