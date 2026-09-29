# Testing — PNRO10 speculative ctx_other devices

Target: Qwen3.8-27B-Q8_0, dual 7900 XTX/gfx1100, `-sm tensor`, MTP `n_max=4`.

## Activation
1261 currently has no trace marker; do not infer activation from build success. Add a future once-per-process `BIGCHERRY_PATCH_TRACE` marker at the point where a backend from `ctx_other` is newly appended to the speculative scheduler device list, e.g. `BIGCHERRY_PATCH_HIT patch=1261_nro10 path=ctx_other_device_added`. Until implemented, activation evidence is BLOCKED and must be stated as such.

## Correctness
Run the bound `PNRO10-SPEC-CTX-OTHER-DEVICES` producer and require greedy MTP token parity. If a full-vocabulary backend-reference/logprob leg is added or required by the contract, compare pre-sampling MTP logprobs with max absolute difference <= `5e-4`.

## Work equivalence
Record drafted/accepted counts and acceptance ratio per arm; require MTP draft-acceptance parity before attributing throughput.

## Performance
Use the applicable order-balanced `tools/lab/native-vs-patched/server-ab-*.json` plan on dual gfx1100. Alternate arm order and retain per-pair evidence. Contract thresholds remain authoritative (`min_paired_rounds=10`, max control regression 1%). No validation or sign-off is claimed here.
