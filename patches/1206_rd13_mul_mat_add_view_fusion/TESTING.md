# Testing — RD13 MUL_MAT + RESHAPE + ADD fusion

Target candidate path: Qwen3.8-27B-Q8_0, dual 7900 XTX/gfx1100, `-sm tensor`, MTP `n_max=4`.

## Activation
Require subject hit/control miss for `BIGCHERRY_PATCH_HIT patch=1206_rd13 path=mul_mat_add_view_fusion_(?:f|q)` on a graph/model that actually contains the target pattern.

## Correctness
The bound validation requires backend-reference correctness. For the target MTP lane, compare pre-sampling full-vocabulary logprobs control vs subject with max absolute difference <= `5e-4` where required, and require greedy-token parity.

## Work equivalence
Record drafted/accepted counts and acceptance ratio. The 1241+1206+1245 combo run is VOID for throughput attribution because greedy output and MTP acceptance changed; isolate RD13 and require draft-acceptance parity before using MTP throughput.

## Performance
Current producer intentionally leaves performance/controls unsatisfied; do not bypass that policy. Once an authorized producer exists, run fixed-work `llama-bench` first, then the applicable order-balanced `tools/lab/native-vs-patched/server-ab-*.json` single-patch arm with alternating order and retained raw pairs. No validation/sign-off is claimed.
