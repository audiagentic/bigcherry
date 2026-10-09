# Testing — RD13 MUL_MAT + RESHAPE + ADD fusion

Current contract-positive path: `tierA-qwen4b-q6k` (Qwen3.5-4B GDN hybrid), one physical gfx1100/gfx1201/gfx1030 device per run. The older Qwen3.8-27B dual-XTX/MTP lane is an optional non-contract control, not the required promotion path.

## Activation
Require subject hit/control miss for `BIGCHERRY_PATCH_HIT patch=1206_rd13 path=mul_mat_add_view_fusion_(?:f|q)` on a graph/model that actually contains the target pattern.

## Correctness
The bound validation requires backend-reference correctness. For the target MTP lane, compare pre-sampling full-vocabulary logprobs control vs subject with max absolute difference <= `5e-4` where required, and require greedy-token parity.

## Work equivalence
Record drafted/accepted counts and acceptance ratio. The 1241+1206+1245 combo run is VOID for throughput attribution because greedy output and MTP acceptance changed; isolate RD13 and require draft-acceptance parity before using MTP throughput.

## Performance
The current patch-local producer (implemented 2026-09-21) already runs the standard scaffold's paired positive/control tg128 lanes and returns typed promotion effects. `performance_benchmark_cli = "forbid"` prevents a duplicate generic lane; it does not disable producer-owned benchmarking. Rebaseline at b11474 using four independent same-architecture sessions with >=10 paired rounds each, CI95-low >0% positive and <=1% control regression. Keep gfx1100/gfx1201/gfx1030 isolated and serial; older b11126 results are historical. Capture subject-only marker, exact full-vocab reference and graph replay before any promotion. No new hardware qualification is claimed.
