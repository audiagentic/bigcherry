# Testing — GP11 MMVQ fusion ncols gate

Target candidate path: Qwen3.8-27B-Q8_0, dual 7900 XTX/gfx1100, `-sm tensor`, MTP `n_max=4`. Existing negative evidence remains a disposition, not validation/sign-off.

## Activation
1245 has no trace marker. Do not use source guard presence as activation proof. If this candidate is re-examined, first add a once-per-process `BIGCHERRY_PATCH_TRACE` marker in the successful host-side widened fusion-selection branch, before dispatch, exactly: `BIGCHERRY_PATCH_HIT patch=1245_gp11 path=mmvq_fusion_q8_0_ncols6`. Require subject hit/control miss.

## Correctness
Before any performance claim, run ncols=6 Q8_0 backend-reference coverage and the target MTP server lane. Compare pre-sampling full-vocabulary MTP logprobs with max absolute difference <= `5e-4` where validation requires it; report greedy-token divergence explicitly.

## Work equivalence
Record drafted/accepted counts and acceptance ratio. `runs/combo-ab1/RESULT.md` is VOID for throughput attribution: the 1241+1206+1245 combo changed greedy output and acceptance (0.95580 vs 0.90101). Isolate 1245 and require MTP draft-acceptance parity.

## Performance
Only after correctness/work equivalence, use fixed-work `llama-bench` plus the applicable order-balanced `tools/lab/native-vs-patched/server-ab-*.json` single-patch arm. Retain the historical negative result as context; do not treat it as contract-qualified evidence.
