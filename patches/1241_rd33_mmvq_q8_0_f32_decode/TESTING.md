# Testing — RD33 Q8_0 F32 decode

Target candidate path: Qwen3.8-27B-Q8_0, dual 7900 XTX/gfx1100, `-sm tensor`, MTP `n_max=4`.

## Activation
Require subject hit/control miss for `BIGCHERRY_PATCH_HIT patch=1241_rd33 path=q8_0_f32_decode`. Exercise the relevant `ncols_dst` range, especially the MTP verify width.

## Correctness
Run the declared backend-reference check. For server/MTP qualification, add a pre-sampling full-vocabulary logprob comparison and require max absolute difference <= `5e-4` where the contract/validation demands it; also report first divergent token/logprob if greedy output differs.

## Work equivalence
Record drafted/accepted counts and acceptance ratio. `runs/combo-ab1/RESULT.md` is VOID for MTP-lane throughput attribution: the 1241+1206+1245 subject changed greedy output and acceptance (0.95580 vs 0.90101). Therefore isolate 1241 before attributing any MTP throughput.

## Performance
First run a fixed-work `llama-bench` comparison so timing does not depend on speculative acceptance. Then run the applicable order-balanced `tools/lab/native-vs-patched/server-ab-*.json` single-patch arm. MTP throughput is attributable only after correctness/logprob and work-equivalence gates pass. No validation or sign-off is claimed.
