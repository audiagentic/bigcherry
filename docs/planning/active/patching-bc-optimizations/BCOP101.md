---
id: BCOP101
order: 101
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-10T00:13:34+00:00'
created-by: agent
priority: P2
work: S
---

# PNRO16: retire duplicate offline expert-placement compiler

## Discovery / disposition

PNRO16's eight-module GGUF inventory/placement/replay proposal is superseded by existing `expert-place.py --plan-only`, `routing-balance.py`, validated 1281/1283 expert split, MET08 placement evidence and MET11's load-time owner. The original unresolved GGUFReader import and model-evidence choices are already resolved by those paths. Retire PNRO16 (state done); do not create another tool, cache, scheduler, solver, registry or patch.

## Ownership / independent activity

MET11 owns load-time placement and immutable old/new expert-ID mapping; MET08 owns per-device routing/performance evidence; MET09 numerical parity; MET07/1337/1338 profile/cache; MET04/1281/1283 range execution; RPL01 topology model. PNRO16 last independently changed 2026-09-24; MET11/expert-place last independently changed 2026-10-06. The selected capability and immediate implementation path had no independent commits, submissions or queued work within the preceding 12 hours. Recent Radiance MXFP4, 1355/1356, 1334/1347, RR04 and Meta split-cache work was excluded and untouched. Recent BCOP91-100 addressed different capabilities.

## Bounded remaining action / terminal gate

MET11 should reuse existing `--plan-only` (not add `--dump-permutation`), reject negative caps/NaN traffic, bind profile/model/shards, validate packed-block bijection and per-device bytes, then use real routing traces to decide whether load-time placement warrants a prototype. Existing proposed ID `1343_moe_expert_placement` collides with reserved `1343_mtp_nextn_rereserve`; choose an unused ID only after a positive gate. Host-model cases demonstrated invalid input admission and rank-only 181/161 versus synthetic weighted 171/171 routing loads; no GPU/build/test-suite run or new throughput result. If no >=3% E2E plausible gain with safe fit, close MET11 without implementation. PNRO16 is terminal unless a distinct measured requirement appears.

## References

PNRO16; MET04/MET07/MET08/MET09/MET11; RPL01; tools/lab/flash-next/expert-place.py; tools/lab/strata/routing-balance.py; 1281/1283/1338/1343; upstream llama.cpp #29887/#29963; vLLM EPLB balanced_packing; SGLang EPLB expert_distribution.
