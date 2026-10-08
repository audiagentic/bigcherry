---
id: BCOP68
order: 68
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-08T13:06:40+00:00'
created-by: agent
priority: P2
work: S
---

# PEF07: gate retained PM4 on actual public-HIP-graph submission cost

## Discovery / disposition

Pinned b11474 already captures/replays HIP graphs; QFP13's ~1,000-1,300 device kernels/token are not a host-launch count. A changed property resets graph warmup even when kernel topology is unchanged. Upstream unmerged #29768 proposes a narrower recapture path, but its +1.14% CI95 [+0.44%, +1.84%] is on RTX 5090 only. #29796 explicitly excludes HIP. Hipfire's PM4 speedups are external, not BigCherry qualification.

## Authoritative ownership / existing work

PEF07 owns only the retained-submission transport decision. QFP13 owns measured launch ranking and E2E acceptance; HI14/1231 owns capture-lifecycle evidence; native HIP graph owns capture/replay; validated 1302 owns OOM eviction. Rejected 1233 stable-key and 1304 graph-cap approaches are not revived. Recent QFP37, QFP43/MTP, PA47 and 1340 Meta work is protected by the 12-hour rule; no neighbouring plans, code or queued lanes were changed.

## Bounded gate / terminal outcome

On an isolated 1x gfx1100 and 1x gfx1201 dense-27B Q4_K_M plain-decode control (no MTP/Meta/collectives), first count *actual* eager/graph host submissions, capture/instantiate/reset, per-key hit rate, CPU submit/sync time and GPU work/token at 8K/80K; existing once-only lifecycle markers cannot count hit rate. `graphs-ab.sh` is a multi-GPU/MTP default and must not be used unchanged for the initial one-device discriminator. If replay already >=95% and host submit/sync <5% E2E, close PM4. If resets dominate, isolate upstream #29768 before considering PM4. If incompatibility dominates, return to its existing owner. Only a >5% attributable host-submission bottleneck after a qualified public graph justifies a default-off retained prototype. Promotion: >=4 sessions x >=10 paired ABBA rounds, CI95-low >=3% E2E TG, <=1% fallback/control regression, exact work/logits/KV/recurrent parity and memory safety; transport-only effect separated from wait removal. No hardware test/build was run in this documentation audit.

## References

PEF07; QFP13; HI14/1231; 1302; QFP06/1304; RD73/1233; https://github.com/ggml-org/llama.cpp/pull/29768 ; https://github.com/ggml-org/llama.cpp/pull/29796 ; https://github.com/warpfront/hipfire/blob/a89ed0a8e9d8dc7a22d4e6dcbacd57bf2abfad74/crates/redline-dispatch/HIPFIRE-GRAFT.md
