---
id: RHP08
order: 8
plan: patching-radiance-heat-placement
state: pending
created-at: '2026-10-31T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Runtime tail-heat collector

## Description

Establish the actual route frequency and coverage of the experts the 6900 XT owns, including the MTP layer, from the RHP02 feed. The static profile gave the tail 0.68% of routes; natural text gave 1.7%, random-word prompts 14%.

## Steps

1. Write the protocol, experiment or collector against the code as it is (radiance 1.14.0, remote-MoE protocol v19).
2. Land card-free tests first where the item has any logic; measure only with the gpu lease.

## Detailed Solution & Technical Design

## Code Samples & Guidance

## Files

## Validation

Observed route counts match the engine histograms; the static-profile bias is explained; no second production GPU collector.

## Effort & Risk

Depends on: RHP02. Added by the 2026-10-31 GPT review of this plan.

## First live measurement, 2026-10-31

`R3_MOE_WORKER_HISTOGRAM_PERIOD_S` (RHP02, this change) rewrites the engine's cumulative route histogram `<path>.rank<r>` every N seconds while the engine runs (rename, private non-blocking stream); the file is `uint32[128][512]`, the same format `R3_MOE_WORKER_HEAT` and the launcher's `--heat` read, so a live snapshot can seed the next start. Run: Flash-Next, `flashnext-2xtx-rdna2-tail` with a 4 GiB tail, a documentation-text context sweep (8K, 32K) and three chat prompts (prose, code, French); 25.2 M routes, layer 48 (MTP) present.

Against the static profile (`perf51/heat.rank0`, raw counts), by the placement it would give (XTX 18,640 experts; the tail the 1,740 coldest; the rest host-resident):

| Set | Static share | Live share |
|---|---|---|
| XTX (2 x 22 GiB) | 99.34% | 95.83% |
| host-resident leftover | 0.66% | 3.48% |
| 6900 XT tail (coldest 1,740) | 0.00% | 0.68% |

So the profile under-predicts the leftover traffic five-fold for this mix, and the tail's coldest experts carry 0.68% of routes against 1.68% for the hottest leftovers. Ranking by the live counts would put 98.3% of routes on the XTXs. The hottest 1,740 leftovers are not a usable tail: their worst layer takes 11% of a layer's routes (about 1,800 per 2048-token chunk against the 6900 XT's ~110), which is the prefill stall the head-share layout showed. MTP layer experts carry 0.12% of live routes and are not in the tail set.

Whether the recovered 2.5% of routes shows as speed is open: the tail runs moved host-to-R9700 traffic from 51 to 32 GiB with no change in speed at short context, so the test belongs to RHP10 (long context, where the engine's expert pool shrinks).

