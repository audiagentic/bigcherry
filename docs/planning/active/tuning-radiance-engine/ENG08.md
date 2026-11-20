---
id: ENG08
order: 0
plan: tuning-radiance-engine
state: pending
created-at: '2026-11-19T21:44:41.041586+00:00'
breadth: ''
skill: ''
created-by: agent
work: M
priority: P2
---

# Flash-Next MTP depth 4: calibrate the policy record, A/B policy on, then set the depth

## Description

docs/evidence/2026-11-20-eng05-mtp-depth: on flashnext-4gpu a fixed MTP depth 4 beats the stock 3 by 4.5% at one stream (110.6 vs 105.8 tok/s) and 9.5% at 256K (46.5 vs 42.4), and ties at 6 streams, policy off. Adaptive (--draft-window adaptive, already a resolver key tuning.draft_window) tracks fixed 4 at ceiling 4 and loses at ceiling 5, so it is closed. The default cannot change yet: the live-policy calibration key hashes the depth, so a depth-4 launch runs with the production policy inactive until a depth-4 record exists.

## Steps

1. Calibrate a depth-4 live-policy record on flashnext-4gpu (bc-serve --calibration-run with engine.num_speculative_tokens=4, then calibration-ingest into the production store), under the gpu lease.
2. Policy-on A/B of fixed 3 against fixed 4 with tools/lab/radiance/flashnext-parity.sh measure (451 s warmup, at least 2 launches each, ABBA): decode 1/6 streams, sess8, 256K, cold 27K prefill and 44K document prefill (prefill and TTFT must not regress).
3. If depth 4 holds its gain with the policy on and prefill/TTFT are unchanged, set it for flashnext-4gpu in engines/radiance/configs/flashnext.toml (a config setting, not a global default) and update the parity baseline; otherwise record the result and close.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Policy-on launches (at least 2 per arm, ABBA) show the decode tok/s gain at 1 stream and 256K, no loss at 6 streams, and cold prefill and document-prefill within the baseline allowance; the depth-4 record is the one the production start activates (policy state shown in the launch's feature state).

## Effort & Risk



## Standards



## Acceptance Criteria

flashnext-4gpu serves at depth 4 with the live policy active and the evidence committed, or the A/B shows no gain with the policy on and the item is closed with that result.

## Notes

2026-11-20 (AEDT): paused at the owner's direction so the GPUs go to the LTX and RUM09 work. Done: the native depth-3 calibration filed a v1 record under the production key c2a1aa0e (the key does not include the MTP depth, so depth 3 and depth 4 share it; a depth-4 calibration is refused as a duplicate), the production store now holds only that v1 record, and the pre-v1 reader is retired (branch retire-post1169). The policy-on A/B of fixed depth 3 against 4 is deferred: point 1 (depth 3) is in the scratch folder, point 2 partial. Policy-off evidence already shows depth 4 at +4.5% (1 stream) and +9.5% (256K) with a tie at 6 streams (docs/evidence/2026-11-20-eng05-mtp-depth). To finish: ABBA 3,4,4,3 with the policy on, 2 launches each, then set the winning depth per config in engines/radiance/configs/flashnext.toml and close.


## Change Log

- 2026-11-19T21:44:41.041586+00:00 (created-by): Created by agent
- 2026-11-19T21:44:55.241499+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria
- 2026-11-20T01:49:34.579407+00:00 (updated-by): Updated: section:notes
