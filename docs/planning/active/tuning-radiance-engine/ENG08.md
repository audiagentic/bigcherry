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

1. Reconcile current native reader/key semantics against RUM migration. The Nov 20 note says the production key omits MTP depth and rejects a duplicate depth-4 record, despite earlier step text assuming a distinct depth-4 calibration. Confirm actual current source behavior and owner before modifying any calibration record; do not treat a policy-OFF result or a mismatched/stale record as ON.
2. If the current native contract can validly represent and activate the required depth-4 policy record, create/use it through the maintained calibration path and verify feature state ON on fresh production-key launches. If it cannot, document the precise blocker and route a narrowly scoped key/reader requirement to its current owner; do not invent a second store or bypass fail-closed admission.
3. Once a matching ON record is active, compare fixed depths 3 and 4 with ABBA, at least two fresh launches per arm, same sampled prompts, 1/6 streams, 8 sessions and 256K; include cold 27K and 44K document prefill/TTFT.
4. Track actual proposed/accepted MTP tokens, ms/step, throughput, wall time, VRAM, heat/admission and exact source/model/plan identity. Apply MEN05 quality gates where comparable and coding-task quality for unmatched cases.
5. Set depth 4 only for flashnext-4gpu if measured benefit exceeds noise without prefill/TTFT, multi-stream or quality regression; otherwise retain depth 3 with an explicit measured no-go.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Current native key/reader semantics and source identity are verified before calibration. Any depth-4 record is accepted only if launch feature state proves it active under matching current key. Policy-ON ABBA includes at least two launches per arm, 1/6 streams, 8 sessions, 256K and prefill/TTFT; MTP acceptance and MEN05 quality are recorded.

## Effort & Risk



## Standards



## Acceptance Criteria

flashnext-4gpu serves at depth 4 with the live policy active and the evidence committed, or the A/B shows no gain with the policy on and the item is closed with that result.

## Notes

2026-11-20 (AEDT): paused at the owner's direction so the GPUs go to the LTX and RUM09 work. Done: the native depth-3 calibration filed a v1 record under the production key c2a1aa0e (the key does not include the MTP depth, so depth 3 and depth 4 share it; a depth-4 calibration is refused as a duplicate), the production store now holds only that v1 record, and the pre-v1 reader is retired (branch retire-post1169). The policy-on A/B of fixed depth 3 against 4 is deferred: point 1 (depth 3) is in the scratch folder, point 2 partial. Policy-off evidence already shows depth 4 at +4.5% (1 stream) and +9.5% (256K) with a tie at 6 streams (docs/evidence/2026-11-20-eng05-mtp-depth). To finish: ABBA 3,4,4,3 with the policy on, 2 launches each, then set the winning depth per config in engines/radiance/configs/flashnext.toml and close.



2026-11-20 clarification (RV4244): dated evidence states production key c2a1aa0e omits MTP depth and store rejects a duplicate depth-4 calibration. This conflicts with the earlier step assumption of a distinct key. Resolve against the current RUM/native reader before the A/B; no policy-OFF measurement or stale record satisfies the policy-ON gate.

## Change Log

- 2026-11-19T21:44:41.041586+00:00 (created-by): Created by agent
- 2026-11-19T21:44:55.241499+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria
- 2026-11-20T01:49:34.579407+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4244
- 2026-11-20T02:33:19.279510+00:00 (updated-by): Updated: section:steps, section:validation, section:notes
