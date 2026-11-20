---
id: ENG02
order: 0
plan: tuning-radiance-engine
state: pending
created-at: '2026-11-10T13:11:37.675845+00:00'
breadth: ''
skill: basic
created-by: agent
priority: P1
work: S
---

# Current-build MTP depth 0–5 policy-ON sweep, excluding ENG08's depth-4 A/B

## Description

Qualify MTP depths on the current flashnext-4gpu Radiance build with live placement policy confirmed ON. The Nov 11 depth 0..5 result below was policy OFF and remains historical evidence only; it is not policy-ON acceptance and cannot be combined with ENG08's later policy-OFF depth-4 result as though they were one sweep. ENG08 owns the calibrated policy-ON depth-3 versus depth-4 production decision. ENG02 retains the distinct multi-depth policy-ON decision for 0,1,2,3 and 5; coordinate depth 3 baseline with ENG08 and do not duplicate depth-4 calibration/A-B.

## Steps

1. Confirm current source/build/model/manifest and feature state. Record exact policy key and require live placement ON for every accepted arm; keep policy-OFF history separately labelled.
2. Under a later exclusive GPU lease, test depths 0,1,2,3 and 5 across the established prompt set and one-/six-stream workloads. ENG08 alone owns the depth-3 vs depth-4 calibrated ABBA production decision; share its matching depth-3 reference rather than starting duplicate depth-4 arms.
3. Use fresh launches per distinct depth, verify MTP replay for depths above 1 and record ms/step, target tokens/step, proposed/accepted distribution, throughput, prefill/TTFT, VRAM, heat and admission state.
4. Report per-depth decision and whether depth 3 remains the supported default alongside ENG08's separate depth-4 decision. Do not change global defaults based on policy-OFF evidence.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Tables for policy-ON depths 0,1,2,3,5 across the established prompts with source/build, manifest, feature state and plan key. ENG08 supplies the distinct depth-4 ABBA/calibration result. MTP replay passes for each depth above 1; accepted/proposed work and prefill/TTFT are recorded.

## Effort & Risk



## Standards



## Acceptance Criteria

A current policy-ON result supports the targeted depth choice across multiple prompts and one-/six-stream workloads, with MTP activation and acceptance distribution verified. Depth 4 is decided only through ENG08's separate calibrated A/B. Historical policy-OFF tables remain clearly labelled and cannot satisfy this item.

## Notes

Memory: MTP acceptance over several prompts. Source: gateway prior-work audit 2026-11-10.



2026-11-20 reconciliation: the existing Nov 11 sweep was live policy OFF and remains historical. ENG08 owns the calibrated policy-ON depth-3 vs depth-4 A/B; this item retains its unique policy-ON multi-depth evaluation at 0,1,2,3,5. No OFF result is reclassified as ON.

## Result 2026-11-11 (flashnext-4gpu, live policy OFF for every arm, 6 one-stream and 3 six-stream seeds)

| Depth | 1-stream tok/s | acceptance | ms/step | 6-stream tok/s | acceptance |
|---|---|---|---|---|---|
| 0 | 51 | - | 19.4 | 225 | - |
| 1 | 87 | 0.862 | 21.5 | 283 | 0.843 |
| 2 | 110 | 0.774 | 23.1 | 322 | 0.749 |
| 3 | 123 | 0.694 | 25.2 | 339 | 0.643 |
| 4 | 129 | 0.606 | 26.5 | 343 | 0.604 |
| 5 | 112 | 0.456 | 29.2 | 291 | 0.504 |

Depth 5 is clearly worse. Depth 4 is +5% over depth 3 on one stream and +1% on six streams (inside run-to-run noise at six streams). The depth is part of the plan and calibration key, so moving the default from 3 to 4 invalidates the calibrated live-placement policy and needs a fresh calibration; the gain does not pay for that. **Decision: keep the default at 3.** Re-measure depth 4 only together with the next recalibration. The same depth-4 arm on the patched kernel (7.0.13-gfxoff) gave 129/339 tok/s, so the kernel change does not move decode.

## Change Log

- 2026-11-10T13:11:37.675845+00:00 (created-by): Created by agent

## Reviews

- RV4240
- 2026-11-20T02:29:55.302868+00:00 (updated-by): Updated: section:title, section:description, section:steps, section:validation, section:acceptance_criteria, section:notes
