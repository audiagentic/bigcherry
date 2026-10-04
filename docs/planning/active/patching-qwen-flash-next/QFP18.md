---
id: QFP18
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-04T07:54:10.353678+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: M
---

# Lightweight evidence-reuse promotion to validated for the Flash-Next stack

## Description

Owner direction 2026-10-04: the path to 'validated' must be quick and reuse evidence we already produce; no heavyweight per-patch 4-session contract campaigns or bespoke producers for patches that already have strong evidence. Reserve full contracts for patches meant for broad reuse or with weak/ambiguous evidence.

## Steps

1. Define a 'profile-evidence' qualification tier (tooling + docs: docs/reference/testing/PATCH_VALIDATION.md, tools/bigcherry/patch/validation_policy.py / evidence.py) accepting, on the current pin: (a) patch-lint + focused mechanics tests pass; (b) activation evidence - BIGCHERRY_PATCH_HIT marker or a census/trace showing the mechanism (kernel/launch counts, Q8_1 hit trace, timing split); (c) an adoption ABBA (abba-depths.sh or quick-ab-depth) with complete separation or a stated neutral/enabler rationale, and greedy identity across arms; (d) the run logs referenced by path in patches/<id>/evidence/validation.json (append-only).
2. Add a small converter: ABBA run dir -> evidence JSON (per-arm t/s, ms/step, acceptance, greedy hashes, build id, pin).
3. After the pin bump: one v5 re-screen on the new pin supplies the profile evidence; attach per-patch activation evidence; promote 1302, 1303, 1307-1313, 1326 (+ 1291/1292/1294/1297) via the lifecycle skill in one pass.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

patch-verify-evidence passes on the new pin; lifecycle promotion recorded with evidence.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Owner agreed 2026-10-04: qualify after the bump. Current states: evaluated (1302, 1303, 1307-1313, 1326, ...), rejected 1304; 1237/1265/1253 already validated but only now entering the Flash-Next profile.

Correction: 1237/1265/1253 are already in every Flash-Next build via the validated-enhancements patch-set; v5 = v4 + 1326.

Replaces the earlier per-patch contract plan (estimated ~3 dev-days + ~2 GPU-days - rejected as too heavy). Existing evidence: v3/v4/v5 adoption ABBAs (flashnext-v2-fusion-ab-3, flashnext-v4-abba, flashnext-v5c-abba), 1312/1313/1326 screens with census/trace/timing evidence, offline tests for every patch. Correction: 1237/1265/1253 are already validated and in every build via validated-enhancements.

## Change Log

- 2026-10-04T07:54:10.353678+00:00 (created-by): Created by agent
- 2026-10-04T07:55:43.580424+00:00 (updated-by): Updated: section:notes
- 2026-10-04T09:25:05.297276+00:00 (updated-by): Updated: section:title, section:description, section:steps, section:notes
