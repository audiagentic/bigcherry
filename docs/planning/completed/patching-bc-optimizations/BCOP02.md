---
id: BCOP02
order: 2
plan: patching-bc-optimizations
state: superseded
created-at: '2026-10-05T04:31:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Qualify Flash-Next IQ expert service path and MMQ safety

## Description

Follow-through for QFP10 audits that changed the optimization target from isolated vec-dot throughput to total expert-service latency and identified upstream #29941/#29953 MMQ allocation/launch consistency as correctness prerequisites.

## Steps

1. Verify the current llama.cpp pin contains or supersedes #29941 and #29953; if not, qualify/port the minimum correctness fix before performance work.
2. Test MMQ J-boundaries 7/8/9, 15/16/17, 31/32/33, 63/64/65 and 127/128/129 for dense and MUL_MAT_ID on gfx1100/gfx1201; allocation-selected and launch-selected J must agree.
3. Measure complete IQ3_S/IQ4_NL expert service: F32->Q8_1 conversion, temporary traffic, MMVQ/MMQ, routing/scatter, launches/waits, occupancy and spills for ncols 1/2/4/8.
4. Compare stock path, VDR1 extension and direct-F32/MMVDQ-style path. Stop MMVDQ work if activation conversion is <10% of service wall.
5. Promote only with >=10% expert-service reduction and >=3% end-to-end decode/MTP improvement with correctness intact.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria

- MMQ allocation/launch geometry is proven memory-safe on both AMD architectures.
- One measured expert-service breakdown determines the winning path.
- No duplicate tile estimator, dispatch table or padding formula is added.
- QFP10 records promote/reject evidence and implementation owner.

## Notes

2026-10-05 review: superseded by owner item QFP10, which already carries the #29941/#29953 prerequisite, the MMQ J-boundary matrix and the MMVDQ 10% stop gate. Not implemented; work stays open under QFP10. Neither PR is in pin 050439614.

## Related

QFP10, PKC05/1273, HIP autotune; llama.cpp #29941, #29953.

## Change Log

- 2026-10-05T04:36:46.056072+00:00 (updated-by): Updated: section:notes
- 2026-10-05T04:37:31.714015+00:00 (state-transition): State: pending → superseded
