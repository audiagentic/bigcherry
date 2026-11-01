---
id: RAD39
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-11-01T07:17:29.719220+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: S
---

# Helper-layout KLD gate judges against the measured noise band, not one control arm

## Description

The helper-layout quality gate (ab-benchmark quality, e.g. /work/development/tmp/bigcherry/rad23/kld-mixed.json) compares each arm with one control arm (h2) using fixed increase limits. On Flash-Next any change to arithmetic order or rounding moves KLD inside a band: the model page records re-chunking the stock path alone at 0.058-0.062 mean, top-1 88.4%, and sets the bar for helper layouts at 0.058-0.067. 2026-11-01 evidence: h2 is bit-identical (0.05518 / p99.9 1.6424 / top-1 89.745%) on every bf16-partial build, but moves to 0.0636 / 2.095 / 88.54% with unrounded f32 partials and 0.0651 / 2.019 / 88.70% with fp16 row-exponent partials (rad23-kld-f32-11010631, rad23-kld-combn-11010650, rad23-kld-f16-11010708). The 'fails' of the mixed-heat arms (p99.9 2.215; top-1 -1.18) were inside this noise. The gate therefore rewards a lucky rounding pattern in the control and fails arms on noise.

## Steps

1. Measure the band: the same corpus over N (>=6) arithmetic-only perturbations of one configuration (chunk sizes, helper result format, ordering), record mean/median/p99/p99.9/top-1/PPL spread.
2. Express helper-layout limits as absolute bounds from that band (and, for p99.9/top-1, a bootstrap CI over positions), not as increases over one arm.
3. Re-score the RAD23 mixed-heat and f32/f16 arms against the band; record in the Flash-Next model page.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

The h2 control on bf16, f32 and fp16 builds all fall inside the band; a deliberately broken arm (poison=1 on one helper) fails it.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Found while bisecting the f32-partials h2 shift (RAD23).

## Change Log

- 2026-11-01T07:17:29.719220+00:00 (created-by): Created by agent

## More evidence (2026-11-01)

- Bisect: the combine-n build reproduces h2 exactly (0.05518 / 1.6424 / 89.745%, `rad23-kld-combn-11010650`), so the f32-partials build alone moved it; fp16 row partials then gave 0.0651 / 2.019 / 88.70% (`rad23-kld-f16-11010708`). All three are one model under different rounding.
- The gfx1030 dot2 kernel (#486) moved the 14 GiB tail arm from 0.0587 / 2.22 / 89.39% to 0.0660 / 2.28 / 88.93%, failing p99.9 against h2 by the same margin the f32 kernel did (+0.63 against +0.57).
- At 2048-token steps (so the helper-row combine engages, `rad23-kld-comb-11011010`) the h2 control itself is 0.0699 / 2.545 / 88.92%, against 0.0552 at 64-token steps: chunking alone moves the control by more than the gate's mean limit.

