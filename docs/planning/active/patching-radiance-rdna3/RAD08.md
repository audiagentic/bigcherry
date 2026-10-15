---
id: RAD08
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-14T22:16:29.378987+00:00'
breadth: ''
skill: basic
created-by: agent
priority: P2
work: S
---

# KLD tool reports the reference metric set: median, P99, P99.9, top-1, text-only split

## Description

Our KLD runs report a mean only, against the two-XTX run of the same container. The model's published quality is reported as mean, median, P99, P99.9 KL and top-1 agreement, over all tokens and over text+assistant turns. Report the same so layout drift is comparable to quantisation loss.

## Steps

1. Extend the KLD analysis (fn-r9700-kld.sh / its python) with the full metric set and the token split.
2. Re-analyse existing captured logits for the current helper layout.
3. Record results on the Flash-Next model page (RAD07).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Unit test on synthetic logits; re-analysed numbers recorded.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-10-14T22:16:29.378987+00:00 (created-by): Created by agent
