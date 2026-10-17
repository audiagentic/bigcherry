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

2026-10-17: radiance --kld-out JSON already holds mean/median/p90/p95/p99/p99.9/p99.99/max, top-1 and PPL overall ('all') and per corpus source ('by_source'); the summary log line carries the same. Parser + gate extended in feat/rad08-kld-metrics (4919eb61). Remaining: (1) read the JSON instead of the log line; (2) a chat-format KLD corpus whose sources tag turn roles, so 'by_source' yields the text+assistant column of the owner's reference table; (3) record the helper layout's full set on the Flash-Next model page (RAD07).

## Change Log

- 2026-10-14T22:16:29.378987+00:00 (created-by): Created by agent
- 2026-10-17T00:33:31.379569+00:00 (updated-by): Updated: section:notes
