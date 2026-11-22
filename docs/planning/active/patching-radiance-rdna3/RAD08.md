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

1. Consume the existing Radiance `--kld-out` JSON and retain the reported mean, median, p99, p99.9, top-1, and per-corpus breakdown; do not rebuild the native KLD producer.
2. Use an explicitly role-labeled chat corpus to compute the requested text+assistant subset, with token/source counts and clear handling of missing labels.
3. Reanalyse the available helper-layout logits or run the minimum attributable capture needed if they are unavailable; compare the full metric set to the named reference/pin with no invented pass.
4. Link the run identity and measured full/reference/split table to RAD07's Flash-Next model page.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Synthetic fixture unit tests for JSON read, quantile/top-1 extraction, role-source grouping and missing-label behavior; one source-linked helper-layout replay or measured run with full overall/text+assistant metrics.

## Effort & Risk



## Standards



## Acceptance Criteria

1. The maintained KLD analysis reads the native JSON record and reports mean, median, p99, p99.9 and top-1, both overall and for explicitly labeled text+assistant tokens.
2. Fixture tests verify role counts, label filtering, expected quantiles and missing-role handling; no silently invented text+assistant split.
3. An identifiable reference and helper-layout real-data reanalysis reports the full metrics with source, pin, layout, token counts, pass/fail or noncomparability.
4. The Flash-Next model page under RAD07 records and links those results; the old MEN11 helper ROWS throughput A/B is not substituted for quality acceptance.

## Notes

2026-10-17: radiance --kld-out JSON already holds mean/median/p90/p95/p99/p99.9/p99.99/max, top-1 and PPL overall ('all') and per corpus source ('by_source'); the summary log line carries the same. Parser + gate extended in feat/rad08-kld-metrics (4919eb61). Remaining: (1) read the JSON instead of the log line; (2) a chat-format KLD corpus whose sources tag turn roles, so 'by_source' yields the text+assistant column of the owner's reference table; (3) record the helper layout's full set on the Flash-Next model page (RAD07).



BACKLOG TRIAGE 2026-11-04 (revalidated against main 6eb31a708): the proposed disposition "SUPERSEDED by #249; keep RAD39 for interpretation" was NOT applied; this item stays open. Why: PR #249 delivered mean, median, p99, p99.9 and top-1 reporting, but RAD08.md:44 still lists JSON input, chat-role-tagged text+assistant split and re-recorded helper metrics as missing. docs/archive/radiance/LAB_HISTORY.md:58-60 independently says the text+assistant column still needs per-token output. Complete role-split KLD and helper-layout reanalysis before retiring RAD08.

CONSOLIDATION 2026-11-22: MEN11's 'KLD through the harness' remaining outcome belongs here, consuming existing Radiance/native KLD producers and role-tagged evidence. RAD07 publishes the results. KLD acceptance remains unproven until the named replay.

## Change Log

- 2026-10-14T22:16:29.378987+00:00 (created-by): Created by agent
- 2026-10-17T00:33:31.379569+00:00 (updated-by): Updated: section:notes

## Ledger-events

- chg_20261017_012848_radiance-quality-checks-report_4522
- 2026-10-17T01:28:52.271660+00:00 (updated-by): Updated: section:ledger-events
- 2026-11-04T04:23:42.444583+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4470
- 2026-11-22T09:19:34.133445+00:00 (updated-by): Updated: section:steps, section:validation, section:acceptance_criteria, section:notes
