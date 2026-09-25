---
id: PRBE112
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-25T23:16:25.986252+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# Diagnose 1215 (RD39-42) regression and decide rework vs retire

## Description

1215 rejected 2026-09-26: under GGML_CUDA_GRAPH_OPT=1 the MoE target lane regressed (-2.9..-3.3% gfx1100, -8.1% gfx1201) AND the dense control lane regressed too (-1.6..-2.2% / -6.8%). A MoE-overlap payload cannot explain a dense regression, so one of the non-MoE parts is the likely cause: RD39 (matmul honors the assigned stream, so upstream graph-opt branches such as QKV now truly run concurrently), RD40 (per-(device,stream) BLAS handles) or RD41 (dedicated branch scratch). Production does not enable graph-opt, so 1215 is inert there as shipped.

## Steps

1. Measure graph-opt itself: baseline GRAPH_OPT=0 vs =1 on the dense and MoE models (gfx1100, gfx1201) to learn whether upstream concurrency already costs on RDNA.
2. rocprofv3 trace one dense decode step on 1215 vs control under GRAPH_OPT=1: stream overlap, extra syncs, BLAS handle creation, scratch allocation.
3. If RD39 concurrency is the cost: variant that keeps RD40/RD41 correctness fixes but only overlaps the MoE shared expert (RD42) and leaves other branches on the main stream.
4. Re-run the RD39-42 contract on the variant; reject for good if still regressing.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

4 sessions per arch under the frozen improvement_no_regression_v1 policy; dense control lane within 1%.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Owner asked 2026-09-26 whether 1215 can be fixed. 1216 (join-fusion guard) was separated from 1215 the same day: it guards upstream's own graph-opt concurrent regions.

## Change Log

- 2026-09-25T23:16:25.986252+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260925_232219_first-patch-promoted-to-the-pr_2756
- 2026-09-25T23:22:33.660056+00:00 (updated-by): Updated: section:ledger-events
