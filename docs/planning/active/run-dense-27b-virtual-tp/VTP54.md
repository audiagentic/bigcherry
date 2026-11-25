---
id: VTP54
order: 0
plan: run-dense-27b-virtual-tp
state: pending
created-at: '2026-11-23T21:25:58.957879+00:00'
breadth: ''
skill: intermediate
created-by: agent
work: S
priority: P1
---

# flashnext-4gpu: separate wait from transport in the MoE mailbox at 6-8 streams with the existing stamps (engine and RDNA2 helper)

## Description

In the flashnext-4gpu step trace (2020.8.4, six and eight streams) the engine rank spends 10.7 ms of a 28.1 ms verify pass in the MoE mailbox add, 38% of the pass. The helper logs of the same run show the RDNA2 tail helper at 871 us a request against 548-559 us on the two XTX helpers, with a third of the routes, so tail compute can explain the whole wait (the per-request model gives a 6.8 ms gap at 20 rows and 49 layers). The data cannot separate that from transport: the helper logs are whole-run averages that include prefill rows, and the step tape has no helper timestamps.

The lab mailbox trace (ENG05, `experiment.diagnostics.lanes.trace_level`, on main) already stamps publish, wait bookkeeping, finish and the helper's own local / act / gemm / write times per request, and was run once at one stream (evidence 2026-11-20): protocol cost 2.75 ms of 26 ms, helper GPU 111 us, engine wait 67.5 us. It has not been run at 6-8 streams, nor with the RDNA2 helper split out per helper.

**What we will do.** One traced serve run of flashnext-4gpu on the current release with trace_level 1 at 6 and 8 streams (the existing lab experiment, no new patch), and a per-helper, per-layer table of: helper GPU time, engine wait, publish, wait bookkeeping, finish, for each of the three helpers. Decision it supports: how much of the 10.7 ms is the slowest helper's compute (tail kernels, VTP55) and how much is protocol and transport (publish, bookkeeping, host staging).

## Steps

1. Check the lab-experiment preconditions on the current release (calibration store, heat file in the bundle folder) and that the stamps exist in the release.
2. One run under a short gpu lease: serve flashnext-4gpu with the trace experiment, stream-bench 6 and 8 streams, 512-token prompts, 96 new tokens; keep the helper logs.
3. Per helper and layer: p50 / p95 of helper GPU, wait, publish, bookkeeping, finish; per pass: sum over the 49 layers of the slowest helper against the engine's wait.
4. Record as an evidence note; the traced run is for diagnosis, not headline numbers.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

- All stamps valid (mailbox_stamp_ok) in the rows used; engine-clock stamps compared only with engine-clock stamps.
- The sum of per-layer slowest-helper time versus the traced engine wait reported together.

## Effort & Risk

Small. Risk: the lab experiment needs the live-policy calibration store, which the earlier traced runs did not use.

## Standards



## Acceptance Criteria

- A per-helper table at 6 and 8 streams and a stated split of the mailbox wait into helper compute, protocol and unexplained, each with its basis.

## Notes

**2026-11-24 not workable as written, not a regression.** A serving start is production-only (no lab experiment), so the stamped run cannot be started; the compute side comes from the helper's own kernel timing instead. Left open as an instrumentation question for the product owner.

2026-11-26 next execution roadmap (docs/planning/NEXT_EXECUTION_ROADMAP.md; main 725bd8951). Assumes native YAML serving cutover and initial min-cap qualification accepted; does not claim they are done. Stage 2/3: prior stamped run blocked by production-only start. Use existing authorized lab YAML arm/diagnostic setting after cutover; otherwise minimum existing diagnostic-surface fix to activate ENG05 stamps, not new protocol. Correlate actual same-step engine/helper timings and union of exposed wait, keep cross-clock uncertainty. VTP55 needs this before fusion; ENG01/03 share baseline.

## Change Log

- 2026-11-23T21:25:58.957879+00:00 (created-by): Created by agent
- 2026-11-23T21:26:29.042735+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria
- 2026-11-23T23:21:51.904383+00:00 (updated-by): Updated: section:notes
- 2026-11-25T18:51:55.930820+00:00 (updated-by): Updated: section:notes
