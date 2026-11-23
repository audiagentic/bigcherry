---
id: VTP51
order: 0
plan: run-dense-27b-virtual-tp
state: in_progress
created-at: '2026-11-23T21:15:18.602198+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P1
---

# Weight prefetch placed in the XTX ranks' all-reduce wait windows instead of on the critical path

## Description

The prefetch op is 2.33 ms of rank 0's traced verify pass (3.05 ms on the R9700) and prefetch on versus off was measured at +1.9% (0.42 ms) at one stream and level at four (VTP37). The op runs as its own launch in the step's sequence, so its cost is on the critical path even though its work is only to warm the cache for later layers. In the same pass rank 0's all-reduce is 9.7 ms and the R9700's 14.1 ms, mostly waiting: the XTX ranks sit idle for several ms a step.

**What we will try.** Issue the weight prefetch for the next layers from a second stream (or as a low-priority launch) timed to start when a rank enters an all-reduce wait, so the cache fill uses time the rank would otherwise idle; keep the size per layer as today (derived from cache size and bytes per sequence, not the hand-picked sequence-count constant). The prefetch only reads, so results are unaffected; the gate is timing.

**Expected.** Up to the 2.3 ms rank-0 prefetch cost on the XTX ranks minus whatever hit rate is lost if the window is too short; realistic 0.5-1.5 ms a step at 6-8 streams.

## Steps

1. From the step trace: per layer, the all-reduce wait window length on rank 0 and the prefetch placement; table of windows against prefetch work.
2. Scratch engine-side option (manifest field, off by default) to launch the prefetch on a second stream at the start of the all-reduce wait; byte identity trivially (read-only).
3. ABBA, untraced, 1, 4, 6, 8 streams, three seeds; trace one run to confirm the prefetch overlaps the wait.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

- Temp-0 text and acceptance unchanged.
- ms/step ABBA over the seed spread at 6 and 8 streams.

## Effort & Risk

Medium. Risk: the second stream contends for the bandwidth the all-reduce staging copies need, or the cache lines are evicted before use.

## Standards



## Acceptance Criteria

- Gain above the seed spread at 6 or 8 streams with no regression at 1 and 4, or a recorded NO-GO with the measurement.

## Notes

**2026-11-24 ceiling retracted; on hold.** The 2.6 ms ceiling came from the tape (56 prefetch ops of about 21 us). The engine already issues the weight prefetch beside the collective (prefetch_beside; on a chained launch it waits for the GEMM and then reads beside the collective without holding it), is read-only, and applies only to decode and verify heights; the whole mechanism measured only 0.42 ms at one stream (on against off) and equal at four. So the item as titled is already implemented. What is left is bounded by a prefetch on / off A/B at 6 and 8 streams (three ABBA cycles, queued after the tp2 comparison): more than 1 ms of off-cost at 6 or 8 streams opens a derived-cap item for the delta-net out-projection read (bytes per sequence against the cache budget, replacing the hand-picked four-sequence limit); within the harness floor closes this item as NO-GO.



**2026-11-24 closed NO-GO.** Weight prefetch on against off, three ABBA cycles, 2020.8.5, qwen27b-tp3: off costs +0.08 ms (s.e. 0.29) at 6 streams and +0.31 ms (+1.0%, s.e. 0.11) at 8 streams (evidence: `docs/evidence/2026-11-24-vtp51-weight-prefetch-ab/README.md`). The whole mechanism is worth about 0.3 ms at 8 streams, so neither a second-stream placement nor a derived cap for the delta-net out-projection read (below the 1 ms threshold) is worth building.



**2026-11-24 ACCEPT-WORK, reopened under the no-minimum-gain standard.** The four-sequence limit on the delta-net out-projection read is a hand-picked literal and a defect whatever it is worth; the whole mechanism measured +0.31 ms at 8 streams, so a derived rule can only help or be neutral. Replacement: derive the limit from the recurrent-state bytes per sequence and the pinned bytes against the device's cache size (managed engine patch; the prefetch is read-only, so output is unchanged by construction; the A/B is a regression check). Brief written (worktree gdnout-prefetch-derive). The earlier 'ceiling retracted / NO-GO' text above stands as the measurement; only the decision changed.

## Change Log

- 2026-11-23T21:15:18.602198+00:00 (created-by): Created by agent
- 2026-11-23T21:15:53.292353+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria
- 2026-11-23T22:51:59.751375+00:00 (updated-by): Updated: section:notes
- 2026-11-23T23:17:45.678952+00:00 (updated-by): Updated: section:notes
- 2026-11-23T23:17:50.756017+00:00 (state-transition): State: pending → completed
- 2026-11-23T23:20:41.183102+00:00 (state-transition): State: completed → in_progress
- 2026-11-23T23:21:11.045838+00:00 (updated-by): Updated: section:notes
