---
id: QFP11
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-03T16:33:36.231209+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P3
work: S
---

# Retire per-AllReduce graph-capture work; consolidate verify latency onto measured rank-skew owners

## Description

Timed synctrace + rocprof originally attributed ~67 us of apparent dead time to each MTP verify AllReduce boundary and proposed putting AllReduce inside one captured graph. The later boundary decomposition disproved that as the primary lever: once all ranks arrive, pre/post boundary overhead plus cpu-root round trip is only ~25 us, while arrival skew is ~78-82 us median and can reach ~800 us p90 at 80K. The R9700 is the last rank for 67-75% of boundaries. With ~30 AllReduce points/token, measured skew is ~2.4 ms/token while graph split/launch overhead is only ~0.4 ms/token.

QFP11 is therefore a consolidation item, not a new scheduler implementation. Do not add an AllReduce-in-graph path unless later evidence shows >=1 ms/token recoverable boundary overhead after QFP07/QFP09 balancing. The active optimisation belongs to QFP09 (per-class rank lateness / expert balance) and QFP07 (depth-dependent attention placement).

## Steps

1. Preserve `tools/lab/flash-next/ar-boundary.py` as the regression/attribution tool and report arrival skew separately from boundary overhead.
2. Feed per-op-class late-rank attribution into QFP09; feed attention-only late-rank attribution into QFP07. Do not create a second load-balancer or graph scheduler here.
3. Re-run the boundary census after QFP07/QFP09 changes. Only reopen graph-capture work if post-balance boundary overhead is >=1.0 ms/token or >=3% of decode wall on two representative depths.
4. If reopened, first test fewer scheduler splits / launch coalescing using existing graph-cache machinery. Device-side AR-in-graph is the last option because it increases replay/keying/lifetime complexity and overlaps RNX11/PRBE66/67 ownership.
5. Keep graph-memory accounting with QFP06; QFP11 must not duplicate graph-cache policy.

## Detailed Solution & Technical Design

Treat each AllReduce as three measured intervals: `arrival_skew = max(last_pre_AR_kernel_end) - min(last_pre_AR_kernel_end)`, `collective = AR_end - max(last_pre_AR_kernel_end)`, and `resume = first_post_AR_kernel_start - AR_end`. Optimisation ownership follows the interval: arrival skew -> producer/load placement (QFP07/QFP09); collective -> QFP01/1291 provider; resume/split overhead -> QFP11 only if material.

This decomposition prevents a scheduler optimisation from being credited for idle time it cannot remove. A single captured graph cannot make a late producer rank arrive earlier; at the current measurements its optimistic bound is ~0.4 ms/token, below the programme's normal ~3% promotion threshold.

Fresh upstream scan (2026-10-05) reinforces the fusion-before-scheduler rule. llama.cpp PR #29948 fuses dense FFN gate/up MMQ and GLU at the operator/kernel boundary and reports +4.6-10.7% prompt throughput on DGX Spark and +1.9-4.7% on RTX 5090 depending on model/quant. The useful mechanism is reducing materialisation/launch boundaries where producer/consumer semantics permit it; it does not apply directly to Flash-Next routed `MUL_MAT_ID` expert execution and must not be copied into QFP11. If BigCherry pursues an MoE equivalent, it belongs with QFP17/PKC after profiling routed-MMQ critical-path share and must reuse existing fusion/dispatch infrastructure.

Current upstream release scan is b11392 (2026-10-04, commit 46847e6). Rebase/qualification work should use b11392 or newer rather than the earlier b11390 baseline.

## Code Samples & Guidance

No production code is justified at current evidence. Keep the attribution boundary explicit in tooling:

```text
late_rank_us = max(pre_ar_end[gpu]) - min(pre_ar_end[gpu])
collective_us = ar_end - max(pre_ar_end[gpu])
resume_us = min(post_ar_start[gpu]) - ar_end
```

Do not combine these into a single `gap` metric when deciding whether to alter graph capture.

## Files

- `tools/lab/flash-next/ar-boundary.py`: retain/extend attribution output only.
- `docs/planning/active/patching-qwen-flash-next/QFP09.md`: owner for generic per-class rank lateness and expert split.
- `docs/planning/active/patching-qwen-flash-next/QFP07.md`: owner for attention placement at depth.
- QFP06 / RNX11 / PRBE66/67: graph-memory, replay and keying owners; no duplicate implementation here.

## Validation

At 10K/80K (and 160K where stable), record median/p90 arrival skew, collective and resume intervals separately, late-rank frequency by op class, decode ms/token and cached-graph count. Greedy output must remain identical for any later implementation experiment.

## Effort & Risk

Current action is low-risk consolidation. Reopening AR-in-graph is high complexity relative to its measured <=~0.4 ms/token optimistic bound and risks duplicating graph replay/keying work.

## Standards

Optimise measured critical-path ownership, not aggregate idle time. Reuse existing scheduler, graph-cache, collective and architecture-dispatch facilities. No new generic scheduler or communicator from this item.

## Acceptance Criteria

QFP11 remains implementation-parked while split/collective/resume overhead is <1.0 ms/token and <3% decode wall after load balancing. Reopen only if both a representative short-context and long-context lane cross one of those gates. Any reopened change must improve end-to-end decode >=3% on one lane, regress no representative lane >1%, preserve greedy identity, and reduce the specifically attributed boundary interval rather than merely moving arrival skew.

## Notes

Evidence runs: flashnext-v2-synctrace-timed (sync-tracer.c records blocked time), flashnext-v2-profile (rank-census.py), flashnext-v2-1304-memlog. Related: QFP01 (1291 cpu-root AR), QFP06 (graph memory), QFP09 (rank skew), QFP07 (attention placement), RNX11 (AR graph-replay hardening), PRBE66/67 (graph keying).

2026-10-04 boundary decomposition (`tools/lab/flash-next/ar-boundary.py`, d8192/d65536, ~30 ARs/token): gap_before ~9 us, gap_after ~3 us, cpu-root round trip after all ranks arrive ~13 us (p90 14). Arrival skew dominates: median 78 us at ~10K (p90 110), 82 us at ~80K (p90 800); R9700 arrives last in 75%/67% of ARs. Skew x ~30 ~=2.4 ms/token versus ~0.4 ms/token split/launch bound.

2026-10-05 consolidation: original QFP11 implementation path is parked. QFP07/QFP09 own the measured bottleneck. Upstream #29948 is recorded as operator-fusion mechanism evidence only; its dense FFN MMQ+GLU path is not equivalent to Flash-Next routed MoE and should not be transplanted here. Upstream release baseline advanced to b11392.

## Change Log

- 2026-10-03T16:33:36.231209+00:00 (created-by): Created by agent
- 2026-10-03T16:35:18.028294+00:00 (updated-by): Updated: section:notes
- 2026-10-03T16:35:21.329774+00:00 (updated-by): Updated: priority='P3'
- 2026-10-05: Consolidated QFP11 onto QFP07/QFP09 after boundary decomposition disproved graph-capture as the primary lever; added explicit reopen gates and upstream #29948 mechanism triage.
