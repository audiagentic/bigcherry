---
id: QFP16
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-04T05:13:42.699438+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: M
---

# Target verify submit host time 5-7 ms per round (15-17% of an MTP step)

## Description

FMTP01 Gate 0 (1317) measured llama_process(ctx_tgt) host time at 5.06 ms median (mean 6.7) per verify round at ~10K and 5.83 ms (mean 7.1) at ~80K, 15-17% of the round - larger than drafting-side options F/C/B combined. llama_process returns before the sync, so part of this may overlap GPU work; how much of it is on the critical path is unknown. Likely contributors: llama graph rebuild every round (verify batch size changes 1..n_max+1 so graph reuse misses), ggml_backend_sched split/assignment over the meta tensor-split backend, HIP graph capture/update or instantiation per shape, CPU token_embd get_rows split at graph start, input/KV-view setup.

## Steps

1. Break submit into sub-spans with ggml_time_us (graph build/reuse hit, sched alloc/split, per-backend graph_compute_async submit incl. HIP graph launch/update) and log per round; record whether llama graph reuse hit.
2. rocprof: GPU idle time between the previous sync and the first target kernel of the next round (is submit on the critical path?).
3. Candidates by evidence: (a) keep llama graph reuse across MTP verify widths (pad verify batch to a fixed n_max+1 or cache one graph per width); (b) HIP graph instances keyed per width already cached (QFP06) - confirm no recapture; (c) move CPU token_embd lookup off the critical path; (d) cheaper sched re-split when the graph is reused.
4. Patch the dominant term; ABBA ms/step.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Per-round submit_us drop with identical greedy output; ms/step ABBA at ~10K and ~80K.

## Effort & Risk

Diagnosis S-M; fix M. Padding verify batches costs a little GPU work per round; measure.

## Standards



## Acceptance Criteria



## Notes

Source: FMTP01 Gate 0 notes 2026-10-04. Related: QFP06 (HIP graph working set), QFP13 (launch count), FMTP01.

## Change Log

- 2026-10-04T05:13:42.699438+00:00 (created-by): Created by agent
