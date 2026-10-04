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

2026-10-04 1318 draft-loop timing (flashnext-drafthost, v4 diag build): per MTP draft step ~10K / ~80K: submit (llama_process(ctx_dft) host) 0.53 / 0.89 ms, draft-GPU wait 1.36 / 1.77 ms, host rest (sampling over trimmed 65K vocab + nextn hidden read) 0.23 / 0.24 ms; draft call 6.36 / 8.74 ms (3 steps). Host share of the draft ~36-40%. Submit grows with context on both contexts (target 5.06 -> 5.83 ms/round 10K -> 80K; draft 0.53 -> 0.89 ms/step), pointing at O(n_kv) host input work per call - prime suspect set_input_kq_mask (seen in host perf: set_input_kq_mask_impl<unsigned short,true>) rebuilding the full KQ mask on the CPU every step, plus kpool/QSA inputs. Fix candidates once 1319 confirms the inputs_us share: incremental mask update (append-only positions for decode), device-side mask construction, or caching mask rows across reused graphs. Same lever applies to the draft context (FMTP01 F/B scope).

2026-10-04 1319 submit split at ~10K (flashnext-submitsplit-d10k): target ctx (4-token verify, 98% graph reuse) graph 0.003 ms, inputs 0.09 ms, graph_compute host 5.01 ms median (mean 5.83) - the KQ-mask hypothesis is wrong; essentially all of the submit is inside ggml_backend_sched_graph_compute_async over the meta (-sm tensor) backend. Draft ctx (6900): graph reuse only 50% (alternates 1-token draft steps and 4-token process), graph 0.13 ms, inputs 0.04, compute 0.42 ms. Meta backend structure: graph split at every AllReduce; per subgraph x per device one ggml_backend_graph_compute_async (one HIP graph launch) then the AllReduce enqueue - ~(n_subgraphs x 3) graph launches per verify round. 1320 queued to measure n_subgraphs and launch vs AllReduce host time. Candidate fix if launches dominate: since decode-size AllReduces are stream kernels (1291 cpu-root produce/consume), capture each device's whole subgraph+AllReduce sequence into ONE HIP graph per device per round (graph reuse keeps topology fixed), turning ~100 launches per device into 1. Potential ~3-4 ms/round (~8-10%). Draft graph reuse: cache two graphs (width 1 and width n_max+1) instead of one.

2026-10-04 1319 at ~80K: target graph 0.003 ms, inputs 0.33 ms (0.09 at 10K), graph_compute 5.53 ms median (mean 6.98); draft inputs 0.20 ms (0.04 at 10K), compute 0.56 ms, graph reuse 50%. Input setup scales with n_kv (KQ mask built on the CPU) - real but secondary (~0.25 ms/target round + ~0.16 ms/draft step at 80K, ~1%); the meta graph_compute host time (5-5.5 ms) is the main term.

## Change Log

- 2026-10-04T05:13:42.699438+00:00 (created-by): Created by agent
- 2026-10-04T06:12:21.750192+00:00 (updated-by): Updated: section:notes
- 2026-10-04T06:17:35.435151+00:00 (updated-by): Updated: section:notes
- 2026-10-04T06:21:34.071163+00:00 (updated-by): Updated: section:notes
