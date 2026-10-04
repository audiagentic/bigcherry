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

Prep 2026-10-04 (confirm with 1320 data first: n_subgraphs and launch_us share).

Current decode path per verify round (-sm tensor, 3 devices): ggml_backend_sched -> ggml_backend_meta_graph_compute (ggml/src/ggml-backend-meta.cpp ~L1970). Subgraphs are cached while cgraph->uid is unchanged (needs_rebuild false on graph reuse). Execution loop (~L2441): for each subgraph i, for each device j: ggml_backend_graph_compute_async -> ggml_backend_cuda_graph_compute (ggml-cuda.cu ~L4810: graph key lookup, set_enabled, check_compability over all nodes, uid fast path, cudaGraphLaunch); then comm_allreduce (1291 cpu-root small path: bc_cpu_root_produce/consume kernels on each device stream, device-advanced epochs, no per-call host state besides the launch). Host cost ~ n_subgraphs x 3 x (lookup + compat scan + hipGraphLaunch) + n_subgraphs x AllReduce enqueue.

Proposed patch (one HIP graph per device per round):
1. Meta level (ggml-backend-meta.cpp): when !needs_rebuild and the round is decode-shaped (all subgraph cgraphs have <= MMVQ_MAX_BATCH_SIZE tokens; reuse the 1307 decode-graph rule) and a per-uid 'outer graph' exists and is valid -> for each device launch its outer graph instance (3 launches total) and return.
2. Capture: on the first stable reuse (2nd call with same uid, mirroring the CUDA backend warmup), begin capture (relaxed mode) on every device stream, run the normal execution loop with the CUDA backend in pass-through mode (evaluate kernels eagerly into the outer capture, no inner capture/launch), end capture per device, instantiate. Streams never wait on each other through events (cpu-root AllReduce synchronizes through pinned host memory spins), so per-stream independent captures are legal.
3. CUDA backend pass-through: add a proc-address API (registered like ggml_backend_cuda_* helpers, fetched via ggml_backend_reg_get_proc_address) e.g. ggml_backend_cuda_set_outer_capture(backend, bool); when set, ggml_backend_cuda_graph_compute calls ggml_cuda_graph_evaluate_and_capture(use_cuda_graph=false) on the capturing stream.
4. Invalidation: any needs_rebuild, uid change, max_tmp_size/max_subgraphs growth, AllReduce provider path change (must be the small cpu-root path for every subgraph), CUDA graph properties change (input pointer moves) -> drop the outer graphs and fall back for that call. Fail closed: any capture error -> disable for the process with a WARN.
5. Env gate BIGCHERRY_META_GRAPH=1; activation marker BIGCHERRY_PATCH_HIT.
Risks: 1307 Q8_1 cache begin_generation and producer decisions are host-side per call - already baked inside today's per-subgraph graphs, so outer capture does not change semantics, but verify greedy identity; memory: one extra executable graph per device per verify width; QFP06 working-set (graph LRU) unaffected but recheck VRAM at 240K.
Validation: 1319/1320 show graph_compute host ~5 ms -> < 0.5 ms; ms/step ABBA ~10K/~80K; greedy identical; prefill unaffected (not decode-shaped).
Secondary items: draft context alternating widths (1 vs 4) -> 50% graph reuse; KQ mask input O(n_kv) 0.25 ms/round at 80K.

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

2026-10-04 1320 meta breakdown at ~10K (flashnext-metatime-d10k): 97 subgraphs per verify graph (7119 nodes), rebuilds ~0 (2/84); per round launch 1.69 ms median (17.4 us per subgraph across 3 devices), AllReduce enqueue 0.54 ms (5.6 us each), meta total 2.24 ms median (mean 3.14). So the one-HIP-graph-per-device capture saves at most ~2 ms/round (~5%), not 8-10%. The other ~2.8 ms of the 5.0 ms target graph_compute host time is in the scheduler outside the meta backend - suspects: CPU splits (token_embd and the 26.8 GiB per_layer_token_embd get_rows run synchronously on the host) and their input copies/synchronization into the GPU split. 1325 (per-split scheduler timing) queued (queue-sched-split.sh). If CPU-split time dominates: move the lookups off the critical path (overlap the CPU get_rows of round n+1 with the GPU work of round n is impossible because tokens are unknown; instead make the CPU split cheaper, or move token_embd to GPU now that VRAM allows at 240K, or async the copy).

## Change Log

- 2026-10-04T05:13:42.699438+00:00 (created-by): Created by agent
- 2026-10-04T06:12:21.750192+00:00 (updated-by): Updated: section:notes
- 2026-10-04T06:17:35.435151+00:00 (updated-by): Updated: section:notes
- 2026-10-04T06:21:34.071163+00:00 (updated-by): Updated: section:notes
- 2026-10-04T06:32:01.947191+00:00 (updated-by): Updated: section:detailed_solution
- 2026-10-04T06:45:39.484407+00:00 (updated-by): Updated: section:notes
