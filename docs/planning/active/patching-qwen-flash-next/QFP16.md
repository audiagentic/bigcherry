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

2026-10-04 1325 scheduler split timing at ~10K (flashnext-schedsplit-d10k): target verify graph = 2 splits: CPU (4 nodes, token_embd/per_layer_token_embd get_rows) compute 0.27 ms; Meta(ROCm0-2) split with 28 inputs: input handling 2.63 ms median + launch 2.23 ms (= 1320's meta total). Draft (ROCm3) split, 15 inputs: input handling 0.40-0.45 ms per call (~4 calls per round incl. process) + launch 0.01-0.02 ms. ROOT CAUSE of the remaining submit time: ggml_backend_sched_compute_splits copies split inputs synchronously - the CUDA/meta backends' cpy_tensor_async does not accept a CPU source, so each input takes the fallback (ggml_backend_synchronize(input_backend), event/backend synchronize of the split backend, blocking ggml_backend_tensor_copy), and mirrored meta inputs go to all 3 devices. Fix candidates (new patch, ~4 ms/round = ~10%): (a) allocate the CPU split outputs/inputs in a pinned host buffer (cuda host buffer type) and in the scheduler fallback use the split backend's set_tensor_async from host memory when the source buffer is host-resident (ordering is safe: the copy is enqueued on the split backend's stream before its compute; the split-backend event wait already protects reuse of the input copy); (b) meta backend: implement set_tensor_async/cpy_tensor_async for host->mirrored by fanning out per-device async copies. Priority above the one-graph-per-device capture (~2 ms).

## Change Log

- 2026-10-04T05:13:42.699438+00:00 (created-by): Created by agent
- 2026-10-04T06:12:21.750192+00:00 (updated-by): Updated: section:notes
- 2026-10-04T06:17:35.435151+00:00 (updated-by): Updated: section:notes
- 2026-10-04T06:21:34.071163+00:00 (updated-by): Updated: section:notes
- 2026-10-04T06:32:01.947191+00:00 (updated-by): Updated: section:detailed_solution
- 2026-10-04T06:45:39.484407+00:00 (updated-by): Updated: section:notes
- 2026-10-04T07:05:05.821690+00:00 (updated-by): Updated: section:notes


## 2026-10-06 post-1326 ownership and residual gate

1326 has already resolved the dominant decode-side QFP16 finding and is in validated-enhancements. Its safe staged result measured +9.4% decode at ~8K and +7.8% at ~64K with greedy identity. 1325 had attributed ~2.63 ms/target verify round and ~0.40-0.45 ms/draft call to split-input handling. Therefore the earlier outer-HIP-graph design above is no longer the first action. Reopen it only if fresh post-1326 profiling still places >=1 ms/round of critical-path host time in Meta launch/AllReduce.

The remaining QFP16 question is large/prefill host inputs. The unsafe direct-source version had shown +3.7%/+7.2% prefill, but caller-owned pinned inputs can be rewritten before DMA completion. Scheduler-owned pageable staging fixed that lifetime issue but measured -2.3%/-2% prefill, so 1326 correctly caps its fast path at 4 MiB.

ROCm HIP documentation makes the mechanism explicit: hipMemcpyAsync with non-pinned host memory is performed synchronously; pinned/page-locked host memory is required for genuinely asynchronous/overlapped H2D. Thus current pageable staging should be described as a lifetime-safe synchronization-elision path, not true overlapped H2D.

Current llama.cpp master still falls back to synchronizing/copying when cpy_tensor_async cannot handle a host source, so upstream has not superseded 1326 as of 2026-10-06.

### Cheapest discriminator before more code

Instrument the current safe 1326 path on representative prefill ubatches and report: bytes staged; host memcpy time; set_tensor_async call time; destination synchronization/fallback time; total split input-handling time; source/staging pinned status; Meta destination count. Run ubatch 512/2048/4096 on the current production composition.

**Stop** if large-input handling is <1 ms or <3% of prefill wall time. Do not implement another scheduler mechanism.

### Only candidate if the discriminator passes: bounded pinned staging ring

Extend 1326 rather than creating a new scheduler-copy owner. Allocate scheduler-owned pinned host slots through an existing backend host-buffer primitive. Reuse a slot only after every destination stream that consumed its prior contents has completed. For Meta mirrored fan-out, completion on one device is insufficient; slot reuse must join all consuming devices. Bound pinned storage (initial qualification cap 64 MiB) and fall back to current 1326/upstream behavior if the pool/event contract is unavailable or full.

Do not add a new runtime option unless qualification proves an independently selectable production policy is required. Preserve the current <=4 MiB safe path as the control.

Implementation seams if promoted:
- ggml/src/ggml-backend.cpp: ggml_backend_sched_compute_splits and scheduler-owned slot lifetime.
- ggml/src/ggml-backend-meta.cpp only if a generic completion/join hook is required; do not duplicate existing fan-out.
- patches/1326_sched_async_host_inputs/patch.py and its existing mechanics test.

Correctness must include >=3 requests in one process, multi-ubatch prefill, forced slot wrap/reuse, ubatch-size changes between requests, greedy identity, bounded pinned bytes and physically plausible transfer accounting.

Hardware promotion requires >=5% median prefill improvement at ubatch 2048 or 4096, no >2% decode regression and no correctness failure. Otherwise close the residual and retain 1326 unchanged.

External mechanism references: ROCm HIP host-memory/asynchronous-copy documentation; current llama.cpp ggml_backend_sched_compute_splits; vLLM's use of pinned host inputs for non-blocking H2D and event-scoped reusable offload buffers.
