# 1356_meta_dispatch_workers

**Status:** evaluated
**Plan item:** QFP41

## Mechanism

`ggml_backend_meta_graph_compute` currently submits each simple backend's `cgraph_main` serially from one host
thread. With `BIGCHERRY_META_DISPATCH_THREADS=1`, this package creates one persistent worker per simple backend
and submits the same subgraph index to those workers concurrently. The caller joins all submissions before any
AllReduce/collective decision, exactly where the serial loop had completed all host submissions.

No thread is created per graph. Default is off during qualification.

## Shared-state boundary

Caller-thread only:
- meta graph rebuild and `backend_configs[*].cgraphs/nodes` mutation;
- split-state cache and rotating simple-tensor containers;
- 1340 arena planning/binding and graph allocation;
- auxiliary AllReduce graphs, copy scheduling, collective selection/fallback and comm context;
- scheduler callbacks/input staging before/after meta graph compute.

Worker-owned during a job:
- exactly one distinct `backend_config.backend`;
- its simple-backend graph submission and backend-local CUDA/HIP graph/pool state.

The join occurs before the caller reads the submitted subgraphs for AllReduce. Workers are destroyed before the
simple backends in the meta-context destructor.

## Evidence boundary

QFP41 records ~40% split idle as the opportunity, not a proven dispatch-thread saving. No performance or thread-safety
claim is made until identity, repeatability, stress and hardware ABBA pass.

## Shared state: CUDA graph capture (2026-10-09)

First hardware run (b-metamem-mdw1, Flash-Next production profile, two ABBAs per depth): prefill +2.5% at 8K, +0.5 to
+1.5% at 24K, +0.3% at 98K with every threaded run above every baseline run, but 5 of 12 threaded runs produced a
different greedy text (one garbage). Bisect on the same binary at 8K, three ABBAs per variant: workers only 2 of 6
different; fusion off 5 of 6 different; `GGML_CUDA_DISABLE_GRAPHS=1` 0 of 6 different with the prefill gain intact
(1249-1289 against 1209-1238 t/s).

Cause: `ggml_backend_cuda_graph_compute` captures, instantiates and updates HIP graphs per device, and with the
workers on those run concurrently with another device's capture or launch. Fix in this package
(`ggml/src/ggml-cuda/ggml-cuda.cu`): a process-wide `std::shared_mutex`, taken exclusively by a capturing call and
shared by a replaying call, held until the call returns. It is only taken when `BIGCHERRY_META_DISPATCH_THREADS` is
on, so the default path is unchanged. Direct (non-graph) evaluation stays unlocked; the graphs-off variant showed it
is safe across devices.

Re-test required after this change: repeated-run identity at 8K (at least six threaded runs), ABBA at 8K / 24K / 98K,
and a long decode.

## Lock counters (2026-10-10)

`BIGCHERRY_META_DISPATCH_STATS=1` prints one line when the process exits:

    BIGCHERRY_1356_LOCK_STATS capture n=.. wait_ms=.. hold_ms=.. | replay n=.. wait_ms=.. hold_ms=..

`capture` counts the `graph_compute` calls that needed a HIP graph capture or update (the exclusive lock), `replay`
the ones that replayed (the shared lock); `wait_ms` is the time spent waiting for the lock and `hold_ms` the time
from taking it to the end of the call. The calls are counted with the workers off as well, where no lock is taken,
so the share of capturing calls can be read off a production run first.

Why: the lock above is exclusive for the whole of a capturing call. If most prefill calls capture, the workers run
the devices one at a time, which would match the measured gain shrinking from +2.5% at 8K to +0.3% at 98K. The
counters say whether that is so before the lock is narrowed. The mechanism and the lock are unchanged.

## Demoted to evaluated (2026-10-11)

On the current build (main at 1042e84d plus the lock counters of PR #129), Flash-Next production profile, 98K, with
the request sent through the chat template (#124): 3 of 8 runs with the workers on give a different greedy text;
all 8 runs with the workers off agree (runs lk1, lk2r1, lk2r2, lk2r3). The three differing runs give the same
alternative text (md5 10a864c9 against b301a49a) and each counts one HIP graph capture fewer (1977 against 1978).
The run position does not decide it (2-B, 3-B, 2-B). 8K and 24K are identical in every run.

This contradicts the identity claim the promotion was made on (identical at 8K, 24K and 98K, two ABBAs each, and a
52-run stress at 8K). Those runs sent a raw completion whose first token was end-of-turn, which the harness then
forced past; the 98K stress was never run. Two other differences from then are not ruled out: the profile now
switches on the split-K MoE router (1357), and the counters add timing code inside the locked section.

State is `evaluated`: the prefill gain is measured (+3.5% / +1.7% / +1.6% at promotion, +4.4% / +1.0% / +1.1% in
run lk1) and the output race is open. The patch is out of the production recipe and composed by the
`meta-dispatch-workers` experiment. The flag was default off and in no profile, so no served configuration changes.

To promote again: find the race (the missing capture is the lead) and fix it in this package, then at least twelve
runs with the workers on at 98K with identical text, and 98K added to the stress.

## The race needs HIP graphs and the fusion pass (2026-10-11, runs mdw4r*, mdw4s*, rb-*, hp2, hp4)

All at 98K on `b-mdw4` (production + 1356, no lock counters), workers on:

| change to both arms | workers-on runs with a text different from the workers-off one |
|---|---|
| none | 1 of 6 |
| router split-K off (`BIGCHERRY_MOE_ROUTER_SPLITK=0`) | 1 of 4 |
| sparse flash attention off (`BIGCHERRY_FA_SPARSE=0`) | 2 of 6 |
| deferred catch-up off (`BIGCHERRY_MTP_DEFERRED_CATCHUP=0`) | 2 of 6 |
| asynchronous inputs off (`BIGCHERRY_SCHED_ASYNC_INPUTS=0`) | 3 of 6 |
| HIP graphs off (`GGML_CUDA_DISABLE_GRAPHS=1`) | 0 of 6 |
| fusion pass off (`GGML_CUDA_DISABLE_FUSION=1`) | 0 of 6 |

So the lock counters, the router switch, sparse attention, the deferred catch-up and the asynchronous inputs are
cleared, and the race goes away when either HIP graphs or the fusion pass is taken out.

The HIP runtime itself is cleared by the standalone probes (`tools/lab/hip-probes`): one thread a card, 900 rounds
of concurrent capture + instantiate + launch (run hp2) and 2,700 rounds mixing in-place graph update, replay and
direct launches under four locking rules including this patch's and none at all (run hp4), with zero wrong results
and zero errors. Those graphs are small; a fault that needs the model's large graphs would not show there.

What is left is state in our own code that is shared between cards on the path where a fused node is captured
into or replayed from a graph. That is the next thing to read: the fusion pass's per-node decisions and anything
they cache across devices.


## Code-read follow-up: graph decision race (2026-10-11)

### Ranked causes

1. **CUDA-graph UID false hit — strongest match, hardware confirmation required.** Upstream
   \`ggml_cuda_graph_update_required()\` returns \`false\` immediately when the nonzero \`cgraph->uid\` matches the
   cached graph UID, before comparing \`node_properties\`. That bypasses the only check of tensor bytes,
   source-data pointers, shapes and strides. 1340 re-binds the current per-device simple graph through
   \`ggml_gallocr_alloc_graph_reuse_from()\` before compute; the CUDA graph cache does not participate in that
   binding. If one same-UID call reaches graph_compute with a changed bound property, the UID shortcut classifies
   exactly one required direct/warm-up/re-capture transition as replay. That matches the fixed alternative text and
   exactly one missing capture better than a generic memory race. It is still a code-read hypothesis until the 12x
   98K run confirms it.
2. **1339/1340 fusion-overlap counters — definite data race, weak match to the symptom.**
   \`bc_fusion_overlap_stats.checks/refused\` were process-global plain \`uint64_t\` and every concurrent fusion
   memory-range check writes them. They do not feed fusion admission or graph reuse, so they should not cause one
   missing capture, but the behavior is C++ UB. 1356 now makes only these counters atomic.
3. **Optional diagnostic fusion counters — real only when enabled.** 1342's process-global
   \`bc_fusion_taken_stats.by_op\` map is mutated only with \`BIGCHERRY_META_MEM=1\`; it is not worker-safe. Keep that
   diagnostic off for acceptance. It does not run in the production profile by default.
4. **Fused-node cross-device mutation / per-process fused table — not found.** Upstream and 1313/1334/1350/1355/1357
   fusion paths build launch arguments/local fused tensors on the stack. The fusion pass reorders each device's own
   cgraph node-pointer array for stream scheduling but does not rewrite shared tensor \`op/src/op_params\`, and there
   is no process-global fused-node table.
5. **Q8_1 / dispatcher caches — scoped or synchronized.** 1307's cache is owned by
   \`ggml_backend_cuda_context\` and internally mutexed; its \`ggml_hip_q81_decode_graph\` flag is intentionally
   \`thread_local\`, which becomes one stable copy per persistent device worker. 0200's dispatcher has a TLS L1,
   mutex-protected process L2, per-device \`call_once\` hardware cache, atomic counters, and a TLS recursion guard.
   No unsynchronized result cache reachable from production graph_compute was found.

1356 therefore bypasses only the UID shortcut while \`BIGCHERRY_META_DISPATCH_THREADS=1\`; worker-off remains
upstream. The existing full node-property comparison decides direct/capture/update/replay. This preserves replay
overlap and avoids serializing the fusion pass.

### Mutable-state audit reachable from CUDA graph_compute

| state / file | writer | reader | scope | 1356 coverage |
|---|---|---|---|---|
| \`ggml-cuda.cu: ggml_cuda_graph::{uid,node_props,warmup_complete,graph,instance,nodes}\` | graph decision/capture/update | graph decision/replay | per CUDA backend context + graph key | decision precedes lock; capture/update/replay body locked; UID shortcut bypassed in worker mode |
| \`ggml-cuda.cu: cuda_ctx->cuda_graphs\` incl. 1302 OOM eviction | lookup/create/update/evict | graph decision/replay | per CUDA backend context | per-device; eviction is inside exclusive capture/update |
| \`ggml-cuda.cu: ggml_cuda_stream_context::{concurrent_events,stream_mapping,original_order}\` | graph evaluate | graph evaluate | per CUDA backend context | per-device; direct evaluation is intentionally unlocked |
| \`ggml-cuda.cu: cgraph->nodes ordering\` | graph evaluate stream reorder/restore | fusion/evaluate | per-device Meta subgraph node-pointer array | no shared tensor-field mutation found |
| \`ggml-backend-meta.cpp: backend_config.cgraphs/nodes\`, simple-tensor containers, 1358 split-state cache | Meta caller | workers | per Meta context/backend config | caller mutates before submit and waits before reuse |
| \`1340/ggml-alloc.c: arena plans/arena_galloc + tensor data bindings\` | caller-side allocator before graph_compute | graph-property/fusion code | per backend config | outside graph lock; full property compare protects graph reuse from changed bindings |
| \`hip-q81-cache.*: q81_cache\` generation/cursor/entries/slabs/capture_active | graph_compute + Q8_1 producers | MMVQ/fused producers | per CUDA backend context, internal mutex | independent synchronization; not cross-device |
| \`hip-q81-cache.h: ggml_hip_q81_decode_graph\` | 1307 graph_compute | 1309-1312 producers | thread-local | one persistent worker/device; reset for each cache-enabled graph |
| \`0200/hip-autotune-dispatch.cu: g_thread_bindings, g_in_dispatch\` | dispatch resolver | dispatch resolver | thread-local | one persistent worker/device; keys include device/signature |
| \`0200/hip-autotune-dispatch.cu: g_bindings\` | dispatch resolver | dispatch resolver | process-global | guarded by \`g_bindings_mutex\`; exact HW/signature recheck |
| \`0200: hardware identity cache\` | first lookup | dispatch resolver | per-device static | \`std::call_once\`, immutable after init |
| \`1339/1340: bc_fusion_overlap_stats\` | fusion memory-range check | exit diagnostic | process-global | **fixed in 1356 to atomics**; did not feed decisions |
| \`1342: bc_fusion_taken_stats.by_op\` | successful fusion when \`BIGCHERRY_META_MEM=1\` | exit diagnostic | process-global map | not covered; optional diagnostic must stay off during threaded acceptance |
| upstream/1313/1334/1350/1355/1357 env/lazy gates | magic-static first use | kernel/fusion selection | process-global read-only | C++ thread-safe init; no lock needed |
| one-shot patch traces | trace paths | trace/report | mostly atomics; some diagnostic-only statics | process-global diagnostics | default off; no math/graph decision input found |
| HIP current device | worker via \`ggml_cuda_set_device(cuda_ctx->device)\` | HIP launch path | HIP thread-local runtime state | naturally per worker |
| \`ggml_cuda_lock_counter\` + destructor mutex/cv | graph capture begin/end / teardown | teardown | process-global atomic/mutex | upstream synchronization; does not choose reuse |
| 1281 MoE ID range helpers / 1350 / 1355 / 1357 launch selection | current call | current call | stack/per-context + immutable gates | no mutable cross-device cache found |

### Hardware diagnostic

\`BIGCHERRY_META_DISPATCH_GRAPH_TRACE=1\` emits one line per device graph_compute call:

\`BIGCHERRY_1356_GRAPH device=<d> call=<per-device-n> decision=<direct|capture|update|replay> props=<fnv64> uid=<uid> ...\`

\`props\` is FNV-1a over the same \`ggml_cuda_graph::node_properties\` image used by
\`ggml_cuda_graph_update_required()\`: node bytes plus every direct source data pointer/shape/stride. Diff a clean
and differing 98K run by \`device,call\`; the first decision/hash divergence localizes the bad graph_compute.

Acceptance: twelve 98K worker-on runs, graphs and fusion on, identical greedy text. Keep \`BIGCHERRY_META_MEM\` off.
If this fix still differs, a clean-vs-bad trace with the same property hash but different output shifts the lead from
the graph decision to fused launch state not represented by \`node_properties\`.
