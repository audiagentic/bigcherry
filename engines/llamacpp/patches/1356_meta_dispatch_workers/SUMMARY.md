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

The strongest explanation for the fixed alternative text plus exactly one missing capture is the upstream UID fast
path in \`ggml_cuda_graph_update_required()\`: when the current nonzero \`cgraph->uid\` equals the cached graph UID it
returns false before constructing/comparing \`node_properties\`. That makes the UID an assertion that every tensor
pointer/shape/stride/op parameter used by the captured graph is unchanged.

That assertion is too strong for Meta + 1340. 1340 binds the current per-device simple graph into its arena before
compute, so simple-tensor data bindings can change independently of the Meta subgraph object's UID. Fusion admission
also depends on those addresses (\`ggml_cuda_check_fusion_memory_ranges\`). A false UID hit therefore has exactly the
observed shape: replay one old fused graph instead of re-capturing it. Graphs-off removes the stale replay; fusion-off
removes the address-sensitive fused topology/arguments.

1356 now disables only that UID shortcut while \`BIGCHERRY_META_DISPATCH_THREADS=1\`; worker-off remains upstream.
The full existing property comparison then decides capture/update/replay. This is deliberately narrower than
serializing all graph_compute calls and preserves replay overlap.

### Mutable-state audit reachable from CUDA graph_compute

| state | writer | reader | scope | 1356 lock |
|---|---|---|---|---|
| \`ggml_cuda_graph::{uid,node_props,warmup_complete,graph,instance,nodes}\` | graph decision/capture/update | graph decision/replay | per CUDA backend context + graph key | decision is before lock; capture/update/replay body is covered |
| \`cuda_ctx->cuda_graphs\` incl. 1302 OOM eviction | graph lookup/update/evict | graph lookup | per CUDA backend context | lookup/decision before lock; eviction occurs in exclusive capture/update |
| \`ggml_cuda_stream_context::{concurrent_events,stream_mapping,original_order}\` and per-call cgraph node-order restore | CUDA graph evaluate | CUDA graph evaluate/fusion | per CUDA backend context/subgraph | covered for capture/replay; direct evaluation is unlocked |
| Meta \`backend_config.cgraphs/nodes\`, simple-tensor containers, split-state cache | Meta caller thread | workers | per Meta context | intentionally outside lock; caller finishes mutation before submit and waits before reuse |
| 1340 arena plans/bindings/simple-tensor \`data\` | Meta caller thread before dispatch | CUDA graph property/fusion code | per backend config, but changes can outlive a reused subgraph UID | not covered by graph lock; fixed by forcing full property comparison in worker mode |
| 1307 Q8_1 activation cache/generation/capture-active | each CUDA graph_compute | MMVQ/fused producers | per CUDA backend context | backend-local; cache-off hardware still reproduces, ruled out as primary |
| 1339 fusion-overlap counters (\`bc_fusion_overlap_stats\`) | fusion memory-range checks | exit diagnostic | process-global plain counters | direct calls are unlocked; a diagnostic counter data race under workers, but it does not feed fusion decisions/output |
| 1342 fusion-family \`std::map\` counters | successful fusion call site when \`BIGCHERRY_META_MEM=1\` | exit diagnostic | process-global | direct calls are unlocked; unsafe diagnostic-only state when that env is enabled |
| fusion env/lazy flags (upstream disable flag, 1313/1334/1350/1355/1357 gates) | C++ thread-safe first-use init | fusion/kernel selection | process-global read-only after init | no lock needed |
| one-shot trace booleans in several production patches | trace path | trace path | process-global diagnostic | some are non-atomic; default-off and do not feed math/graph decisions |
| HIP current device | each worker via \`ggml_cuda_set_device(cuda_ctx->device)\` | kernel/fusion dispatch | HIP thread-local runtime state | no shared lock needed |
| \`ggml_cuda_lock_counter\` / destructor mutex+cv | capture begin/end/destruction | teardown | process-global atomic/mutex | independent upstream synchronization; does not decide graph reuse |

The 1339/1342/one-shot diagnostic races are real concurrency hygiene issues but are ranked below the UID false-hit:
they do not feed the graph-update decision, and the observed failure is specifically one missing capture. Avoid
\`BIGCHERRY_META_MEM=1\` during the 12-run acceptance so its map race cannot confound the result.

### Hardware diagnostic

\`BIGCHERRY_META_DISPATCH_GRAPH_TRACE=1\` emits one line per device graph_compute call:

\`BIGCHERRY_1356_GRAPH device=<d> call=<per-device-n> decision=<direct|capture|update|replay> props=<fnv64> uid=<uid> ...\`

\`props\` is FNV-1a over the exact \`ggml_cuda_graph::node_properties\` representation used by
\`ggml_cuda_graph_update_required()\`, including node bytes and every direct source data pointer/shape/stride.
Diff a clean and differing 98K run by \`device,call\`; the first decision/hash divergence is the useful locus.

Acceptance remains twelve 98K worker-on runs with identical greedy text. Keep graphs and fusion on. If the race
survives this full-compare fix, the trace distinguishes the next branch: a differing property hash with replay means
the decision path is still wrong; identical hashes with different output points at fused launch state not represented
by \`node_properties\`.
