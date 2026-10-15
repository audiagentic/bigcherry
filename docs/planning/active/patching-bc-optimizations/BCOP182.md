---
id: BCOP182
order: 182
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-13T09:06:00Z'
created-by: agent
priority: P2
work: S
---

# QFP27: fence upstream scheduler input-event reuse on Meta tensor split

## Discovery / change

Pinned llama.cpp b11552 enables pipeline-parallel scheduler copies only for layer split. `ggml_backend_sched_new` therefore uses `n_copies=1` on BigCherry's production tensor-split Meta target; upstream open PR #29963 (head `273cd26`, updated 2026-10-12) uploads inputs early and records `input_events` only for `n_copies>1`. Meta reports `caps.events=false` and `event_new=nullptr`, while mirrored `set_tensor_async` enqueues on multiple child streams. A single/null event cannot establish safe pinned-source reuse. This is a source-level admission/lifetime finding, not a reproduced GPU corruption or measured speedup.

## Owners / already acting

QFP16 and patch 1326 own the validated <=4 MiB pageable async-input shortcut (historical safe staged decode +9.4% ~8K / +7.8% ~64K, no new measurement). QFP27 owns upstream applicability and child-completion qualification. QFP24 owns conditional pinned backing only; MET06 owns host-expert policy. Recent independent Radiance helper/RAD01 GPU work is excluded and untouched. Do not port #29963's MoE placement or create a second scheduler/copy registry.

## Unresolved action / terminal gate

Use existing 1325/1326 instrumentation to prove large-input prefill cost >=1 ms AND >=3% wall on the exact pin. Otherwise close. If positive, implement only within 1326: <=64 MiB scheduler-owned pinned slots, every Meta child completion joined before reuse, fail-closed pageable/synchronous fallback before partial submission, 10,000-generation host lifetime tests, multi-request/ubatch/MTP/logits/KLD and four paired hardware sessions. Promote >=5% median prefill/TTFT, CI95-low>0, <=2% decode regression, zero stale generations; otherwise reject/retire. Do not queue competing hardware work.

## Evidence

Actual-source checks 10/10; source-shaped host fixture 9/9 plus Python compile. No repository pytest, HIP build, GPU execution or new throughput result. Sources: [llama.cpp #29963](https://github.com/ggml-org/llama.cpp/pull/29963), [vLLM pinned H2D](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py), BigCherry QFP16/QFP24/QFP27 and 1326 SUMMARY.
