---
id: PNRO08
order: 0
plan: patching-nasone-rdna-optimizations
state: pending
created-at: '2026-09-09T10:52:46.423935+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P1
---

# GPU-resident LRU cache for host-offloaded MoE expert weights

## Description

TODO, NOT-READY. CORRECTED per GPT review: the prior 'first-check' design (per-llama_context cache, include/llama.h params, src/llama-context.cpp wiring, ggml-backend.cpp:1693 integration point) misidentified the real host-weight MUL_MAT_ID branch. Verified at b11126: `ggml_backend_sched_compute_splits()` (ggml-backend.cpp:1646) contains the real `used_ids`/`copy_experts` expert-copy-reduction logic (used_ids bitset built at ~1731-1737, copy_experts lambda at ~1745, consumed ~1760-1781) -- it only copies routed expert ranges for the CURRENT split; it does not retain them across decode calls. This confirms the item is relevant (not upstream-absorbed) but still undesigned against the real anchor.

## Steps

1. Read ggml_backend_sched_compute_splits() (ggml-backend.cpp:1646) in full, including the used_ids bitset construction (~1731-1737) and copy_experts lambda (~1745, consumed ~1760-1781), before authoring any Edit() -- this is the real host-weight MUL_MAT_ID branch, not the ggml-backend.cpp:1693 line previously cited.
2. Anchor cache ownership in ggml_backend_sched (verify the exact struct/scheduler-state location) rather than a new per-llama_context object, since the real copy-reduction logic lives at the scheduler level, not the context level.
3. Explicitly specify: CPU skip-table construction (which expert IDs are cache-resident and should be skipped from the host copy), cached-ID remap into resident GPU slots, CUDA MUL_MAT_ID execution against the remapped IDs, the merge of CPU-computed and GPU-cached-slot results, generation publication (atomic, only after a full slice upload completes), slot lifetime, and teardown -- all currently unspecified -- BEFORE authoring patch 1262.
4. Prove exact decomposition: CPU skips cached rows, GPU remaps uncached IDs to zero slot, and summed outputs equal stock for every routing ID.
5. Start with a deterministic static mapping and synthetic routing trace; separate LRU policy from graph correctness before asynchronous replacement.
6. Model mapping generations and publish only after full expert-slice upload; test eviction, active requests, teardown, repeated context creation and callback lifetime.
7. Instrument hits/misses/inserts/evictions/bytes avoided; sweep slots/throttle across no-locality and phase-changing traces; measure host/PCIe traffic, decode TPS, VRAM and warmup.
8. Keep prefill and fully GPU-resident models as strict non-selection controls; promote only when net gain exceeds maintenance and VRAM cost.

## Detailed Solution & Technical Design

First-check command for the next agent (run before writing code): `git -C work/upstream/llama.cpp.git show b11126:ggml/src/ggml-backend.cpp | sed -n '1620,1790p'` and `git -C work/upstream/llama.cpp.git grep -n -E 'LRU|cache.*expert|expert.*cache|cached.*expert|mul_mat_id|expert.*used|offload.*MoE' b11126` -- trace the line-1693 code into its caller(s) and temporary-buffer lifetime. If upstream already keeps bounded resident expert slices persistent across decode calls with a hit/miss remap and safe eviction, close this item as UPSTREAM-ABSORBED instead of implementing. Otherwise (expected): design a new per-llama_context cache object (not model-global -- routing locality, decode lifetime, worker teardown, and concurrent contexts are context-specific): `llama_moe_expert_cache` owning per-MoE-layer cache records (device companion weight = K expert slices + one all-zero dummy slot, active expert_id->slot map, LRU metadata, current generation counter, pending generation, upload-completion event). Gate via a new opt-in context param `llama_context_params.moe_expert_gpu_cache_slots = 0` (default disabled) plus CLI `--moe-expert-gpu-cache-slots N`; additionally require decode (not prefill), host-offloaded MoE weights, and supported GPU backend before selection -- prefill and fully GPU-resident MoE models are mandatory non-selection controls. Graph decomposition per frozen generation: CPU branch runs stock host MoE weights but skips/cache-masks the cached expert IDs; GPU branch runs K resident expert slices + zero dummy, remapping each routing ID to its resident slot if cached or to the dummy slot if not; result = CPU branch output + GPU branch output, so every route is computed exactly once. Implement the static deterministic mapping (synthetic routing trace) first and prove this decomposition exactly equals stock output for every routing ID before adding any LRU/async replacement. Async phase: the active generation/map is immutable for an in-flight decode; the upload worker only touches slots not referenced by that in-flight generation, uploads a complete expert slice, waits on/records completion, builds a pending host+device map, and only at the next decode boundary atomically publishes the pending generation -- never overwrite active slot contents or mutate the active map in place. Context destruction must join/cancel the worker and destroy events before tensors/state. Activation marker (once per context): `pnro08: moe expert GPU cache active slots=<K>`.

## Code Samples & Guidance

Patch package: `patches/1262_pnro08_moe_expert_gpu_cache/{patch.toml,patch.py,validation/producer.py,SUMMARY.md,README.md,TESTING.md}` (id="1262_pnro08_moe_expert_gpu_cache", order=1262, kind="enhancement", state="untested", requires=[] -- adjust the numeric slot if PNRO11-13/15 claim 1262 first; use the actual next-free id at authoring time). patch.py skeleton (`from bigcherry.patcher import Edit, FilePatch`), every anchor below is NEEDS-VERIFICATION against real b11126 source before authoring -- do not paste these anchor regexes into patch.py unverified: (1) `include/llama.h` near `llama_context_params` struct -- insert `moe_expert_gpu_cache_slots` field; (2) `common/arg.cpp` near the CLI params parser init -- insert `--moe-expert-gpu-cache-slots`; (3) `src/llama-context.cpp` at context construction/destruction and the decode-boundary call site -- add cache ownership, eligibility check, generation publication, teardown; (4) `ggml/src/ggml-backend.cpp` at the function enclosing line ~1693 -- integrate resident-hit handling without disturbing the existing stock expert-copy-reduction; (5) CPU mul_mat_id implementation (cached-ID skip mask) and GPU mul_mat_id path (expert-ID -> resident-slot/dummy remap) -- exact files NEEDS-VERIFICATION (likely ggml/src/ggml-cpu/ops.cpp and a ggml-cuda/hip mul-mat-id kernel file).

## Files

include/llama.h; common/arg.cpp; src/llama-context.cpp; ggml/src/ggml-backend.cpp; ggml CPU mul_mat_id; GPU/HIP mul_mat_id kernel; patches/1262_pnro08_moe_expert_gpu_cache/*; synthetic routing-trace fixtures; hit/miss/insert/eviction/bytes-avoided counter evidence.

## Validation

Offline: `PYTHONPATH=tools python -m bigcherry patch-lint patches/1262_pnro08_moe_expert_gpu_cache`; `PYTHONPATH=tools python -m bigcherry patch-rebase-check --focal-overlay 1262_pnro08_moe_expert_gpu_cache --source bigcherry-tuning`. Correctness gates in order: (1) static deterministic mapping (e.g. cache experts {1,4}) with synthetic routing covering cached-only/uncached-only/mixed/repeated-IDs/duplicate-routes/all-IDs/zero-hits/full-hit cases, asserting CPU(skip cached)+GPU(remap uncached->dummy) == stock for every route/shape; (2) temp-0 token/logit identity vs stock; (3) prefill and fully-resident models never activate (non-selection controls); (4) LRU eviction and phase-changing/no-locality traces remain correct; (5) generation publication never exposes partially-uploaded slices; (6) active decode's resident slot is never overwritten mid-flight; (7) context destroy/recreate and multiple simultaneous contexts are clean (no leaks); (8) hit/miss/insert/eviction/bytes-avoided counters reconcile exactly. Hardware (Brutus, not run here): `python -m bigcherry.patch.validation_campaign --overlay 1262_pnro08_moe_expert_gpu_cache --arch gfx1100` sweeping slots/throttle across no-locality and phase-changing traces, recording PCIe/host traffic, VRAM, warmup, decode TPS; promote only if decode gain survives VRAM/maintenance cost under negative-locality controls.

## Effort & Risk

Work=L (already set). High correctness risk: async publication race (partial upload exposed, active slot overwritten mid-decode) is the primary hazard the design must close before any performance work; static-mapping-first staging is mandatory, not optional.

## Standards

Correctness first; bounded resources; source SHA; no global callback leaks; decode-only until separately expanded.

## Acceptance Criteria

Cache path is selected only for host-offloaded decode; exact outputs and safe publication hold under stress; measured traffic reduction and end-to-end gain outweigh maintenance/VRAM budget; otherwise reject without using third-party hit-rate claims.

## Notes

Supersedes: NRO09
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-nasone-rdna-optimizations-nro09

2026-09-24 relevance at b11126: TODO. GPT design request req_dcceb6cbd73245d6 (dev-gpt-agent) completed successfully and its design (per-context llama_moe_expert_cache, CPU-skip/GPU-remap decomposition, static-map-before-LRU staging, generation-counter publication) is incorporated above. GPT flagged that ggml-backend.cpp:1693's existing 'copy only used experts' optimization must be read in full by the next agent before authoring, in case it already covers this scope (would reclassify to UPSTREAM-ABSORBED) -- this was not yet confirmed either way in this session.

2026-09-24 GPT review req_215c89d0b13a4bb7 applied: corrected the real anchor from ggml-backend.cpp:1693 to ggml_backend_sched_compute_splits() (ggml-backend.cpp:1646), with the actual used_ids/copy_experts logic verified at ~1731-1781. Cache ownership moved from a proposed new per-llama_context object to ggml_backend_sched itself, matching where the real copy-reduction logic lives. Required explicit CPU skip-table/GPU remap/merge/generation-publication/teardown design before authoring patch 1262 (previously left to the implementer).

## Change Log

- 2026-09-09T10:52:46.423935+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:09:08.508331+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.086059+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.732296+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:42:53.658407+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_024304_three-nasone-successors-now-pr_2691
- 2026-09-10T02:43:04.884669+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:32:53.119489+00:00 (updated-by): Updated: section:description, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk
- 2026-09-24T02:33:08.094920+00:00 (updated-by): Updated: section:notes
- 2026-09-24T04:49:41.845664+00:00 (updated-by): Updated: section:description, section:steps, section:notes
