---
id: PRBE56
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:57:22.875427+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# UP-MGPU-001: Speculative scheduler caching/reduced cross-GPU synchronization

## Description

TODO. Decompose upstream PR #27173's compound multi-GPU speculative-scheduler change into 3 independently-portable subchanges (plan caching, redundant-sync removal, output mirroring). Relevance at b11126: the general multi-GPU split/sync machinery is confirmed present in the CORE scheduler, not the speculative layer -- `ggml_backend_sched_split_graph` (ggml-backend.cpp:1066) builds per-backend `ggml_backend_sched_split`s, and cross-backend synchronization uses per-split-per-copy events (`sched->events[backend_id][sched->cur_copy]`, `ggml_backend_event_synchronize`, ggml-backend.cpp:1664-1687). No speculative-specific multi-GPU plan-cache or sync-skip logic was found in common/speculative.cpp (grepped 'synchronize\|cross.*gpu\|multi.*gpu\|devices\[': only unrelated device-name-string hits at lines 50/54). This means PR #27173's compound change is either not yet in this pin, or lives in a part of the scheduler this grep pass did not localize -- genuinely TODO, and the exact insertion point needs the deeper read in step 1 before any patch is written.

## Steps

1. Read `ggml_backend_sched_split_graph` (ggml-backend.cpp:1066-1460) and the graph-compute/sync loop (ggml-backend.cpp:1628-1700) in full to find where a recurring speculative-decode graph (same shape/topology across verify calls) would be re-split from scratch each call, and where a cross-device `ggml_backend_event_synchronize` fires that a cached/pre-computed split plan could skip.
2. Grep for any existing split/plan caching: `git -C work/upstream/llama.cpp.git grep -n 'plan.*cache\|cached.*split\|reuse.*split' b11126 -- ggml/src/ggml-backend.cpp common/speculative.cpp` -- confirm absence before designing new caching.
3. Subchange (a) plan caching: cache the `ggml_backend_sched_split` array (topology: which nodes go to which backend) keyed by graph shape signature, for the speculative draft/verify graphs specifically (common/speculative.cpp's `common_speculative_impl_draft_mtp`/`_draft_simple` call sites into `llama_decode`/backend sched), invalidated on shape change. Gate behind an experimental flag; must produce byte-identical splits to the uncached path when correct.
4. Subchange (b) sync removal: identify `ggml_backend_event_synchronize` calls (ggml-backend.cpp:1664-1687) that are provably redundant for the speculative draft/verify graph's known dependency structure (e.g. two consecutive splits on the same backend needing no cross-device wait) and skip them only in that narrow, provably-safe case.
5. Subchange (c) output mirroring: a separate, independently gated experiment -- mirror the output-layer tensor to multiple GPUs to avoid one cross-device fetch, trading VRAM for reduced traffic; keep this as its own patch depending on nothing from (a)/(b).
6. Each subchange gets its own env-gated flag, its own temp-0 identity + rollback-correctness test, and is benchmarked independently on dual XTX/R9700 (MTP widths 2..8) against single-GPU and speculation-off controls before any promotion; port only subchanges with independent, repeatable wins.
7. Instrument scheduler CPU time, GPU sync count, copies/token, and effective TG per subchange.

## Detailed Solution & Technical Design

Because the exact upstream PR #27173 diff is not available offline and no speculative-specific multi-GPU caching/sync-skip code was found at this pin, this plan targets the general scheduler's split/sync machinery (ggml-backend.cpp) as the concrete implementation surface, applied narrowly to the speculative draft/verify call path. Each of the 3 subchanges is its own reversible BigCherry patch with its own flag -- never a combined all-or-nothing port, matching the item's own decomposition requirement.

## Code Samples & Guidance

Real anchors (verified via `git -C work/upstream/llama.cpp.git show b11126:ggml/src/ggml-backend.cpp`):
- Split building entry point: `void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph)` at line 1066.
- Cross-device sync: lines 1664-1687, e.g.
```cpp
            if (sched->events[prev_backend_id][sched->cur_copy] != NULL) {
                ggml_backend_event_synchronize(sched->events[prev_backend_id][sched->cur_copy]);
            }
```
Three separate patch skeletons (subchange-scoped), e.g. patches/<order>_prbe56_speculative_plan_cache/patch.toml:
```toml
schema = 1
id = "<order>_prbe56_speculative_plan_cache"
order = <next available>
state = "untested"
kind = "enhancement"
origin = "local"
backend = "agnostic"
plan-ids = ["PRBE56"]
requires = []
conflicts = []
subsystems = ["scheduler", "speculative-decoding", "multi-gpu"]
hardware = ["amd"]
validation-architectures = []
backends = []
```
(patches/<order>_prbe56_speculative_sync_skip/ and patches/<order>_prbe56_speculative_output_mirror/ follow the same shape, each with its own `order`, no `requires` on each other -- independently reversible per the item's own acceptance criteria.) patch.py for each is a `FilePatch(path="ggml/src/ggml-backend.cpp", ...)` with `Edit(anchor=..., mode="insert_before"/"replace", ...)` targeting the real text at the anchors above -- exact insertion text is TODO-VERIFY pending the full-function read in step 1 (the function bodies are long; only the entry points are confirmed here, not the exact statements to cache/skip).

## Files

ggml/src/ggml-backend.cpp; common/speculative.cpp (call-site wiring only); 3x patches/<order>_prbe56_speculative_{plan_cache,sync_skip,output_mirror}/{patch.toml,patch.py,SUMMARY.md}

## Validation

Offline: `PYTHONPATH=tools python -m bigcherry patch-lint`, `patch-rebase-check` per subchange. Temp-0 identity + rollback correctness per subchange (single flag). Hardware (Brutus, not run here): dual XTX/R9700, MTP widths 2..8, single-GPU and speculation-off controls, scheduler CPU time/sync count/copies-per-token/effective TG, via `python -m bigcherry.patch.validation_campaign`. Port only subchanges with independent wins; output mirroring additionally requires favorable VRAM/traffic evidence.

## Effort & Risk

L / high -- touches core cross-device scheduler synchronization; correctness risk is significant if sync removal is misjudged (silent data races), hence the strict per-subchange rollback/identity gate.

## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Each promoted subchange must independently preserve temp-0 identity/rollback and show repeatable scheduler or E2E benefit; output mirroring requires favorable VRAM/traffic evidence or remains unported.

## Notes

Supersedes: RD67
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd67

2026-09-24 relevance at b11126: no speculative-specific multi-GPU plan-cache or sync-skip code found in common/speculative.cpp; general scheduler split/sync machinery confirmed in ggml-backend.cpp (ggml_backend_sched_split_graph:1066, event-sync loop:1664-1687) as the real implementation surface. GPT design request req_59325a19cc8d4adb (batched with PRBE51, submitted, response pending as of this pass -- written directly from source evidence, to cross-check against GPT response once available).

## Change Log

- 2026-09-09T10:57:22.875427+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:14:36.234929+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.380667+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.183399+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:13:12.765992+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031346_repaired-four-more-migrated-pa_4345
- 2026-09-10T03:13:46.478053+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:35:42.791029+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
