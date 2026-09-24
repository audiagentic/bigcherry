---
id: PRBE52
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:57:05.575180+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# UP-MTP-001: Adaptive MTP draft depth

## Description

IMPLEMENTED-AS-PATCH, needs wiring. Existing patch patches/1255_nro06_adaptive_mtp_depth/ (state=untested, plan-ids=[NRO06]) adds a pure `bigcherry_nro06_adaptive_mtp` controller struct (n_cur/n_climb/n_drop, climb_threshold/drop_pressure/reset/update) inserted before `struct common_speculative_impl_draft_mtp` in common/speculative.cpp -- confirmed via direct read of patch.py. It is deliberately unwired: 'Runtime MTP remains fixed-depth until the controller's exhaustive state-machine tests and request-reset plumbing land' (patch.py comment). This item's remaining scope is exactly that wiring. Relevance at b11126: `common_speculative_impl_draft_mtp`'s ctor reads `this->params.n_max`/`this->params.n_min` (common_params_speculative_draft, common/speculative.cpp:1330-1380) as the per-request draft-depth bounds already -- these are the natural floor/cap hook points for the adaptive controller.

TODO (reopened detail). Wire the already-authored `bigcherry_nro06_adaptive_mtp` controller (patch 1255, unwired) into `common_speculative_impl_draft_mtp`. GPT confirmed the source anchor is real: `get_fa_tuning_params_scalar()`-style enclosing struct is known, and GPT supplied concrete anchor points superseding the prior TODO-VERIFY placeholders: state lives beside the existing `pending_h` member; reset happens in `common_speculative_impl_draft_mtp::begin`; the depth-limit check to replace is `if (params.n_max <= (int) result.size())` (adaptive effective-depth selection replaces this fixed comparison); per-seq drafted count must be recorded; controller `update(last_n_draft[seq_id], n_accepted, ...)` is called in `accept()`. Config surface (`n_min_adaptive`) must be added as explicit disabled-by-default fields in `common_params_speculative_draft` and `common/arg.cpp`, kept separate from the existing `n_min` field (do not overload it).

## Steps

1. Read the full `common_speculative_impl_draft_mtp` struct (common/speculative.cpp, starts line 1330) end-to-end, specifically wherever `n_max`/`params.n_max` is consulted per-draft-step (the actual depth-limiting loop, not just the ctor assert) -- this is the real per-sequence hook point for `bigcherry_nro06_adaptive_mtp::n_cur`.
2. Grep for sequence lifecycle hooks: `git -C work/upstream/llama.cpp.git grep -n 'seq_id\|reset\|rewind\|retry' b11126 -- common/speculative.cpp | head -40` to find where per-sequence draft state is created/reset/torn down (needed for `bigcherry_nro06_adaptive_mtp::reset()` call sites on request/sequence/context-rewind/failure/retry/recreation).
3. Add `n_min_adaptive` (floor) and `n_max` (cap, reuse existing) to `common_params_speculative_draft` with explicit range validation (floor <= cap, floor >= 1) in the CLI/config parsing path (common/arg.cpp or common/common.h wherever `common_params_speculative_draft` fields are populated -- grep for existing `n_max`/`n_min` CLI flags there to match convention).
4. Add a `std::vector<bigcherry_nro06_adaptive_mtp> adaptive_state` (per-seq, sized like the existing `pending_h`/`verify_h` vectors in the struct) to `common_speculative_impl_draft_mtp`, with `reset()` called at every point identified in step 2, and `update(n_draft, n_accepted, n_max, n_min_adaptive)` called after each verify step using the real accepted-count value already computed by the struct (grep for where accepted-token count is computed in the verify path).
5. Instrument requested/accepted depth transitions to a trace buffer or log line, and build an offline acceptance-trace replay harness (a small standalone tool under tools/lab/ that replays a recorded trace through the controller state machine in isolation, for the exhaustive transition tests -- no llama.cpp runtime needed for this part).
6. Add exhaustive unit tests for the controller itself (already testable in isolation per its 'pure/testable' design comment): floor/cap boundary, full-accept climb, miss-pressure drop, floor non-accumulation, multi-sequence independence, malformed config rejection.
7. Add deterministic parity tests: temp-0 output must be IDENTICAL between fixed-depth MTP and adaptive MTP (adaptive changes speculative work, never final-token semantics) -- this is the critical correctness gate before any promotion.
8. Compare adaptive vs best fixed-depth control over prose/code/repetitive/reasoning content at multiple contexts: acceptance rate, rejected work, target/draft latency, total TPS.

1. Read `common_speculative_impl_draft_mtp` in full (common/speculative.cpp) and confirm the exact real text of: the `pending_h` member declaration (insertion point for adaptive state), `begin()` (reset call site), the `if (params.n_max <= (int) result.size())` comparison (depth-limit replacement site), and `accept()` (update() call site, and where the accepted-count value is already computed).
2. Add a per-sequence `std::vector<bigcherry_nro06_adaptive_mtp> adaptive_state` member declared immediately beside `pending_h`.
3. In `begin()`, call `adaptive_state[seq_id].reset(n_max, n_min_adaptive)` alongside existing per-sequence setup.
4. Replace `if (params.n_max <= (int) result.size())` with an adaptive effective-depth selection that consults `adaptive_state[seq_id].n_cur` when adaptive mode is enabled (env/config gated), else preserves the exact original fixed comparison.
5. Record per-seq drafted count (`last_n_draft[seq_id] = result.size()` or equivalent, at the real site where the draft count is finalized).
6. In `accept()`, call `adaptive_state[seq_id].update(last_n_draft[seq_id], n_accepted, n_max, n_min_adaptive)` using the accepted-count value `accept()` already computes -- do not recompute it separately.
7. Add `n_min_adaptive` as a new, explicitly-disabled-by-default field to `common_params_speculative_draft` (common/common.h or wherever the struct lives) and its CLI flag in `common/arg.cpp`, following the existing `n_max`/`n_min` flag convention but as a SEPARATE field/flag, never overloading `n_min`.
8. Add exhaustive unit tests for the controller itself (isolable per its pure/testable design): floor/cap boundary, full-accept climb, miss-pressure drop, floor non-accumulation, multi-sequence independence, malformed config rejection.
9. Add deterministic parity tests: temp-0 output must be IDENTICAL between fixed-depth MTP and adaptive MTP when adaptive mode is disabled (default); with adaptive enabled, only speculative work changes, never final-token semantics.
10. Compare adaptive vs best fixed-depth control over prose/code/repetitive/reasoning content at multiple contexts: acceptance rate, rejected work, target/draft latency, total TPS (hardware, Brutus, not run here).

## Detailed Solution & Technical Design

Second BigCherry patch, `requires = ["1255_nro06_adaptive_mtp_depth"]`, that wires the already-authored controller into `common_speculative_impl_draft_mtp`'s per-sequence draft-depth decisions and adds CLI/config surface. The controller itself (climb/drop semantics, floor/cap) is treated as an experimental hypothesis per the item's own standards -- this wiring patch must not silently change fixed-MTP behavior when adaptive mode is not explicitly selected.

## Code Samples & Guidance

Real anchor (verified, patches/1255_nro06_adaptive_mtp_depth/patch.py): the controller struct is inserted immediately before this line in common/speculative.cpp:
```cpp
struct common_speculative_impl_draft_mtp : public common_speculative_impl {
```
and exposes:
```cpp
struct bigcherry_nro06_adaptive_mtp {
    int n_cur = 0, n_climb = 0, n_drop = 0;
    void reset(int n_max, int n_min_adaptive);
    void update(int n_draft, int n_accepted, int n_max, int n_min_adaptive);
};
```
New patch patches/1256_prbe52_nro06_adaptive_mtp_wiring/patch.toml:
```toml
schema = 1
id = "1256_prbe52_nro06_adaptive_mtp_wiring"
order = 1256
state = "untested"
kind = "enhancement"
origin = "local"
backend = "agnostic"
plan-ids = ["PRBE52"]
requires = ["1255_nro06_adaptive_mtp_depth"]
conflicts = []
subsystems = ["speculative-decoding", "mtp"]
hardware = []
validation-architectures = []
backends = []
```
patch.py skeleton (anchors for the per-sequence hook, sequence-reset points, and CLI flags are TODO-VERIFY pending the step-1/2/3 full reads -- the struct's exact per-draft-step consumption of n_max and its sequence-reset call sites were not pasted into this plan):
```python
from bigcherry.patcher import Edit, FilePatch

PATCHES = [
    FilePatch(
        path="common/speculative.cpp",
        description="PRBE52 wire bigcherry_nro06_adaptive_mtp into common_speculative_impl_draft_mtp",
        edits=(
            Edit(
                id="prbe52-adaptive-state-member",
                anchor=r"<TODO-VERIFY: real text of pending_h/verify_h member declarations in the struct>",
                rationale="add per-sequence adaptive controller state alongside existing per-sequence vectors",
                mode="insert_after",
                text=r"<TODO-VERIFY: std::vector<bigcherry_nro06_adaptive_mtp> adaptive_state; plus adaptive_enabled/n_min_adaptive params>",
                guard=r"adaptive_state",
            ),
            Edit(
                id="prbe52-reset-hook",
                anchor=r"<TODO-VERIFY: real sequence-reset function body from step 2>",
                rationale="reset per-sequence controller on request/sequence/context-rewind/failure/retry/recreation",
                mode="replace",
                text=r"<TODO-VERIFY: old body + adaptive_state[seq_id].reset(...) call>",
                guard=r"BIGCHERRY_PATCH_HIT patch=prbe52.*reset",
            ),
            Edit(
                id="prbe52-depth-and-update-hook",
                anchor=r"<TODO-VERIFY: real per-draft-step n_max consumption + accepted-count computation from step 1>",
                rationale="use adaptive_state[seq_id].n_cur as the effective draft depth when adaptive mode is enabled; call update() after verify with the real accepted count",
                mode="replace",
                text=r"<TODO-VERIFY>",
                guard=r"BIGCHERRY_PATCH_HIT patch=prbe52.*depth",
            ),
        ),
    ),
]
```

## Files

common/speculative.cpp; common/arg.cpp or common/common.h (CLI/config for n_min_adaptive); tools/lab/mtp-adaptive-replay/ (new, offline acceptance-trace replay harness); patches/1256_prbe52_nro06_adaptive_mtp_wiring/{patch.toml,patch.py,SUMMARY.md}

## Validation

Offline: `PYTHONPATH=tools python -m bigcherry patch-lint`, `patch-rebase-check --focal-overlay 1256_prbe52_nro06_adaptive_mtp_wiring --source bigcherry-tuning`. Controller unit tests (already isolable, per patch.py's own 'pure/testable' framing): exhaustive floor/cap/climb/drop/reset-boundary and multi-sequence-independence cases, malformed-config rejection. Deterministic parity: temp-0 identical token IDs fixed vs adaptive. Hardware (Brutus, not run here): heterogeneous prose/code/repetitive/reasoning content at multiple contexts, acceptance/rejected-work/latency/TPS vs best fixed-depth control, via `python -m bigcherry.patch.validation_campaign`.

## Effort & Risk

L / medium-high -- correctness-sensitive (deterministic parity is a hard gate), but the controller state machine itself is already authored and isolable; risk concentrated in the sequence-lifecycle reset wiring (step 2/4).

## Standards

Backend-neutral policy; source constants are hypotheses; no outcome-conditioned pair deletion; final-token correctness and work accounting required.

## Acceptance Criteria

All consolidated RD62/NRO06 requirements are explicit: state machine, reset boundaries, heterogeneous qualification, deterministic correctness, fixed-depth controls, instrumentation, and anti-overfitting. Fixed draft-mtp behavior remains unchanged; adaptive promotion requires improved results versus the best relevant fixed control across the pre-registered workload mix.

## Notes

Supersedes: RD62
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd62

Supersedes: RD62; consolidates duplicate NRO06 scope.
Inherited semantic scope: carry forward RD62's detailed state machine and NRO06's richer design/qualification constraints; historical source evidence remains on completed predecessors.
Migration: capability-rebaseline-v3-2026-09

2026-09-24 relevance at b11126: existing untested patch 1255_nro06_adaptive_mtp_depth confirmed to add the controller struct only, explicitly unwired per its own patch.py docstring; common_speculative_impl_draft_mtp (common/speculative.cpp:1330) confirmed to read params.n_max/n_min as the natural depth hook. GPT design request: not separately obtained for PRBE52 in this pass (batched request submitted for PRBE52+PRBE59 was queued but did not return in time; this plan was written directly from source evidence -- patches/1255's own patch.py plus common/speculative.cpp struct layout -- and should be cross-checked against the GPT response if/when it lands).

2026-09-24 GPT review req_2b65d50ebe9547fd applied: NOT-READY -- replaced TODO-VERIFY placeholder anchors with GPT-supplied concrete ones (pending_h-adjacent state, begin() reset, `if (params.n_max <= (int) result.size())` replacement, accept()-site update() call); added explicit separate n_min_adaptive config field.

## Change Log

- 2026-09-09T10:57:05.575180+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:14:19.351820+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.363199+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.151518+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:10:18.602333+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes
- chg_20260910_021313_the-semantic-audit-is-now-trac_4827
- 2026-09-10T02:13:13.335804+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:36:51.577194+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
- 2026-09-24T04:42:51.022605+00:00 (updated-by): Updated: section:description, section:steps
- 2026-09-24T04:43:00.189120+00:00 (updated-by): Updated: section:notes
