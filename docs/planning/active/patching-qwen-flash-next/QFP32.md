---
id: QFP32
order: 32
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-07T00:39:26.596251+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: S
---

# MTP re-reserve: no full realloc + sync per prefill chunk when MTP outputs turn on

## Description

External report: re-reserving the graph once when MTP outputs turn on ended a full GPU realloc+sync per prefill chunk (88.5K prefill 1042 -> 1130 t/s). Treat that number as a hypothesis from different hardware.

At llama.cpp b11402 (`d89651a7b205`) the same structural hazard exists. `llama_context` performs its initial `sched_reserve()` in the context constructor while `cparams.embeddings_nextn == false`. Later, `common_speculative_impl_draft_mtp` enables NextN output on both target and draft contexts with `llama_set_embeddings_nextn(...)`. Unlike `set_embeddings_layer_inp()`, `llama_context::set_embeddings_nextn()` changes graph/output shape state without setting `sched_need_reserve`.

For Qwen4Exp this is not a metadata-only toggle: `src/models/qwen4exp.cpp::llama_model_qwen4exp::graph::build` conditionally exposes `t_h_nextn`, suppresses the last-layer output crop for unmasked target NextN, and performs the crop later. The draft MTP graph also changes `t_h_nextn` layout according to `embeddings_nextn_masked`.

First gate: prove repeated scheduler/gallocr replans actually occur on our production build. Patch 1340 already exposes replan reasons and reserve counts. If a 98K MTP prefill shows no repeated reserve/replan attributable to this toggle, close QFP32 with the measurement and write no patch.

## What we already have

### b11402

- `common/speculative.cpp::common_speculative_impl_draft_mtp::common_speculative_impl_draft_mtp`
  - creates the MTP driver after target/draft contexts already exist;
  - calls `llama_set_embeddings_nextn(ctx_tgt, true, false)`;
  - calls `llama_set_embeddings_nextn(ctx_dft, true, true)`.
- `src/llama-context.cpp::llama_context::llama_context`
  - initializes `cparams.embeddings_nextn = false`;
  - allocates the initial output buffer;
  - calls `sched_reserve()` before MTP later changes the output mode.
- `src/llama-context.cpp::llama_context::set_embeddings_nextn`
  - currently only assigns `cparams.embeddings_nextn` and `cparams.embeddings_nextn_masked`;
  - does **not** set `sched_need_reserve`.
- `src/llama-context.cpp::llama_context::set_embeddings_layer_inp`
  - is the useful precedent: changing a graph-visible extraction mode sets `sched_need_reserve = true`.
- `src/llama-context.cpp::llama_context::sched_reserve`
  - synchronizes once, rebuilds the scheduler, reserves worst-case prompt-processing and token-generation graphs, then clears the reserve requirement.
- `src/llama-context.cpp::llama_context::process`
  - calls `sched_reserve()` before processing the logical batch, so a reserve request made by the setter is naturally consumed once before the first MTP-enabled batch.
- `src/llama-context.cpp::llama_context::output_reserve`
  - separately grows the host output buffer; with unmasked NextN it sizes `embd_nextn` by `n_batch`, not selected outputs.
- `src/models/qwen4exp.cpp::llama_model_qwen4exp::graph::build`
  - lines around the final block conditionally keep all rows while unmasked NextN is active and build `res->t_h_nextn`.
- `src/models/qwen4exp.cpp::llama_model_qwen4exp::graph_mtp::graph_mtp`
  - returns masked/unmasked `t_h_nextn` according to the context setting.

### BigCherry overlap

- `patches/1339_meta_memory_report`: `BIGCHERRY_META_MEM=1` reports physical Meta compute/static buffers.
- `patches/1340_meta_per_device_arena`:
  - `ggml_backend_meta_alloc_graph()` records reserve activity;
  - `bc_gallocr_replan_reasons[]` identifies node-count, leaf-count, external-now-needed and larger-than-planned replans;
  - `BIGCHERRY_META_MEM arena_time` reports `reserve_calls`.
- `patches/1317_spec_round_timing` and `1318_mtp_draft_timing` measure speculative round/draft cost, but do not measure prompt-time scheduler re-reserves.

This optimisation is **not already covered**. 1340 measures/reduces Meta arena overhead but does not make `set_embeddings_nextn()` reserve the post-toggle graph.

## Steps

1. Baseline one 24K and one ~98K prompt with MTP enabled and `BIGCHERRY_META_MEM=1`; record logical scheduler reserve/replan evidence and 1340 per-device `reserve_calls`.
2. If repeated replans are absent, close QFP32 as not applicable at this pin/configuration.
3. Otherwise implement the one-time reserve request in a new patch package, flag default off.
4. Verify exactly one reserve is requested when each context changes NextN mode and no reserve is requested for an idempotent setter call.
5. Run offline mechanics, patch-lint, activation evidence, then fully separated ABBA at 24K and ~98K.

## Detailed Solution & Technical Design

### Qualification mechanism

Create `patches/1343_mtp_nextn_rereserve` with qualification flag:

`BIGCHERRY_MTP_RERESERVE=0|1`, default `0`.

Patch `src/llama-context.cpp::llama_context::set_embeddings_nextn(bool value, bool masked)`.

Under the flag:

1. Compare the requested `value` and `masked` with the existing `cparams` fields.
2. Return without touching reserve state if both are unchanged.
3. Assign the new fields.
4. Set `sched_need_reserve = true`.

Do **not** call `synchronize()` or `sched_reserve()` directly from the setter. The next `llama_context::process()` already calls `sched_reserve()` at the correct serialized boundary. This makes the cost one explicit pre-batch reserve instead of introducing a new lifecycle/API seam.

The target and draft contexts are independent. The MTP constructor toggles both, so each context can reserve its own post-toggle worst-case graph once. No fixed GPU count, device id, tensor split, attention split, or model size is encoded.

### Why the setter is the right owner

The shape-changing state belongs to `llama_context`, not the speculative driver. Fixing only `common_speculative_impl_draft_mtp` would leave EAGLE/DFlash/other callers of `llama_set_embeddings_nextn` exposed to the same late-toggle problem.

The setter is also symmetric with `set_embeddings_layer_inp()`, which already marks the scheduler dirty for a graph-visible extraction change.

### Host output buffer

Do not mix host-output preallocation into v1. `output_reserve()` is a separate buffer mechanism and should normally grow at most once after NextN is enabled. If instrumentation shows repeated `output_reserve()` reallocations, record that as a separate defect before extending this item.

### Meta backend interaction

The change is above the Meta backend. `sched_reserve()` builds the graph using the current context parameters and then `ggml_backend_sched_reserve()`/1340 reserve the transformed per-device graphs. No change is made to split state, device placement, collective ordering, or per-device arena ownership.

With `BIGCHERRY_META_PER_DEVICE_ARENA=1`, the desired evidence is that the initial MTP-enabled reserve creates/reuses the correct device plans and subsequent prompt chunks do not increment reserve/replan counters. With that flag off, the ordinary common scheduler arena should show the same one-time behavior.

### Architectures and fusion

This is architecture-neutral: gfx1100, gfx1201 and gfx1030 all use the same host scheduler path. It must not alter CUDA/HIP kernels, fusion eligibility, QSA, MoE expert splitting, or MTP arithmetic.

## Code Samples & Guidance

Likely anchor in `src/llama-context.cpp`:

```cpp
void llama_context::set_embeddings_nextn(bool value, bool masked) {
    LLAMA_LOG_DEBUG("%s: value = %d, masked = %d\n", __func__, value, masked);

    cparams.embeddings_nextn        = value;
    cparams.embeddings_nextn_masked = masked;
}
```

Qualification logic should preserve this source exactly when the flag is off. Do not add a legacy compatibility path.

Add one low-volume activation marker, emitted only when the setter changes state and marks the scheduler dirty, for example:

`BIGCHERRY_PATCH_HIT 1343 mtp_nextn_rereserve value=<0|1> masked=<0|1>`.

Do not emit per ubatch.

## Files

If Gate 0 proves the problem:

- create `patches/1343_mtp_nextn_rereserve/patch.py`;
- create/update its normal `README.md` and `SUMMARY.md`;
- add `tools/tests/patch/test_1343_mtp_nextn_rereserve.py`;
- production target patched by that package: `src/llama-context.cpp`;
- use existing `patches/1339_meta_memory_report` and `1340_meta_per_device_arena` only as diagnostics; do not change them;
- use `tools/lab/flash-next/queue-env-ab.sh` for ABBA.

If Gate 0 is negative, create no patch package and close this item with evidence.

## Validation

### Offline

- Patch mechanics/idempotence test for the exact b11402 setter anchor.
- Flag off leaves setter behavior source-equivalent.
- Flag on marks reserve only on a real `value`/`masked` transition.
- Repeating the same setter call does not request another reserve.
- Patch-lint and normal patch composition.

### Activation evidence

For one MTP-enabled prompt:

- exactly one `BIGCHERRY_PATCH_HIT 1343 ...` per target/draft context transition;
- scheduler reserve/replan counters stop increasing after the first MTP-enabled reserve;
- with 1340 enabled, `BIGCHERRY_META_MEM arena_time reserve_calls` must not scale with prefill chunk count.

### Hardware ABBA

Use `tools/lab/flash-next/queue-env-ab.sh` with complete process separation.

A: `BIGCHERRY_MTP_RERESERVE=0`
B: `BIGCHERRY_MTP_RERESERVE=1`

Keep all other production flags identical. Run 24K then ~98K prompt lanes; ub512 first. Record prompt t/s, total prefill wall time, replan/reserve counts, MTP acceptance and decode t/s after the prompt.

This is allocation/scheduling only, so require greedy target-output identity. If output differs, reject regardless of speed.

No multi-session contract campaign is required.

## Effort & Risk

Effort: S.

Risk: low-to-medium. The code change is tiny, but scheduler reserve rebuilds graph state and Meta 1340 currently has unqualified allocator work. Main risks are an unnecessary reserve on callers that toggle NextN repeatedly, extra startup latency, or composition with per-device arena plans.

Mitigation: transition check, default-off qualification, one-shot activation marker, replan counters and full-process ABBA.

Expected gain on our three-card topology: potentially medium for long prefills **only if** repeated replans exist. If Gate 0 is negative, expected gain is zero and the item should close immediately.

## Standards

- No fixed topology or device numbering.
- No q4 KV.
- No legacy/back-compat shim.
- One mechanism owner: this item only fixes late NextN scheduler reservation.
- 1339/1340 remain diagnostic/allocation owners.
- Default off during qualification.
- Prefer a one-time reserve over any per-chunk special case.
- External 4-GPU numbers are hypotheses, not promotion evidence.

## Acceptance Criteria

- Gate 0 proves repeated post-NextN reserve/reallocation on our MTP prefill, or the item is closed as not applicable.
- Flag-off behavior is unchanged.
- Flag-on requests one scheduler reserve on a real NextN mode transition and none for idempotent calls.
- Activation evidence shows the post-toggle reserve occurs before the first MTP-enabled batch.
- Replan/reserve counts no longer scale with prompt chunks.
- Greedy target output is identical.
- 24K/~98K fully separated ABBA shows no repeatable prefill/decode/acceptance regression.
- No Meta split, fusion, kernel or collective semantics change.

## Notes

Ordering: first among QFP31-QFP41. It is a cheap, architecture-neutral check with a plausible long-prompt payoff and excellent existing diagnostics.

## Change Log

- 2026-10-07T00:39:26.596251+00:00 (created-by): Created by agent
- 2026-10-07: grounded at b11402 and BigCherry head 83b692ee; added setter-level one-time reserve design, 1340 replan Gate 0, qualification flag and lightweight ABBA.
