---
id: QFP33
order: 33
plan: patching-qwen-flash-next
state: deprecated
created-at: '2026-10-07T00:39:30.097678+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: S
---

# Draft ubatch cap: smaller draft compute buffer

## Description

External report: capping the draft physical ubatch reduced its compute buffer from 457 to 247 MiB. Treat that result as a hypothesis from different hardware.

At b11402 the draft/MTP context inherits the target's global `n_batch` and `n_ubatch` through `common_base_params_to_speculative()`. Our production target wants `-ub 512` for target prefill, but the MTP draft usually executes tiny generation batches (depth 3) and, without QFP31, prompt catch-up batches that can be internally split. There is no draft-specific ubatch knob at this pin.

This is primarily a VRAM-footprint experiment, not a presumed throughput optimization. On the RX 6900 XT sidecar it may recover useful headroom; on built-in MTP using target devices it may reduce each draft-context compute arena. A lower ubatch can also make prompt catch-up slower, so qualification must measure both memory and end-to-end latency.

## Steps

1. Record the current draft-context compute-buffer size with target `-ub 512` for both production Qwen4Exp sidecar MTP and the 27B built-in-MTP case.
2. Add a draft-MTP-only physical ubatch cap, default off.
3. Test caps 256, 128 and 64 only as qualification points; do not hard-code one as optimal.
4. Confirm large prompt catch-up is correctly split and draft generation semantics are unchanged.
5. Run memory + fully separated ABBA. Close if memory saved is negligible or prompt/draft latency regresses enough to dominate.

## Detailed Solution & Technical Design

### Placement and dispatch

Implement in `common/speculative.cpp::common_base_params_to_speculative()`, after copying base params and only when `COMMON_SPECULATIVE_TYPE_DRAFT_MTP` is enabled.

Qualification flag:

`BIGCHERRY_MTP_DRAFT_UBATCH=<tokens>`, default unset/0 = current behavior.

When enabled:

`result.n_ubatch = min(result.n_ubatch, requested_cap)`.

Do not change `result.n_batch`. The logical draft batch may remain larger; normal context processing splits it to the smaller physical ubatch. Do not alter the target context or global `params.n_ubatch`.

Reject nonsensical values during qualification rather than silently inventing topology-dependent defaults. Start with >=32 because the common parameter contract documents physical prompt batches below 32 as outside the normal BLAS-oriented regime.

### Why this owner

The cap is a property of constructing the speculative context, not of Qwen4Exp kernels or Meta. `common_base_params_to_speculative()` is the single existing point where target parameters are converted into draft parameters and already contains draft-specific device/model overrides.

Do not patch `llama_context::sched_reserve()`: that would make a generic context report one ubatch while reserving another and would break the allocator contract.

### Memory and data layout

No tensor format changes. A lower draft ubatch reduces the maximum token dimension of temporary graph tensors reserved by `sched_reserve()`; KV capacity remains tied to draft `n_ctx` and is unchanged.

For the single-device ROCm3 sidecar, this should reduce the normal HIP draft compute arena. For a built-in MTP context using a Meta tensor split, the logical draft graph is smaller and 1339/1340 can expose per-device physical arena changes.

### QFP31 interaction

Qualify QFP33 with QFP31 off first. If QFP31 later replays a 2048-token tail, that replay must be chunked using `llama_n_ubatch(ctx_dft)`; QFP31 must not assume its configured window fits one physical draft batch.

### Architectures and fusion

The change is host-side and architecture-neutral across gfx1030/gfx1100/gfx1201. Kernel selection/fusion sees the actual smaller batch shapes and otherwise remains unchanged. Do not add architecture branches.

Potential performance tradeoff: a 512-token MTP catch-up that currently executes as one physical ubatch may become 2/4/8 ubatches. That is the primary risk and must be included in the long-prompt ABBA.

## Code Samples & Guidance

Likely anchor in `common/speculative.cpp`:

```cpp
common_params common_base_params_to_speculative(const common_params & params) {
    const bool has_draft = params.speculative.has_dft();

    const auto & params_spec = params.speculative.draft;
    common_params result = params;
```

and the existing tail:

```cpp
result.cache_type_k  = params_spec.cache_type_k;
result.cache_type_v  = params_spec.cache_type_v;
result.n_outputs_max = params.n_parallel;
```

Keep flag parsing local to this conversion path. Emit one startup activation line, e.g.:

`BIGCHERRY_PATCH_HIT 1345 mtp_draft_ubatch inherited=512 effective=128`.

No per-batch logging.

## Files

New mechanism:

- create `patches/1345_mtp_draft_ubatch/patch.py`;
- create its `README.md` and `SUMMARY.md`;
- add `tools/tests/patch/test_1345_mtp_draft_ubatch.py`;
- production file patched: `common/speculative.cpp`.

Use 1339/1340 only for Meta memory evidence; do not move this mechanism into those patches.

## Validation

### Offline

- Exact/idempotent patch mechanics and patch-lint.
- Flag unset/0 preserves inherited draft `n_ubatch`.
- Cap greater than inherited value does not enlarge it.
- Cap 128 with target `-ub 512` produces target 512 / draft 128.
- Logical draft batch > cap is accepted and physically split by normal context processing.
- No change to `n_batch`, `n_ctx`, KV type or output limits.

### Activation/memory evidence

- One `BIGCHERRY_PATCH_HIT 1345 ...` at draft-context construction.
- Capture draft compute-buffer MiB from startup logging.
- If draft uses Meta, also capture `BIGCHERRY_META_MEM compute/arena` per device.
- Show that only draft-context compute memory changes.

### Hardware ABBA

Via `tools/lab/flash-next/queue-env-ab.sh`, complete process separation.

A: cap off.
B: one selected cap after the 256/128/64 sweep.

Use 8K and 24K first; include ~98K if QFP31 is off and prompt catch-up remains full. Record VRAM, prompt t/s, time-to-first-token, draft timing, acceptance and steady decode t/s.

This mechanism should be numerically equivalent: require greedy target identity and no acceptance change attributable to arithmetic. Small timing-driven scheduling differences are not a valid reason for output divergence.

No multi-session contract campaign.

## Effort & Risk

Effort: S.

Risk: low for correctness, medium for performance. The main regression mode is extra draft prompt-catch-up chunks/launches. Memory savings may be valuable even if steady decode is neutral.

Expected gain on our topology: low direct speed gain, potentially useful VRAM recovery on ROCm3 or the two-XTX built-in-MTP case. Prioritize after QFP32/QFP31 because it is cheap and can enable later kernels/contexts to fit without touching target `-ub 512`.

## Standards

- Draft-specific only; target `-ub` is untouched.
- No fixed device count or attention split.
- No q4 KV.
- No implicit architecture default.
- No legacy alias/shim.
- Default off while qualifying.
- External memory number is a hypothesis, not an acceptance threshold.

## Acceptance Criteria

- Draft and target effective ubatch values are independently evidenced.
- Draft compute memory decreases materially for at least one production MTP configuration.
- Large draft logical batches split safely at the cap.
- Greedy target output is identical.
- Acceptance is unchanged within run noise.
- Fully separated ABBA shows no material end-to-end regression for the chosen cap.
- If no useful memory saving exists, close without promotion.

## Notes

Ordering: third overall, after QFP32 and QFP31. It is cheap and low-risk, but primarily a memory optimization; QFP32/QFP31 have larger plausible long-prompt latency upside.

2026-10-07 REJECTED for the production topology. Production load (metamem-mp24, ctx 245760): the draft context's compute buffer is 616.31 MiB and it lives on the RX 6900 XT (ROCm3), which holds 4.37 GB of 16 GB in total with the draft model, its KV (120 + 480 MiB) and that buffer - about 12 GB free. A smaller draft ubatch would return ~300 MiB on a card that has no use for it, and the draft's prompt catch-up would run in more, smaller chunks, which is the opposite of what QFP31 needs. It could matter only where the draft shares a target card (the 27B's built-in MTP on the two XTXs); reopen there if that layout runs short of memory.

## What we already have

### b11402

- `common/speculative.cpp::common_base_params_to_speculative(const common_params &)`
  - copies the complete base `common_params`, therefore inheriting `n_batch` and `n_ubatch`;
  - then overrides draft devices/model/KV/output limits, but not batch sizes.
- `common/speculative.cpp::common_speculative_init_result::common_speculative_init_result`
  - converts those draft params with `common_context_params_to_llama()`;
  - creates the MTP context with `LLAMA_CONTEXT_TYPE_MTP`;
  - sets draft `n_ctx` equal to target `n_ctx`.
- `common/common.cpp::common_context_params_to_llama`
  - maps `common_params.n_batch/n_ubatch` directly to `llama_context_params`.
- `src/llama-context.cpp::llama_context::sched_reserve`
  - reserves its worst-case graph with `n_tokens = min(n_ctx, n_ubatch)`; therefore draft `n_ubatch` directly affects the reserved compute graph/arena.
- `src/llama-context.cpp::llama_context::process`
  - uses the context's own physical ubatch when splitting a logical batch, so lowering only draft `n_ubatch` does not require lowering target `-ub`.
- `common/speculative.cpp::common_speculative_impl_draft_mtp::draft`
  - generation batches are normally only active sequences / draft depth, far below 512.
- `common/speculative.cpp::common_speculative_impl_draft_mtp::process`
  - prompt catch-up may submit a much larger logical batch; the draft context must safely split it when its physical ubatch is capped.

The existing global `-ub/--ubatch-size` is **not** an adequate solution because it also caps target prefill and would confound the production benchmark.

### BigCherry overlap

- `patches/1339_meta_memory_report` can show per-device Meta compute arena sizes when the draft context itself uses Meta.
- `patches/1340_meta_per_device_arena` changes how Meta compute storage is physically reserved but does not change the draft graph's `n_ubatch`.
- QFP31, if implemented, reduces how many prompt tokens the draft replays; it is complementary. QFP33 still changes reserve size and should be measured independently first.

This idea is **not currently covered**.

## Change Log

- 2026-10-07T00:39:30.097678+00:00 (created-by): Created by agent
- 2026-10-07: grounded at b11402; identified inherited target ubatch as the mechanism, added draft-only cap design, Meta/single-device memory evidence and separated ABBA.
- 2026-10-07T09:29:48.796311+00:00 (updated-by): Updated: section:notes
- 2026-10-07T09:29:52.353140+00:00 (state-transition): State: pending → deprecated
