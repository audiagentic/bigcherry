---
id: PNRO11
order: 0
plan: patching-nasone-rdna-optimizations
state: done
created-at: '2026-09-09T10:52:56.925530+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P1
---

# DFlash/DSpark shared lm_head: retire target-wide replication

## Terminal disposition — 2026-10-12

**Close the proposed `1264_pnro11_lmhead_replication_tp` without implementation.** The original design would mirror the target's entire `output.weight` (and bias/scale) across every tensor-split target rank, affecting normal target/MTP execution for the sake of a single draft consumer. Existing **untested** BigCherry patch `1286_draft_local_shared_tensors` instead copies only *borrowed* tensors in a Meta buffer to the **single-device draft's** own backend once at draft-context construction. QFP43 owns this compatibility mechanism and any remaining correctness fixes. PNRO11 has no separate patch, runtime selector, allocator, scheduler, CLI flag or hardware queue. Historical September implementation sketches are superseded; consult Git history for them.

Do **not** infer that 1286 is production-qualified or that every DFlash tensor-split failure is solved. Separate the two source-level failure modes:

1. **Foreign Meta-buffer ownership:** the single-device draft graph borrows `tok_embd`, `output.weight`, and optionally `output_s` from a tensor-split target. The draft scheduler cannot necessarily execute a preallocated tensor in the target's `Meta()` buffer. Patch 1286's draft-local copies address this ownership/placement mismatch without changing the target's split axis. This is the relevant PNRO11 remainder.
2. **Split-vocabulary TOP_K:** if the draft itself executes through a tensor-split Meta backend, target `output.weight` with `SPLIT_AXIS_1` produces axis-0 split logits; the in-graph DFlash2 `GGML_OP_TOP_K` cannot consume that representation. Patch 1286 is gated to `!model.devices[0].is_meta` and does **not** solve this separate Meta-selector case. Upstream draft PR #27858 proposes CPU-side selector ranking from gathered logits and keeping the target head sharded, but is **open/unmerged** as checked 2026-10-12. No BigCherry adoption without correctness and host-transfer cost evidence.

## Exact source path, allocation and lifetime

Pinned llama.cpp **b11474** `src/llama-model.cpp` assigns `pattern_output_weight` to `GGML_BACKEND_SPLIT_AXIS_1` unless `is_dsv4`; `pattern_output_bias` independently selects `SPLIT_AXIS_0`. Current upstream master retains this selection. Do not assume bias follows weight's mirrored state. `common/speculative.cpp::common_speculative_impl_draft_dflash` constructs only after `ctx_tgt` and `ctx_dft` exist; discovering the draft's selector metadata there is too late to change target-model load placement. `src/models/dflash.cpp` allows draft-owned `tok_embd`/`output`, otherwise binds the target's tensors through `ctx_other`.

BigCherry `1286_draft_local_shared_tensors/patch.py` edits `src/llama-cparams.h`, `src/llama-context.{h,cpp}`, and `src/models/dflash.cpp`. During draft-context construction it inspects each borrowed tensor's buffer device, creates a `ggml_dup_tensor` in a draft-local `ggml_context` for Meta-resident tensors, allocates one buffer from `ggml_backend_dev_buffer_type(model.devices[0].dev)`, gathers via `ggml_backend_tensor_get` into temporary host bytes, then uploads via `ggml_backend_tensor_set`. `bc_other_ctx`/`bc_other_buf` own the GPU copies for the draft-context lifetime; the temporary `std::vector<uint8_t>` is released after construction. This is **synchronous setup**, not an asynchronous per-token cache or a GPU P2P transfer. The draft-owned-output case is unchanged; layer-split/single-device target buffers are not copied by this Meta-only path.

**Source-level conditional defect in the untested patch:** the two DFlash graph builders choose
`output_s = cparams.other_output ? cparams.other_output_s : model_other->output_s`.
The condition tests the **weight copy**, not the **scale copy**. If weight is Meta and a non-Meta scale exists, it chooses a null scale; if weight is non-Meta but scale is Meta, it ignores the scale copy and uses the original foreign tensor. A 12-case deterministic host truth table over owned/borrowed, Meta-weight, scale-present and Meta-scale configurations found **2 pointer-selection mismatches**. This proves conditional inconsistency, **not** that either mixed-residency case occurs in the deployed GGUF or causes a GPU failure. Fix/qualify this in QFP43/1286 only: choose each tensor independently, prove draft-backend readability for uncopied tensors, and fail closed rather than silently dropping a required scale.

**Memory/traffic gate:** compute `ggml_nbytes` for each borrowed full tensor and record draft-device free/peak VRAM, temporary host peak, gather/upload bytes, setup time and device order. A single draft-local full head is not free; on no-P2P gfx1100/gfx1201/gfx1030 topology, expect a host-mediated one-time copy, not peer DMA. Do not silently use the target's mirrored memory budget or rely on `--fit` to discover an OOM after allocation. Original target-wide mirroring adds approximately `full_head_bytes - target_shard_bytes` on **every** target rank and may slow unrelated MTP. Upstream issue #28777 reports an **external NVIDIA** matched-session 7–10% MTP step-time penalty from target-head mirroring; it is not a BigCherry/AMD benchmark.

## Existing measured BigCherry evidence and value decision

- 2026-10-04 Qwen3.8-27B Q8_0 dual-XTX target: built-in MTP **72.7/73.1 t/s at 10K**, **71.8/72.0 t/s at 32K**, versus no-spec **37.5/36.5 t/s**. Tensor-split DFlash/DSpark had context-creation/scheduler failures; those runs did not establish a head-replication speedup.
- QFP43's 2026-10-08 Brutus five-start series with production + **untested 1286**, 1665-token prompt, 128 greedy tokens: MTP depth 4 **80.2 t/s, 67.3% acceptance**; DFlash2 Q8_0 block 4 **23.8 t/s, 1.7%**, block 7 **20.7 t/s, 0.7%**; DSpark Q8_0 block 6 **21.9 t/s, 2.0%**. The draft files and feature semantics require further isolation; this is not a clean 1286-versus-baseline performance A/B. **Do not queue a PNRO11 speed campaign while acceptance is collapsed.**
- No PNRO11 implementation, new hardware run, new speedup or production promotion occurred in this audit.

## QFP43-owned bounded residual gate (not a PNRO11 reopening)

1. **Host correctness first:** extend `tools/tests/patch/test_1286_draft_local_shared_tensors.py` beyond string/idempotence checks with a 2×2×2 owned/borrowed and Meta-residency/scale-presence selector table, both graph builders, draft-owned head, absent scale, asymmetric Meta weight/scale, layer-split and Meta-draft negative controls. Require pointer identity, backend readability and complete scale preservation. Fix the conditional only in 1286 if an admissible mixed case exists; otherwise document and enforce the invariant. Run patch-composition tests with 1303 and the current production recipe.
2. **Placement/lifetime:** prove one-time copy after target load and before draft graph reservation; buffer stays alive through graph capture/replay, context reset and repeated same-process requests. Measure actual host/device bytes and peak budgets; reject implicit P2P, stale pointers, wrong-device buffers, missing scale or silent fit failure.
3. **Hardware correctness, only if demand remains:** on dual gfx1100 target plus single-device gfx1030 or gfx1201 draft, compare full-vocab logits/greedy and candidate selection with a same-draft single-GPU/layer-split reference, 10K/32K/98K contexts, multiple ubatches, two requests in one process and graph on/off. Include draft-owned head and ordinary MTP/no-spec controls. DFlash acceptance and feature taps must be correct **before** timing.
4. **Promotion/rejection:** retain 1286 default-off until the exact foreign-Meta-buffer failure is reproduced in the control and disappears in the subject without numerical, MTP, VRAM or graph regression. If acceptance remains near 1–2% or the Flash-Next 2560-wide contracted feature tap is unavailable, close the performance lane; compatibility-only work needs an explicit user requirement. For a Meta draft TOP_K failure, independently compare upstream #27858's CPU selector and distributed-top-k alternatives; do not broaden PNRO11 into target-wide replication.

## Consolidation / dependencies

**QFP43 + patch 1286** own draft-local borrowed-tensor binding, scale coherence and DFlash/DSpark correctness. **QFP42/1348** own deferred target-feature lifetime; **PRBE52** has rejected adaptive MTP depth; **MSM03** owns Meta attention-input residency; **PGC/active Radiance** own collectives/host transport. Do not modify their active implementations or invent a second shared-tensor registry. Upstream #27833 and #28777 distinguish borrowed-buffer and TOP_K failures; #27858 is an unmerged alternative. vLLM's separate `draft_tensor_parallel_size` illustrates keeping draft placement distinct from target TP, not a portable GGUF patch.

## Validation / provenance

Inspected 1286 patch, mechanics test, QFP43 and pinned b11474 `llama-model.cpp`, `speculative.cpp`, `dflash.cpp`; compared current upstream model split selection; inspected upstream #27858 changed-file diff and #27833/#28777. Source-static checks **13/13** after correcting the inspection line range. Deterministic host pointer-selection model: **12 cases, 2 mismatches**. No repository pytest, HIP/C++ build, real model execution or GPU benchmark was run. Last identified independent PNRO11 update 2026-10-04; no PNRO11/1286/DFlash independent implementation, PR or queued lane in the preceding 12 hours. BCOP159 is the disposition ledger.
