---
id: PNRO10
order: 0
plan: patching-nasone-rdna-optimizations
state: pending
created-at: '2026-09-09T10:52:53.458006+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P1
---

# Add ctx_other model devices to speculative scheduler backends

## 2026-10-14 implementation audit — incorrect patch placement; qualification blocked

**Disposition: pending / BLOCKED_SOURCE. Do not promote patch 1261 or claim the current dual-XTX MTP lane activates it.** This is a source-confirmed scheduler-backend registration defect, not a measured throughput regression. The earlier 2026-09-24 note claiming that the insertion was moved outside the ordinary device loop is **not true of the current patch**; retain that note as history only.

### Exact current-pin failure (llama.cpp b11552)

- `engines/llamacpp/patches/1261_nro10_spec_ctx_other_devices/patch.py::PATCHES[0]` inserts after regex `backends\\.emplace_back\\(backend\\);` with `expect_matches=2, occurrence=0`. In `src/llama-context.cpp` b11552, the **first** occurrence is inside `for (const auto & dev : model.devices)` (about lines 339-346), while the second is inside the later ACCEL loop. Consequently the injected `ctx_other` enumeration executes **once per ordinary model device**, before all ordinary devices have been registered.
- **Empty draft device list:** the ordinary loop executes zero times; no foreign `ctx_other` device is added. **Overlapping future device:** draft [XTX0,XTX1], other [XTX1,R9700] yields [XTX0,XTX1,R9700,XTX1] because the first iteration inserts XTX1 before the ordinary second iteration appends it. The existing deduplication compares only backends already initialized; it cannot prevent that duplicate. A one-device disjoint test may pass and hide both failures.
- The `const llama_model *` assignment **is** corrected in the current patch. Do not reopen that resolved compile issue. The `BIGCHERRY_PATCH_HIT patch=1261_nro10 path=ctx_other_backend` marker **exists** under `BIGCHERRY_PATCH_TRACE`, but only when a missing backend is appended; no marker is expected on a same-device MTP control. `validation.toml` and older README/TESTING descriptions claiming unconditional activation or no marker are stale.
- Upstream [llama.cpp #26636](https://github.com/ggml-org/llama.cpp/pull/26636) remains open (last PR update 2026-09-18). Its actual diff inserts **after** the complete ordinary device loop and **before** ACCEL enumeration. Use that placement as a correctness reference, not as evidence of a merged upstream fix. Current master has diverged; recheck anchors against the selected pinned source before implementing.

### Minimal patch-owner correction (no new scheduler or allocator)

In patch 1261 change only the `Edit` placement: anchor on the unique `// add ACCEL backends (such as BLAS)` comment and use `mode="insert_before", expect_matches=1` with the current 8-space indentation; remove `occurrence=0`/the two-match dependence. Keep the existing `ctx_other` null guard, `const llama_model *`, device-handle deduplication, `ggml_backend_dev_init` error propagation, ownership via `backends.emplace_back`, and conditional trace. Ensure ordinary GPU backends precede foreign devices, which precede ACCEL/CPU. Never append a duplicate backend. Do not silently treat Meta-wrapped tensor-buffer reachability as solved by adding an ordinary device handle.

This is **not** a generic MTP speed patch: native tensor-split MTP uses the same model devices, so the missing-device branch does not execute (historical 2026-09-30 Brutus preflight). The actual positive control must use a separate DFlash/DSpark/EAGLE draft on a device set that does not contain all of the target's physical devices. QFP43/1286 owns one-time copies of borrowed Meta-backed tensors for a single-device DFlash draft; it does **not** replace 1261's foreign-device scheduler admission, nor does 1261 replace its copy. PNRO11's target-wide mirroring proposal is closed. Do not modify QFP43 or active Radiance helper/mailbox implementation.

### Bounded validation and terminal gates

1. **Cheapest static/host gate:** compose 1261 against the exact pinned source; assert the injected block occurs once **outside** the ordinary GPU loop and before ACCEL. Source-shaped device-list cases: empty, same-set, subset, superset, disjoint, overlap-with-future-ordinary-device; assert unique device handles, stable ordering, and one `ctx_other` enumeration per context. Inject a `ggml_backend_dev_init` failure and require explicit exception with no retained partial context.
2. **Patch/build gate:** run `patch-lint`, `patch-rebase-check`, the existing offline package tests, and HIP compilation for gfx1100 with 1261 composed against the production patch set. Treat any source-anchor mismatch or duplicate as a hard reject. No such build/test is claimed by this audit.
3. **Hardware correctness gate, only with a real eligible draft:** record target/draft physical-device IDs, buffer owners, `ctx_other` identities, before/after backend ordering, conditional `BIGCHERRY_PATCH_TRACE` marker, scheduler allocation success, host staging/Meta fallback, and repeated same-process requests. Compare greedy text, logits/KLD where available, accepted/proposed counts, and graph replay at short and long contexts. Missing marker on a known foreign-device positive control is **FAIL**, while missing marker on same-device MTP is expected. Never infer P2P from physical PCIe connectivity; gfx1100/gfx1201 and gfx1030 are separate qualification lanes.
4. **Performance gate:** correctness-first. No throughput claim without matched unpatched control, actual executed-route evidence, and independent paired prompt/decode sessions; reject a change that adds duplicate backends, unbounded VRAM, extra transfers or >1% stable non-target regression. If no supported foreign-device draft/target lane exists, terminal `AWAIT_ELIGIBLE_DRAFT` and do not queue hardware experiments. If upstream lands an equivalent validated fix first, compare exact semantics and retire the local duplicate.

**Audit evidence:** 8/8 source-shaped assertions passed against pinned b11552 source + current 1261 patch; four deterministic host topology cases produced the two expected current-code failures (empty, future-ordinary overlap) and two controls (subset, disjoint). Upstream #26636 diff inspected; vLLM/SGLang separate draft-worker/head ownership is a design contrast, not a transferable GGML backend fix. No patch application, repository pytest, HIP build, GPU execution or new benchmark was performed. BCOP193 is the action ledger.


## Description

TODO, NOT-READY (compile bug found, per GPT review). Add ctx_other model devices to speculative scheduler backends -- CORRECTED: patch 1261 does NOT currently compile. Verified via direct read of patch.py: `llama_get_model()` returns `const struct llama_model *` (include/llama.h:584), but patch.py's inserted line assigns it to a non-const `llama_model * other_model = llama_get_model(other);` -- a type error. Additionally, the anchor `backends.emplace_back(backend);` occurrence 0 is inside the ORDINARY model.devices loop, so ctx_other devices are reprocessed per model device (can duplicate later ordinary devices) and are never added when model.devices is empty.

## Steps

1. Fix the const-correctness bug: use `const llama_model * other_model = llama_get_model(other);` (matching the real return type at include/llama.h:584), not a non-const pointer.
2. Move the whole ctx_other device-enumeration block to AFTER the ordinary model-device loop completes, immediately BEFORE the ACCEL enumeration loop -- CORRECTED anchor: insert_before the unique ACCEL loop `for (size_t i = 0; i < ggml_backend_dev_count(); ++i) { ... GGML_BACKEND_DEVICE_TYPE_ACCEL ...}` (verify this exact loop text at implementation time), not inside the ordinary model.devices loop where it currently sits (which reprocesses ctx_other per ordinary device and skips it entirely when model.devices is empty).
3. Reproduce target/draft device-list mismatch with shared tensors under speculative configuration.
4. Enumerate model_other->devices, deduplicate by backend device handle, and initialize only missing backends after ordinary model devices.
5. Preserve backend ordering relative to ACCEL/CPU and verify ownership/destruction of added instances.
6. Test same-device no-op, subset/superset/disjoint lists, Meta-wrapped tensor split and single-GPU cases.
7. Verify shared tensors schedule on an allocator-valid backend with output parity and no unexpected copies; initialization failure must be explicit.

## Detailed Solution & Technical Design

Device identity comes from the initialized other model, not CLI guesses. This is not a speed patch; backend lifetime/order and shared allocation validity are the acceptance boundary.

## Code Samples & Guidance



## Files

src/llama-context.cpp context backend construction; speculative ctx_other setup; scheduler/shared-tensor integration tests; device-list fixtures.

## Validation

Patch mechanics: `PYTHONPATH=tools python -m bigcherry patch-lint patches/1261_nro10_spec_ctx_other_devices`; `PYTHONPATH=tools python -m bigcherry patch-rebase-check --focal-overlay 1261_nro10_spec_ctx_other_devices --source bigcherry-tuning`; package pytest offline. Fixtures: mismatched-device control vs subject, same-device no-op, subset/superset/disjoint lists, Meta-wrapped tensor split, single-GPU. Static/unit-level output/copy-topology and lifetime checks (no hardware required for most of this item since it's an orchestration/scheduling correctness fix, not a kernel). Hardware confirmation (Brutus, 2+ device speculative config): `python -m bigcherry.patch.validation_campaign --overlay 1261_nro10_spec_ctx_other_devices --arch gfx1100` verifying shared tensors schedule on an allocator-valid backend with output parity and no unexpected copies; initialization failure must be explicit.

## Effort & Risk



## Standards

Affirmative scheduler/output evidence; fail explicitly on missing required backend; no throughput claim.

## Acceptance Criteria

Shared tensors execute on valid allocation backends under mismatch; no duplicate backend or lifetime leak; controls remain unchanged.

## Notes

Supersedes: NRO11
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-nasone-rdna-optimizations-nro11

2026-09-24 relevance at b11126: IMPLEMENTED-AS-PATCH. patches/1261_nro10_spec_ctx_other_devices exists, state=untested. Item's Files section still names raw src/llama-context.cpp paths (pre-patch-package phrasing) but the mapped package already exists (Successor key nro11 / Supersedes NRO11 in Notes, id verified in patches/1261*/patch.toml). No upstream absorption found relevant to this specific ctx_other device-list orchestration bug. Disposition: validate/qualify existing patch; no GPT design needed.

2026-09-24 GPT review req_215c89d0b13a4bb7 applied: verified via direct read that patch.py assigns llama_get_model()'s const-qualified return (include/llama.h:584) to a non-const llama_model* -- a real compile error. Also verified the backends.emplace_back(backend) anchor occurrence 0 sits inside the ordinary model.devices loop, causing per-device reprocessing/duplication and complete skip when model.devices is empty. Fixed both: corrected pointer type and moved the block to run once, after the ordinary loop and before the ACCEL enumeration loop.

2026-09-30: does not apply to MTP on a tensor split. 1261's WARN marker fires only when ctx_other contributes a device backend absent from the draft context (!already_present); with -sm tensor the MTP draft shares the main model's devices, so the path never runs (dev-gpt-agent req_68f6c09055474b6d; Brutus preflight on Qwen3.8-27B Q8_0 dual XTX found no marker). Relevant only for a separate draft model placed on other devices. Off the 27B priority queue.

## Change Log

- 2026-09-09T10:52:53.458006+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:09:20.280241+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.094454+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.745657+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:44:11.869473+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_024433_three-more-nasone-successors-n_7555
- 2026-09-10T02:44:33.151338+00:00 (updated-by): Updated: section:ledger-events
- chg_20260920_064350_patch-1261-pnro10-verified-o_7857
- 2026-09-20T06:43:55.564429+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:26:37.993604+00:00 (updated-by): Updated: section:description, section:validation, section:notes
- 2026-09-24T04:50:24.013352+00:00 (updated-by): Updated: section:description, section:steps, section:notes
- 2026-09-30T13:38:15.199353+00:00 (updated-by): Updated: section:notes
