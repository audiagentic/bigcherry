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

This is **not** a generic MTP speed patch: native tensor-split MTP uses the same model devices, so the missing-device branch does not execute (historical 2026-09-30 dev preflight). The actual positive control must use a separate DFlash/DSpark/EAGLE draft on a device set that does not contain all of the target's physical devices. QFP43/1286 owns one-time copies of borrowed Meta-backed tensors for a single-device DFlash draft; it does **not** replace 1261's foreign-device scheduler admission, nor does 1261 replace its copy. PNRO11's target-wide mirroring proposal is closed. Do not modify QFP43 or active Radiance helper/mailbox implementation.

### Bounded validation and terminal gates

1. **Cheapest static/host gate:** compose 1261 against the exact pinned source; assert the injected block occurs once **outside** the ordinary GPU loop and before ACCEL. Source-shaped device-list cases: empty, same-set, subset, superset, disjoint, overlap-with-future-ordinary-device; assert unique device handles, stable ordering, and one `ctx_other` enumeration per context. Inject a `ggml_backend_dev_init` failure and require explicit exception with no retained partial context.
2. **Patch/build gate:** run `patch-lint`, `patch-rebase-check`, the existing offline package tests, and HIP compilation for gfx1100 with 1261 composed against the production patch set. Treat any source-anchor mismatch or duplicate as a hard reject. No such build/test is claimed by this audit.
3. **Hardware correctness gate, only with a real eligible draft:** record target/draft physical-device IDs, buffer owners, `ctx_other` identities, before/after backend ordering, conditional `BIGCHERRY_PATCH_TRACE` marker, scheduler allocation success, host staging/Meta fallback, and repeated same-process requests. Compare greedy text, logits/KLD where available, accepted/proposed counts, and graph replay at short and long contexts. Missing marker on a known foreign-device positive control is **FAIL**, while missing marker on same-device MTP is expected. Never infer P2P from physical PCIe connectivity; gfx1100/gfx1201 and gfx1030 are separate qualification lanes.
4. **Performance gate:** correctness-first. No throughput claim without matched unpatched control, actual executed-route evidence, and independent paired prompt/decode sessions; reject a change that adds duplicate backends, unbounded VRAM, extra transfers or >1% stable non-target regression. If no supported foreign-device draft/target lane exists, terminal `AWAIT_ELIGIBLE_DRAFT` and do not queue hardware experiments. If upstream lands an equivalent validated fix first, compare exact semantics and retire the local duplicate.

**Audit evidence:** 8/8 source-shaped assertions passed against pinned b11552 source + current 1261 patch; four deterministic host topology cases produced the two expected current-code failures (empty, future-ordinary overlap) and two controls (subset, disjoint). Upstream #26636 diff inspected; vLLM/SGLang separate draft-worker/head ownership is a design contrast, not a transferable GGML backend fix. No patch application, repository pytest, HIP build, GPU execution or new benchmark was performed. BCOP193 is the action ledger.


## Description

**2026-10-09 b11474 disposition: unqualified, do not promote patch 1261.** This is a scheduler correctness repair for a **separate draft** whose shared target tensors reside on devices absent from the draft model's own list; it is not a general MTP speed optimisation. The local package already uses the correct `const llama_model *` return type, but its `Edit(mode="insert_after", occurrence=0)` still injects the ctx_other loop *inside* the first `for (const auto & dev : model.devices)` loop. That can append the same GPU backend twice and skips all ctx_other devices if the draft device list is empty. The 2026-09-24 plan note claiming both issues fixed is incorrect for the published package.

At b11474, `llama_context::llama_context` initially sets `cparams.ctx_other=nullptr`; it assigns it only for Gemma4Assistant, or for Eagle3/DFlash missing local token embedding or output. Native Qwen MTP on the same tensor-split devices does not exercise this patch. The trace marker fires **only when a missing other-model device is added**, not unconditionally. Upstream llama.cpp [#26636](https://github.com/ggml-org/llama.cpp/pull/26636) (open as of 2026-10-09) already proposes the correctly placed mechanism; its reported success is on NVIDIA, not RDNA/no-P2P.

## Steps

1. **Admission before implementation:** require a reproducible supported separate-draft `ctx_other != nullptr` case, missing `tok_embd` or `output`, and a target-owned preallocated tensor on a GPU absent from draft `model.devices`. Record model/quant/GGUF tensor inventory, exact `-dev`/`-devd`/`-sm` flags, device order, graph abort, and an unmodified b11474 control. If absent, terminate PNRO10/1261 without porting or hardware queue.
2. If reproduced and upstream #26636 remains unmerged, repair **only the existing 1261 package**: insert once after the *closing brace* of the ordinary `model.devices` loop and before the ACCEL loop in `src/llama-context.cpp::llama_context::llama_context` (b11474 lines 335-345). Match the enclosing source block uniquely; `backends.emplace_back(backend);` occurs twice and is unsafe as a bare anchor. Preserve `const llama_model *` and fail explicitly if required backend initialization fails.
3. Deduplicate by `ggml_backend_dev_t` against **all** already-instantiated backend handles. Construct target-only backends as new context-owned `ggml_backend_t` instances, never alias another context's backend object. Preserve ordinary GPU order, then missing target devices, then ACCEL, then CPU **last**. Enforce scheduler's `GGML_SCHED_MAX_BACKENDS=16` bound before `ggml_backend_sched_new`; a duplicate is not harmless.
4. Prove `backend_ptrs` and `backend_buft` are rebuilt after enumeration, `sched_reserve()` observes the added backend, and context destruction synchronizes and frees instances. Device admission does **not** prove cross-GPU P2P, a supported Meta split axis, or correct placement; retain the normal host-staged/no-P2P fallback and fail on unsupported shared tensor buffers.
5. Run the host fixture and patch mechanics/compile gate before any hardware. Only if a genuine supported failing case exists, run same-process multi-request and multi-ubatch target/draft controls with graph reserve/replay, output/logit parity, complete accepted-token accounting, memory/allocator safety and no duplicate backend. Otherwise close as no applicable workload.

## Detailed Solution & Technical Design

**Exact b11474 trace:** `src/llama-context.cpp:146-165` sets `cparams.ctx_other`; lines 335-342 enumerate the draft's model devices; lines 344-354 enumerate ACCEL; CPU is appended at 356-361; lines 402-421 populate scheduler backend arrays; `sched_reserve()` follows. `ggml/src/ggml-backend.cpp:903-904,1937-1946` bounds the scheduler to 16 backends and requires CPU last; line 1084 aborts when a preallocated tensor's backend cannot run its operation. `src/models/dflash.cpp` uses `ctx_other` for target-owned `tok_embd` (around 678/917) and `output` (around 803/1013).

**Local defect:** `engines/llamacpp/patches/1261_nro10_spec_ctx_other_devices/patch.py` uses `anchor=r"backends\\.emplace_back\\(backend\\);"`, `expect_matches=2`, `occurrence=0`, `mode="insert_after"`. The first occurrence is **inside** the ordinary loop (b11474 line 341), not after it. A two-device draft [A,B] with target [B,C] yields [A,B,C,B] rather than [A,B,C]. With draft [] and target [C], it adds nothing. No measured throughput or RDNA hardware success follows from this static finding.

**Repair pseudocode (conditional, not implemented):**
```cpp
for (const auto & dev : model.devices) { init_owned_backend(dev.dev); }
if (cparams.ctx_other != nullptr) {
    const llama_model * other = llama_get_model(cparams.ctx_other);
    if (other == nullptr) { fail_explicitly(); }
    for (const auto & dev : other->devices) {
        if (!has_device(backends, dev.dev)) { init_owned_backend(dev.dev); }
    }
}
for (auto accel : available_accel_devices()) { init_owned_backend(accel); }
init_cpu_backend_last();
assert(backends.size() <= GGML_SCHED_MAX_BACKENDS);
```
Use the existing backend vector, scheduler and ownership; do not introduce another placement solver, cache, scheduler or device configuration surface. The real patch must use source-aware, unique placement rather than copying this pseudocode verbatim.

**Dependencies:** PNRO10 owns this repair. PNRO11 owns independent DFlash/DSpark full-vocabulary `lm_head` replication; QFP43/1286 owns draft acceptance/trace and is not modified. Existing 1252 P2P is rejected on this lab topology and must not be required. Upstream #26636 is the direct port/reference; vLLM's `vllm/v1/spec_decode/draft_model.py` deliberately avoids sharing draft embeddings/lm_head and rejects some divergent TP arrangements; SGLang's [parallel-speculation roadmap](https://github.com/sgl-project/sglang/issues/27462) isolates models in separate processes. These are design contrasts, not transplantable RDNA speedups.

## Code Samples & Guidance

**Candidate repair location:** `engines/llamacpp/patches/1261_nro10_spec_ctx_other_devices/patch.py` targeting `src/llama-context.cpp::llama_context::llama_context`. Reuse the current `backends`, `backend_ptrs`, `backend_buft`, `sched_reserve` and context lifetime. No implementation edit is authorized by this audit; patch remains untested/default-off.

## Files

- Authoritative plan: `docs/planning/active/patching-nasone-rdna-optimizations/PNRO10.md`; package `engines/llamacpp/patches/1261_nro10_spec_ctx_other_devices/{patch.py,patch.toml,README.md,SUMMARY.md}`.
- Pinned upstream anchors: `src/llama-context.cpp:146-165,335-365,402-455`, `src/models/dflash.cpp`, `ggml/src/ggml-backend.cpp:1084,1937-1946`. Reference: [llama.cpp #26636](https://github.com/ggml-org/llama.cpp/pull/26636).
- Existing static test module `tools/tests/patch/test_nro_patch_packages.py` does not list 1261 in its `EXPECTED` package matrix; add focused tests **only after** a reproduced supported case warrants repair.

## Validation

**Performed in this audit (no GPU):** 5 pinned-source/patch-static assertions and 5 deterministic host device-list fixtures passed. Fixtures reproduced the actual nested-insertion ordering [A,B,C,B] for draft [A,B]/target [B,C], and empty-draft omission for draft []/target [C]. The corrected after-loop model produces [A,B,C] and [C], respectively. These are **host simulations**, not a compiled patched source, actual allocator result or measured performance.

**Next, conditional:** patch-lint/rebase-check at the pinned b11474 checkout, C++ compile, static test with zero/one/two draft devices, overlap/disjoint/subset/superset, `ctx_other` null and non-null, backend cap, failed device init, CPU-last invariant and no duplicated backend; run repository pytest as appropriate. Then only for a reproduced supported separate draft, qualify target dual gfx1100 + draft R9700 (or a documented alternative) with same-process repeated requests, multiple ubatches, graph rebuild, full-vocab/greedy comparison, MTP/draft acceptance and actual backend/transfer trace. Preserve host-staged transfer fallback because `hipDeviceCanAccessPeer` is false for the lab GPU pairs. Require no correctness failure and <=1% regression on same-device controls. No hardware queue unless the prerequisite failing case exists.

**Terminal:** No supported shared-tensor cross-device draft or upstream absorption => close/retire 1261; valid case and successful upstream fix => adopt upstream instead of maintaining a fork; valid case and upstream still open => one scoped corrected 1261 implementation, with complete correctness and no duplicated backend. Throughput benefit is not required for this correctness repair; never claim a speedup from a missing crash.

## Effort & Risk

Risk: high if applied blindly (duplicate backend objects, scheduler cap, preallocated buffer incompatibility, graph failure); low scope when left default-off. Existing BigCherry 2026-09-30 dual-XTX Qwen MTP preflight observed no 1261 activation marker; 2026-10-04 DFlash/DSpark TP aborts were not root-caused to this exact path. No PNRO10 performance baseline, GPU qualification or patch build exists. Do not schedule unrelated MTP/QFP43 work.

## Standards

Pin actual source and supported draft configuration; fail closed on missing device/allocator proof; distinguish no-op from activation; never infer P2P; preserve target/draft correctness, scheduler lifetime and fallback; use upstream before local fork.

## Acceptance Criteria

No promotion until a real supported shared-tensor cross-device draft fails stock and passes the repaired implementation with correct backend order, no duplicates, CPU last, valid allocator placement, graph replay and full-vocabulary parity. Same-device/native-MTP controls unchanged. If no such case, close without further implementation or benchmarks.

## Notes

**2026-10-09 correction:** The 2026-09-24 note said both const-correctness and insertion placement were fixed. Current published `patch.py` proves only const-correctness was fixed; `occurrence=0` still inserts inside the draft GPU loop. This audit is documentation-only; it does not fix or enable the patch.

**2026-09-30 lab evidence:** 27B Q8_0 dual-XTX native MTP on `-sm tensor` produced no 1261 marker because its draft and target share the same devices. That is a **no-op**, not a failure or success of the separate-draft repair. DFlash/DSpark failures on other device layouts remain separate investigations; do not infer PNRO10 causality.

Historical lineage: successor of NRO11; existing patch 1261 is untested. The current source is b11474, not the earlier b11126 plan snapshot. Upstream [#26636](https://github.com/ggml-org/llama.cpp/pull/26636) remains open (checked 2026-10-09); its independent NVIDIA reproduction is useful for mechanism and placement, not RDNA qualification.

## Change Log

- 2026-10-09 (BCOP99): rebaseline 1261's remaining nested-anchor defect, no-op MTP admission, upstream #26636 and conditional correctness gate.
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
