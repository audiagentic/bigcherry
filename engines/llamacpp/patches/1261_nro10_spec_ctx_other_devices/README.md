# 1261 PNRO10 — ctx_other speculative scheduler devices

<<<<<<< ours
**State: untested; source placement blocker (2026-10-14).** The patch intends to register missing target-model physical device backends in a separate draft context, after draft-model backends and before ACCEL/CPU. It does not optimize native same-device/tensor-split MTP.

Current `patch.py` inserts after the first `backends.emplace_back(backend);`, which is **inside** the ordinary model-device loop at pinned llama.cpp b11552. Empty draft lists skip foreign devices; overlap with a later ordinary device can duplicate a backend. PNRO10 owns the exact fix and validation. Do not promote 1261 until the anchor is moved outside the loop and source/compile/correctness gates pass.

**Positive activation:** a separate draft/target pair with a missing physical device, `BIGCHERRY_PATCH_TRACE=1`, and observed `BIGCHERRY_PATCH_HIT patch=1261_nro10 path=ctx_other_backend` for the appended device. Same-device MTP produces **no** marker and is a negative control. Backend registration alone does not prove Meta-backed borrowed tensors are schedulable; QFP43/1286 owns the distinct draft-local copy.

See PNRO10 and `TESTING.md`. No current-pin hardware qualification or speedup is established.
=======
**Untested; default-off; do not promote.** This is a *correctness* repair for a separate DFlash/Eagle/Gemma4Assistant-style draft that borrows target-owned `tok_embd`/`output` through `ctx_other` while the target tensor resides on a device missing from the draft's own `model.devices`. It is **not** a built-in Qwen MTP throughput patch.

## Current source-level defect (b11474, audited 2026-10-09)

`patch.py` has fixed the earlier `llama_get_model` const-pointer error, but `Edit(anchor="backends.emplace_back(backend);", occurrence=0, mode="insert_after")` still inserts **inside** the ordinary draft model-device loop (`src/llama-context.cpp:335-342`). It can duplicate a later draft backend: draft [A,B] / target [B,C] -> [A,B,C,B], or skip the target backend entirely when draft devices are empty. The patch's statement that it preserves order is therefore false for the current code.

At b11474 `cparams.ctx_other` is null for ordinary native MTP; it is set for Gemma4Assistant and Eagle3/DFlash missing local embeddings/output. The optional `BIGCHERRY_PATCH_HIT patch=1261_nro10` marker is emitted **only** when a new backend is added; marker absence on same-device MTP is expected and cannot qualify this patch.

## Disposition / qualification

Do not enable or queue GPU work without a supported reproducible target-owned shared-tensor failure on a separate draft. If one exists, compare against open upstream [llama.cpp #26636](https://github.com/ggml-org/llama.cpp/pull/26636), which inserts the block **after** the ordinary device loop and before ACCEL. Fix only this package, with a unique source anchor and one insertion, dedup by `ggml_backend_dev_t`, context-owned instances, CPU last, `GGML_SCHED_MAX_BACKENDS=16` bound, allocator/graph reserve/replay proof, explicit init failure and no-P2P fallback. See [PNRO10](../../../../docs/planning/active/patching-nasone-rdna-optimizations/PNRO10.md) for the exact matrix and terminal criteria.

**Evidence:** 5 source-static and 5 deterministic host-list fixtures passed; these demonstrate a structural defect, not a compiled patch, GPU success or speedup. No repository pytest, HIP build or hardware benchmark ran. Existing dual-XTX MTP preflight did not activate the patch. Upstream #26636 reports NVIDIA reproduction only.
>>>>>>> theirs
