# 1261 PNRO10 ctx_other speculative scheduler devices

**Untested; default-off; do not promote.** This is a *correctness* repair for a separate DFlash/Eagle/Gemma4Assistant-style draft that borrows target-owned `tok_embd`/`output` through `ctx_other` while the target tensor resides on a device missing from the draft's own `model.devices`. It is **not** a built-in Qwen MTP throughput patch.

## Current source-level defect (b11474, audited 2026-10-09)

`patch.py` has fixed the earlier `llama_get_model` const-pointer error, but `Edit(anchor="backends.emplace_back(backend);", occurrence=0, mode="insert_after")` still inserts **inside** the ordinary draft model-device loop (`src/llama-context.cpp:335-342`). It can duplicate a later draft backend: draft [A,B] / target [B,C] -> [A,B,C,B], or skip the target backend entirely when draft devices are empty. The patch's statement that it preserves order is therefore false for the current code.

At b11474 `cparams.ctx_other` is null for ordinary native MTP; it is set for Gemma4Assistant and Eagle3/DFlash missing local embeddings/output. The optional `BIGCHERRY_PATCH_HIT patch=1261_nro10` marker is emitted **only** when a new backend is added; marker absence on same-device MTP is expected and cannot qualify this patch.

## Disposition / qualification

Do not enable or queue GPU work without a supported reproducible target-owned shared-tensor failure on a separate draft. If one exists, compare against open upstream [llama.cpp #26636](https://github.com/ggml-org/llama.cpp/pull/26636), which inserts the block **after** the ordinary device loop and before ACCEL. Fix only this package, with a unique source anchor and one insertion, dedup by `ggml_backend_dev_t`, context-owned instances, CPU last, `GGML_SCHED_MAX_BACKENDS=16` bound, allocator/graph reserve/replay proof, explicit init failure and no-P2P fallback. See [PNRO10](../../../../docs/planning/active/patching-nasone-rdna-optimizations/PNRO10.md) for the exact matrix and terminal criteria.

**Evidence:** 5 source-static and 5 deterministic host-list fixtures passed; these demonstrate a structural defect, not a compiled patch, GPU success or speedup. No repository pytest, HIP build or hardware benchmark ran. Existing dual-XTX MTP preflight did not activate the patch. Upstream #26636 reports NVIDIA reproduction only.
