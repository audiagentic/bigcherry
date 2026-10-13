# 1261_nro10_spec_ctx_other_devices

**Status:** untested / BLOCKED_SOURCE (2026-10-14)
**Authoritative plan:** PNRO10

## Scope and current defect

Register target `ctx_other` model devices missing from a separate draft context's scheduler, with physical-device deduplication and owned `ggml_backend_dev_init` instances. Intended order: ordinary draft GPUs → missing target GPUs → ACCEL → CPU.

At llama.cpp b11552, `patch.py`'s `insert_after` of `backends.emplace_back(backend);` (`expect_matches=2, occurrence=0`) places the enumeration **inside** the ordinary GPU loop. Empty draft list adds none; a later overlapping ordinary device can be added twice. The earlier documentation claiming this placement was fixed is stale. Do not promote or claim correctness based on an MTP same-device benchmark.

The `const llama_model *` fix is present. `BIGCHERRY_PATCH_TRACE` emits `BIGCHERRY_PATCH_HIT patch=1261_nro10 path=ctx_other_backend` only when a missing device is appended. No unconditional activation is possible. This does not replace 1286's Meta borrowed-tensor copy.

**Next gate:** move the insertion immediately before the unique ACCEL comment (outside the GPU loop), then run static placement/unique-device fixtures, patch composition, gfx1100 build, and a foreign-device draft positive control with numerical and lifetime parity. If no eligible foreign-device draft exists, `AWAIT_ELIGIBLE_DRAFT`; no hardware queue or performance claim.

**Upstream reference:** [llama.cpp #26636](https://github.com/ggml-org/llama.cpp/pull/26636) (open; diff places the block after the ordinary loop).
