# 1261_nro10_spec_ctx_other_devices

**Status:** untested
**Plan item:** PNRO10

<<<<<<< ours
## Scope and current defect

Register target `ctx_other` model devices missing from a separate draft context's scheduler, with physical-device deduplication and owned `ggml_backend_dev_init` instances. Intended order: ordinary draft GPUs → missing target GPUs → ACCEL → CPU.

At llama.cpp b11552, `patch.py`'s `insert_after` of `backends.emplace_back(backend);` (`expect_matches=2, occurrence=0`) places the enumeration **inside** the ordinary GPU loop. Empty draft list adds none; a later overlapping ordinary device can be added twice. The earlier documentation claiming this placement was fixed is stale. Do not promote or claim correctness based on an MTP same-device benchmark.

The `const llama_model *` fix is present. `BIGCHERRY_PATCH_TRACE` emits `BIGCHERRY_PATCH_HIT patch=1261_nro10 path=ctx_other_backend` only when a missing device is appended. No unconditional activation is possible. This does not replace 1286's Meta borrowed-tensor copy.

**Next gate:** move the insertion immediately before the unique ACCEL comment (outside the GPU loop), then run static placement/unique-device fixtures, patch composition, gfx1100 build, and a foreign-device draft positive control with numerical and lifetime parity. If no eligible foreign-device draft exists, `AWAIT_ELIGIBLE_DRAFT`; no hardware queue or performance claim.

**Upstream reference:** [llama.cpp #26636](https://github.com/ggml-org/llama.cpp/pull/26636) (open; diff places the block after the ordinary loop).
=======
Not promoted: structural defect at b11474 (2026-10-09). Owner: PNRO10 only; no new scheduler/placement owner.

Adds `ctx_other` target model device backends to a separate speculative draft context when target-owned shared tensors require them. The **current local insertion is misplaced** inside the ordinary draft-device loop (first of two `backends.emplace_back(backend)` matches); it may duplicate a later backend and skips other devices when draft devices are empty. Const qualification was fixed earlier. The marker is conditional on actually adding a backend, and native same-device MTP does not exercise this path.

**Disposition:** Hold/default-off. Require a real supported separate-draft cross-device shared-tensor stock failure before repairing the existing package. Use open upstream [llama.cpp #26636](https://github.com/ggml-org/llama.cpp/pull/26636) as the correctly placed mechanism; verify backend uniqueness, CPU-last ordering, 16-backend cap, graph/allocator safety, exact output parity and no-P2P fallback. Close without implementation if no such workload exists. No BigCherry PNRO10 speedup or hardware qualification is established.
>>>>>>> theirs
