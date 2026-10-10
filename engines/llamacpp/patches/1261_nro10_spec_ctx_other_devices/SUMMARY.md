# 1261_nro10_spec_ctx_other_devices

**Status:** untested
**Plan item:** PNRO10

Not promoted: structural defect at b11474 (2026-10-09). Owner: PNRO10 only; no new scheduler/placement owner.

Adds `ctx_other` target model device backends to a separate speculative draft context when target-owned shared tensors require them. The **current local insertion is misplaced** inside the ordinary draft-device loop (first of two `backends.emplace_back(backend)` matches); it may duplicate a later backend and skips other devices when draft devices are empty. Const qualification was fixed earlier. The marker is conditional on actually adding a backend, and native same-device MTP does not exercise this path.

**Disposition:** Hold/default-off. Require a real supported separate-draft cross-device shared-tensor stock failure before repairing the existing package. Use open upstream [llama.cpp #26636](https://github.com/ggml-org/llama.cpp/pull/26636) as the correctly placed mechanism; verify backend uniqueness, CPU-last ordering, 16-backend cap, graph/allocator safety, exact output parity and no-P2P fallback. Close without implementation if no such workload exists. No BigCherry PNRO10 speedup or hardware qualification is established.
