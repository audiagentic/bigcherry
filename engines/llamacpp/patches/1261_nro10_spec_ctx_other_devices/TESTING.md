# Testing — PNRO10 speculative ctx_other scheduler devices

**Status: BLOCKED_SOURCE; do not sign off the current 1261 package.** See PNRO10 for the exact b11552 insertion defect.

## Source and cheap host gates

1. Apply the corrected patch to the exact pinned llama.cpp revision. Assert the ctx_other block is outside `for (const auto & dev : model.devices)`, after that loop, before ACCEL and CPU. Assert a single insertion and unchanged no-ctx_other control.
2. Source-shaped device-list fixtures: empty, identical, subset, superset, disjoint, overlapping future ordinary GPU. Check no duplicates, deterministic order, one enumeration, correct failure propagation when backend initialization fails. In the current patch, empty and overlapping-future cases **fail** by construction.
3. Run package `patch-lint`, `patch-rebase-check`, offline tests and HIP gfx1100 compilation against the composed production set. A successful static mock does not substitute for compilation.

## Activation and correctness

Use **two separate controls**:
- **Negative:** Qwen3.8-27B Q8_0 dual-XTX `-sm tensor` native MTP, where draft and target device sets coincide. No newly added backend and **no activation marker expected**. Greedy/acceptance parity is a regression control, not proof that PNRO10 executed.
- **Positive (only when available):** separate DFlash/DSpark/EAGLE draft with at least one target physical device absent from draft scheduler. Set `BIGCHERRY_PATCH_TRACE=1`; require `BIGCHERRY_PATCH_HIT patch=1261_nro10 path=ctx_other_backend` for each newly appended device, plus actual backend identity/ordering and valid borrowed-buffer ownership. Missing marker here is a failed qualification.

For both: record physical GPU IDs, draft/target model and buffer placement, no duplicate backend handles, init/destruction and repeated same-process requests, graph reuse, long-context/ubatch, greedy text, logits/KLD where supported, acceptance counts, and no unexpected D2H/H2D transfers. Treat Meta-backed borrowed tensors as a separate QFP43/1286 path; do not assume no-P2P GPUs can directly read each other's allocations.

## Promotion / terminal

This is a correctness repair, not a claimed speedup. Do not queue a benchmark until the positive route exists and passes; then use matched control/subject sessions, >=10 paired rounds where the bound contract requires, no >1% non-target regression, and no new unexpected copies/VRAM growth. Without a positive eligible topology record `AWAIT_ELIGIBLE_DRAFT`. No test/build/GPU run was performed in the 2026-10-14 documentation audit.
