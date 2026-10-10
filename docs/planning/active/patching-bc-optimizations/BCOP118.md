---
id: BCOP118
order: 118
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-10T15:11:54+00:00'
created-by: agent
priority: P1
work: S
---

# RPL01: gate placement scores on overlap-safe critical-path evidence

## Discovery / disposition

RPL01's additive Phase-B formula double-counts overlapped compute/copy and scheduler waits; a deterministic 10.0 ms vs 11.5 ms counterexample reverses its winner. Pinned/current llama.cpp scheduler uses async copy with synchronized blocking fallback and event waits. Existing BigCherry `rocprof.py::parse_kernel_trace` discards raw intervals and `prefill-kernel-table.py` sums durations, so neither can qualify a per-request critical-path score. Technical correction and bounded algorithm live **only in RPL01**.

## Ownership and 12-hour exclusion

RPL01: read-only placement scorer and admission. BCOP30: existing lifecycle ledger. QFP46: DAG/trace joins and event lifetimes. PHA03: topology/RCCL identity. MET01/MET04: residency evidence. Existing ggml backend scheduler: execution and copies. No new trace collector, scheduler, allocator, expert solver, placement registry, benchmark queue or runtime flag. Last independent RPL01 work 2026-10-06; no newer RPL01 PR/implementation/benchmark was found. Excluded active QFP41/1356 graph/dispatch and probes, QFP48/49 prefill, QFP36/1357 router, QFP17/1330, Radiance and Flash-Next QSA/MTP; no protected files changed. BCOP108-117 did not audit this cost-scoring mechanism.

## Unresolved action / terminal gate

Gate 0: find two qualified existing placement A/B pairs with aligned event/dependency receipts and same-build wall time; otherwise `INSUFFICIENT_TIMELINE` and no scorer implementation. Gate 1: reuse QFP46 semantics to reject overlap inversion, mixed clocks/identity, duplicate work, impossible VRAM and unsupported P2P. Gate 2: >=5 independent hold-out pairs across >=2 regimes, >=80% decisive winner prediction and zero forbidden recommendations. No candidate may claim >=3% potential without measured exposed critical-path share >=2.9126%. On failure, narrow/close RPL01; on success advisory-only under BCOP30. Dynamic actuation retains separate >=5% two-regime gate.

## Evidence and actual validation

Pinned b11474/current upstream `ggml-backend.cpp`; BigCherry `rocprof.py`, `schema.py`, `prefill-kernel-table.py`, QFP46 model test, PHA03/MET evidence; SGLang prefill transfer completion event and vLLM deferred KV free. Eight disposable Python unittest cases passed (8/8): overlap inversion, sync double count, aggregate ambiguity, clock mismatch, unknown/direct P2P, capacity, identity. No repository pytest, C++/HIP compilation, GPU exercise, hardware benchmark, measured speedup or new experiment.
