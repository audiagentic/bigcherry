---
id: BCOP49
order: 49
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-06T09:08:00+11:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: S
---

# Disposition adaptive ROCm expert-cache evidence through MET01

## Description

2026-10-06 optimisation audit disposition for adaptive routed-expert caching. MET01 remains the sole technical owner. This item records only the action/gate created by the audit.

## What changed/discovered

- BigCherry's current `routing-profile.sh` captures aggregate per-layer expert frequency but not temporal locality/reuse distance, so it cannot decide whether a runtime adaptive cache is worth its promotion/churn cost.
- `ap03906101/moe-hotcache` demonstrates a graph-safe adaptive-cache architecture on ROCm: device-side route counters captured in the HIP graph, background host refresh, copy-before-publish pointer updates, redirect-before-reuse eviction and per-tensor cache budgets.
- The fork is also strong negative evidence: its headline Qwen3.8 dynamic-cache and prefill gains were retracted after batch paths produced corrupt output. Only its quality-verified partial-pin Qwen3.8 decode result (~11.4 -> 13.0-13.4 tok/s on gfx1200) is usable as measured mechanism evidence.
- `Maxritz/Strata-rocm` independently reports PCIe-saturated expert miss handling on gfx1201 and a reverted double-buffer overlap experiment (524 ms vs 485 ms), so overlap/prefetch is not assumed beneficial when the link is already saturated.
- Upstream llama.cpp #29943 remains the preferred integration seam: cache/selective-copy policy belongs in user/context code, not another generic scheduler implementation.

## Already acted upon

MET01 now requires temporal routing traces, an offline static/LRU/decayed-frequency policy replay, graph-safe adaptive refresh, explicit batch/quant dispatch coverage, and fail-closed fallback before adaptive cache implementation. Static/LRU/adaptive policies must share one cache-bank/accounting representation.

MET05 already owns the PCH-attached 6900 auxiliary path. The external result that a chipset-limited GPU can be slower than host memory for miss-driven traffic reinforces the existing MET05 rule; it does not create another auxiliary cache tier.

## Unresolved action

1. Extend the canonical Flash-Next route trace with token order, phase, route weights and workload identity.
2. Offline-replay static-hot, LRU and decayed-frequency/bounded-promotion policies at equal slot budgets.
3. Reject adaptive runtime work unless it reduces miss bytes/cost materially versus static on at least two decode workloads.
4. If the offline gate passes, qualify through the #29943/#29887 user-code cache seam with explicit batch/quant correctness boundaries before performance testing.

## Terminal disposition

- **PASS:** adaptive policy beats the best simpler static/LRU policy by MET01's offline gate, then delivers >=3% end-to-end TG over that simpler policy on at least two decode workloads at equal expert VRAM, while all correctness/batch/quant gates pass. Retain under MET01.
- **FAIL:** offline policy does not beat static/LRU materially, runtime gain is <3%, promotion traffic erases the gain, prefill/unaffected regimes regress >2% without a clean workload gate, or any output/work-accounting check fails. Drop adaptive policy; retain the simpler MET01 cache.
- No separate scheduler, cache allocator, placement solver, telemetry subsystem or 6900 miss-cache mechanism is authorized by BCOP49.

## Dependencies

- MET01: authoritative policy/cache qualification owner.
- llama.cpp #29943/#29887 or successor: preferred copy-callback/cache seam.
- MET05/1328: sole auxiliary 6900 execution/transport owner.
- Existing BigCherry correctness/KLD and benchmark harnesses for promotion evidence.

## Acceptance Criteria

- MET01 contains the implementation-ready temporal/adaptive qualification gate.
- Offline trace replay is run before adaptive runtime code is proposed.
- External ROCm results are treated as mechanism/negative evidence, not transferred performance claims.
- Any runtime candidate proves correct work across dispatch boundaries and representative quant types before speed is considered.
- This BCOP closes when MET01 either promotes one adaptive policy or records a terminal rejection in favour of static/LRU.

## Notes

Filed on main on 2026-10-06 as BCOP30 (commit 432edca1) while the work branch already had a different BCOP30 (the runtime placement cost model). Re-numbered BCOP49 when main was merged for the bc-11474.0.0 release; the text is unchanged apart from the id.
