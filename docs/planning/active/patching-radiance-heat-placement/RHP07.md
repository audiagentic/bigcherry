---
id: RHP07
order: 7
plan: patching-radiance-heat-placement
state: pending
created-at: '2026-10-18T14:26:48.723847+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Tiered expert cache: performance-ranked hosts, decayed heat, hysteresis and interval reallocation

## Description

Owner idea 2026-10-19. Treat expert hosting as a performance-tiered cache. Tiers by measured cost: L1 R9700 engine-resident, L2 the 7900 XTX helpers, L3 the RX 6900 XT (gfx1030, when the RDNA2 backend exists), L4 host RAM (CPU compute or staged copy). Each (layer, expert) lives in the highest tier whose capacity its heat justifies. Reallocation runs at fixed intervals, never per request, with hysteresis so sessions that shift expert use do not thrash experts between tiers, and heat is aged so old demand fades. Builds on RHP02-RHP06 (single heat feed, persisted snapshot, cost-aware ranking, timed reevaluation, versioned migration epochs); this item is the policy layer and its offline evaluation.

## Steps

1. Heat signal: per (layer, expert) decayed counts with the decay constant in requests or tokens, not wall time; separate decode-row and prefill-row weights (decode is latency-critical, prefill is volume); per-session capped contribution so one session cannot dominate; optional critical-path weight from the trace (exposed helper wait by layer). Persist as the RHP03 snapshot, plus a bounded time series of interval snapshots for evaluation (not a database).
2. Tier cost model: measured cost per expert-row per tier from the replay harness (compute speed, transfer path, queue effect), not a fixed rank; capacity per tier (R9700 after trunk and KV, ~22 GiB a XTX, 16 GB 6900 XT, RAM).
3. Assignment: heat-per-cost fill from the top tier down, with a headroom slab per tier for background copies.
4. Stability: dual thresholds (promote above X, demote below Y<X), minimum residency, a move budget per interval, and move only when predicted saving exceeds the move cost (weight bytes over the link). Apply at interval boundaries through the RHP06 epoch flip.
5. Offline first: record real route traces over time (1/2/4/8 sessions, shifting topics), simulate policies with heat_mock/balance_score-style scorers; score critical-path latency and moves per hour, compare against static hot-first (today's baseline) and the RHP04 shadow ranking.
6. Only then a live trial behind a default-off flag with identity and stress evidence.

## Detailed Solution & Technical Design

## Code Samples & Guidance

## Files

## Validation

Offline: the policy beats static hot-first on a shifting-session trace by a stated margin with bounded moves per hour; no oscillation under a synthetic flip-flop trace. Live: identity of outputs across epoch flips, no hang over repeated flips, throughput not below static ownership.

## Effort & Risk

## Standards

## Acceptance Criteria

## Notes

Today there is no heat history: ownership comes from one static histogram file (R3_MOE_WORKER_HISTOGRAM, hot-first via R3_MOE_WORKER_HEAT) taken at startup. Radiance's own mover has an internal runtime tracker (promotions/demotions seen in its end-of-run summary) that is not exported over time. RHP03 deliberately persists a snapshot, not a heat database; this item adds the bounded interval series needed to evaluate policies. Related: RAD10 (multi-stream flattens heat), RCS07 (RDNA2 tier L3).

Added 2026-10-19 (owner): the topology must be configuration, not code. dev (R9700 + 2 XTX + 6900 XT behind slow uplinks) is one case; others are two P2P R9700s, a single 12 GB card plus RAM, or a different mix. So:
1. Topology descriptor: devices (arch, VRAM, usable expert budget after trunk/KV, role), and links as groups with measured bandwidth and latency (host to device, device to device, shared uplinks: devices behind the same hop share its bandwidth). Probed at startup (PCIe link chain from sysfs plus a short H2D/D2H/D2D copy benchmark) with a file override; recorded in the run record. The 6900 XT, for example, sits behind a PCIe 3.0 x4 hop (about 3.5 GB/s to host) while its own link is x16.
2. Cost model uses these measurements: a move costs bytes / measured bandwidth + fixed latency (+ contention on shared links); a placement is worth it only if expected future saving over a horizon (heat x per-use saving) exceeds the move cost. "Speed of moving experts" and "need to move" are both inputs, so slow-link devices end up holding experts permanently and fast-link devices do the swapping.
3. Swap cache: reserve a slab of VRAM (swap-reserve per device, configurable) on the device(s) with the best host link and spare memory, to hold L4 (RAM-resident) experts staged for execution. A background copier fills it (LRU/ARC inside the slab, bounded by interval move budget); an L4 expert in the slab executes on that GPU, a miss goes to CPU compute or a direct stream, per the four-tier dispatch cost model. The slab size trades against resident capacity on that device, so the simulator sweeps it (0 to a few GiB) per topology.
4. The simulator takes a topology file as input and is validated on several (dev, 2 x P2P R9700, single 12 GB GPU + RAM), so nothing in the policy assumes dev.
5. Runtime configuration: a [topology] and [expert-cache] section (interval, hysteresis thresholds, move budget, swap-reserve per device) in the engine runtime profile, with measured defaults.

Miss policy (owner question 2026-10-19: CPU, or copy into the VRAM swap cache and keep it until ejected?): both, chosen by cost, with admission control. On a miss with R ready rows: run on CPU now when cpu_time(R) < copy_time + gpu_time(R) (typical for one decode row: an expert pair is about 2.4 MiB from our numbers, 22 GiB over 9,320 pairs, so a copy over a 3.5 GB/s hop is about 0.7 ms against tens of microseconds of CPU compute from RAM); stage into the swap cache and execute there when R is large (prefill) or the expert is hot enough to be reused. Never stall a step on the copy: execute the current request on CPU (or a direct stream) while an async admit runs, so the expert is resident for its next use. Admission is not automatic: a prefill touches many cold experts once, so admit only when the expert's decayed heat beats the would-be victim's (TinyLFU-style doorkeeper), otherwise the slab fills with one-shot experts. Admitted experts stay until ejected by lowest heat-per-cost; in-use experts are pinned for the duration of the request. The simulator compares CPU-only, admit-on-miss, and hybrid-with-admission on the same traces.

## Change Log

- 2026-10-18T14:26:48.723847+00:00 (created-by): Created by agent
