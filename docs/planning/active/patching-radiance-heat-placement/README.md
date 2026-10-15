# RHP — Radiance unified heat and adaptive cross-GPU expert placement

Status: active plans, not implemented. Prefix: `RHP`. Pinned Radiance: `89cee7ce414486c717aa2085c99e1f522e4c0ea6` (1.3.0).

## Decision and scope

**Reuse Radiance's authoritative router histograms, HeatEngine, residency mover and existing metrics.** Add a *consumer* for cross-GPU ownership planning in BigCherry's `libr3moe`; do not add another GPU counting kernel, heat database, scheduling thread inside the token critical path, or independent monitoring stack. Keep GPU execution asynchronous; CPU does the noncritical scoring. Maintain the difference between (a) Radiance VRAM/host tier residency and (b) which GPU owns and computes a routed expert. Remote owner must never be incorrectly treated as an ordinary locally movable tier.

Existing source seams (pinned upstream):
- `core/runtime/ctx.cpp:Ctx::route_report_impl/route_flush/next_histogram`: device-produced expert counts, banked async host copy, step/generation attribution.
- `core/engine.cpp:Engine::tiering_dispatch`: drains landed banks, `HeatEngine::observe`, mover, promotion stats. Currently long-prefill histograms often not credited due to host run-ahead; investigate loss versus delay, not just staleness.
- `core/place/heat.h/.cpp`: decayed LFU, hysteresis, per-layer scores, promoted/readmit/stream metrics; **disabled** if nothing movable or deterministic.
- `core/place/mover.h/.cpp`: shadow-slot asynchronous weight transfers and publication; **within a rank**, not a cross-GPU helper mover.
- `core/engine.cpp` expert-plane report and `core/server/metrics.cpp`: current observability, to be extended rather than cloned.

BigCherry current duplicate: `libr3/helper/moe_remote_hook.cpp` allocates optional `R3_MOE_HELPER_HISTOGRAM` GPU counts and writes `<path>.rankN` at shutdown; `r3_moe_helper.cpp::select_owned` consumes `R3_MOE_HELPER_HEAT` or static container profile. This duplication is **temporary validation only** until shared-path count equivalence and prefill capture coverage are proven. Existing `rad0002_moe_remote_handoff` patch has declare-time owner/map hooks and `R3_MOE_HELPER_SKIP_OWNED` (remote expert not present on engine); it does **not** support live ownership changes.

## Dependency-ordered work

| Item | Priority | Deliverable | Depends on |
| --- | --- | --- | --- |
| [RHP01](RHP01.md) | P0 | Trace route histogram lifecycle; fix prefill coverage without synchronous GPU waits | RAD09 baseline for helper test |
| [RHP02](RHP02.md) | P0 | One versioned, read-only histogram consumer for HeatEngine and `libr3moe`; retire duplicate collector after parity | RHP01, RAD09 |
| [RHP03](RHP03.md) | P1 | Extend Radiance observability + bounded runtime snapshots, reuse existing storage conventions | RHP02 |
| [RHP04](RHP04.md) | P1 | Cost-aware GPU ownership optimiser, recommendation/shadow only | RHP02, RHP03 |
| [RHP05](RHP05.md) | P1 | Request/time-triggered reevaluation, apply via controlled restart with verified rollback | RHP04 |
| [RHP06](RHP06.md) | P2 | Optional live GPU-to-GPU expert migration with publication epochs | RHP05 and measured need |

## Policy boundaries

- **Collection is continuous; decisions are occasional; physical migration is rarer.** A bounded lag is fine; silent unaccounted bank overwrite is not. Keep dropped, sampled and credited counters and coverage by prefill/decode.
- Proposed starting experiments: reevaluate after **1,000 completed requests OR 30 minutes**, require independently measured benefit (initial hypothesis **>=5%**) and cooldown **30 minutes**. These are **nondefault experimental parameters**, not proven optimal values. Count completed requests using Radiance's existing service counters; clock only triggers an assessment.
- Score per `(layer, model-global expert ID)` using arrival counts and GPU-specific cost estimates (per-phase routed work, helper prefill/decode throughput, resident bytes, available VRAM, remote publish+wait latency, local host-pool misses, topology). Consider prefill and decode separately; use a measured mixture, not unqualified frequency alone.
- All remotely owned experts, including **protected** experts and MTP layers, must follow the exact runtime eligibility/slot map; never infer ownership from container IDs. Keep helper budget, `max_rows`, `min_rows`, unplaced weights, fallback eligibility and rank geometry explicit.
- Preserve `--deterministic`: no dynamic residency/owner changes; instrumentation must not silently force HeatEngine on. For all-VRAM/remote-only plans, shared observation still works without making weights movable.
- Use one existing Radiance metrics/expert-plane surface for live state; machine-local `<work>/state/radiance/<model-digest>/<gpu-cohort>/placement.json` for **accepted placement generation and provenance only**. Heavy telemetry stays in RAM and existing `artifacts/<run-id>/`; reviewed compact evidence in `docs/evidence/<run-id>/`. Do NOT add SQLite just for heat, duplicate stats, Git-track runtime state, or label Prometheus with per-expert IDs.
- Version/CRC/checkpoint active ownership snapshots; no raw prompts or prompt-derived content in persistent heat data. Stage restart changes with conservative validation; zero-copy/unplaced weights require source availability before reassignment.

## Common gates

1. Offline deterministic traces incl. 1/8/64/1024/4096 rows, mixed prefill+decode, MTP repeated reports, ring wrap, event delayed, dropped banks, protected experts, both helpers and reset/restart. No double-credit or wrong-map publishes.
2. Control with heat collection/owner planner disabled: exact server output and no meaningful throughput/TTFT regression; no new GPU synchronisation, no CPU critical-path work, no deadlock.
3. Real Brutus Flash-Next 7,306- and 32K-token prefill plus 256-token decode; 4-document KL (text-only and tail metrics if available), balanced ABBA, `--compare` and graph replay. Treat helper KLD ~0.058–0.067 vs XTX-pair exact as *existing noise-band evidence*, not licence for further unmeasured regressions.
4. Report heat coverage, capture lag, percent remote routes, p50/p95/p99 latency, expert transfer bytes, mover readmits, placement churn, helper wait and GPU memory usage; verify no double-counting across ranks.
5. No automatic promotion to production profiles until independent performance/quality and rollback gates pass.

## Out of scope

Rewrite of Radiance's existing `HeatEngine` policy, independent heat-store service, scheduler rewrite, online training of router, changing model expert selection/top-k, or unqualified automatic live migration.
