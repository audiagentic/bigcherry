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

BigCherry current duplicate: `libr3/worker/moe_remote_hook.cpp` allocates optional `R3_MOE_WORKER_HISTOGRAM` GPU counts and writes `<path>.rankN` at shutdown; `r3_moe_worker.cpp::select_owned` consumes `R3_MOE_WORKER_HEAT` or static container profile. This duplication is **temporary validation only** until shared-path count equivalence and prefill capture coverage are proven. Existing `rad0002_moe_remote_handoff` patch has declare-time owner/map hooks and `R3_MOE_WORKER_SKIP_OWNED` (remote expert not present on engine); it does **not** support live ownership changes.

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
3. Real dev Flash-Next 7,306- and 32K-token prefill plus 256-token decode; 4-document KL (text-only and tail metrics if available), balanced ABBA, `--compare` and graph replay. Treat helper KLD ~0.058–0.067 vs XTX-pair exact as *existing noise-band evidence*, not licence for further unmeasured regressions.
4. Report heat coverage, capture lag, percent remote routes, p50/p95/p99 latency, expert transfer bytes, mover readmits, placement churn, helper wait and GPU memory usage; verify no double-counting across ranks.
5. No automatic promotion to production profiles until independent performance/quality and rollback gates pass.

## Out of scope

Rewrite of Radiance's existing `HeatEngine` policy, independent heat-store service, scheduler rewrite, online training of router, changing model expert selection/top-k, or unqualified automatic live migration.

## Implementation contracts and mock evidence (2026-10-15)

Each RHP01–RHP06 plan now contains low-level source seams, explicit algorithm/API contracts, negative cases, owner-map and lifecycle invariants, measurement requirements and go/no-go checks. RHP01 and RHP02 must establish reliable common histogram evidence **before** cost-aware ownership is trusted; RHP05 applies via controlled restart; RHP06 remains a separate, optional high-risk epoch/mover project.

- Standalone CPU design simulator: [`tools/lab/radiance/heat-placement/heat_mock.mjs`](../../../../tools/lab/radiance/heat-placement/heat_mock.mjs), invoked with `node tools/lab/radiance/heat-placement/heat_mock.mjs`.
- Unittest integration: `python3 -m unittest tools/tests/radiance/test_heat_placement_mock.py` (skips without Node).
- Mock methodology: [`tools/lab/radiance/heat-placement/README.md`](../../../../tools/lab/radiance/heat-placement/README.md).
- Evidence: [`docs/evidence/RHP01/mock-results.json`](../../../evidence/RHP01/mock-results.json): **23/23 pure CPU simulated assertions**. Not a dev GPU or model-quality result.

### Implementation handoff sequence

1. Work on a branch from `main` after verifying `RAD09` and `rad0002` are reconciled. Inspect upstream at the pinned SHA; allocate a free engine-scoped patch package; never edit unpinned vendor source directly.
2. Begin RHP01 by collecting issued/completed/credited/overwritten counts in existing `Ctx` route banks and reproduce the 2% long-prefill capture observation. Decide if evidence is merely late or actually dropped. **Do not add a second GPU histogram**.
3. RHP02 adds read-only engine hook once per credited row and tests exact multiplicity/phase attribution. Retain legacy diagnostic only until the shared producer passes parity, then remove old collection path without aliases.
4. RHP03 publishes bounded metrics and a verified atomic full owner map snapshot; RHP04 scores bounded CPU-only owner candidates against measured per-phase cost curves in shadow mode. Preserve existing local `HeatEngine` and `Mover` semantics.
5. RHP05 evaluates on OR-count/time trigger and stages proposals without disruption; apply under explicit restart authority and canary/rollback gates. RHP06 is gated by evidence that controlled restart cost is materially too high.

### Design caveats requiring confirmation

The pinned engine's route bank may aggregate multiple MTP passes and its histogram phase might be mixed/unknown; do not create fake per-phase counts. Initial all-VRAM / remote-only plans can disable the HeatEngine: the shared reader must not unintentionally activate weight movement. A helper-owned expert marked unplaced in the engine cannot be moved local without weight reallocation and loading. Rank 0-only deduplication is valid **only** for confirmed mirrored TP routing; never discard genuine partitioned routing data. Placement snapshots and the mock's checksum are not production crypto/storage code. GPU parity, real histogram retention and throughput remain unverified.

## Review 2026-10-31 (GPT, against radiance 1.14.0 4a61d72 and the code after RAD23 / RCS07 / libr2)

Pinned Radiance is now 1.14.0 (the pin above is stale). What changed under the plan: the remote-MoE protocol is v19 with up to four helpers
(`kMaxHelpers = kMaxSlots = 4`, engine ranks 2, `Header::slot_count`, `result_index`, `out_rank_of`); placement today is static
(`R3_MOE_WORKER_HEAT` recorded 2026-10-17, `R3_MOE_WORKER_SHARES`, `R3_MOE_WORKER_PLACEMENT=tail` with `R3_MOE_WORKER_MIBS`; `worker_share::plan_tail` is a pure
deterministic planner and the static reference); the RX 6900 XT (libr2) runs as a third helper. Findings that bind the plan:
- Heat is not placement benefit: a 14 GiB tail cut engine transfers and cost 12-15% prefill. The objective is per-layer critical-path completion (max over helpers), decode and prefill separately, with link and movement cost.
- The static heat file has a coverage bias (0.68% / 1.7% / 14% tail traffic by workload; layer 48 absent). Missing observations are not zero demand; a coverage gate precedes any ownership change.
- plan_tail / select_owned run once at start in each helper process from the same environment; a dynamic system needs one canonical proposal, immutable global expert ids, a map digest and acknowledgement. A mailbox request sequence is not a placement epoch (RHP11).
- Restart stays the first mutation boundary. Deterministic mode freezes ownership. A helper that held uniquely owned experts disappearing fails closed and restarts from a verified complete map.
- Collect with the existing HeatEngine feed (Engine::tiering_dispatch, `observe` weight = report multiplicity, decay per credited step): no second GPU collector, no per-callback decay.

Order: RHP10 and RHP09 first (is thrash real; what does the 6900 XT cost), then RHP01, RHP02, RHP08 (reliable heat), then RHP04 offline, RHP03, RHP05, RHP07. RHP06 and RHP11 only if
measurements justify live migration. Seed in this slice: `engines/radiance/components/moe-remote/worker_heat.h` and its test (decayed heat, greedy hysteresis planner with budgets, move cap, delta; not
connected to any live ownership). Not verified by the reviewer: the full RCS07 / RCS10 / RAD23 text, every libr2 call path, the remote hook and worker execution paths.

