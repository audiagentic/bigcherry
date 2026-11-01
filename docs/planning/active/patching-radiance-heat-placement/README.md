# RHP - live heat-driven expert placement roadmap

## Target

One canonical ownership map, one placement decision path, and one live activation mechanism across engine, host and helpers, without restarts. Layout (which cards help and where the engine runs) is an input. Calibration improves placement decisions, never how they are applied: all changes use live epochs. Returning to a known layout must meet its recorded baseline ([RHP12](RHP12.md)).

## Implemented foundation

Heat recording in pinned host memory (#500/#502), drift watch (#501), and heat-ordered start-up placement in `worker_owners.h` are done. Live changes are not, and neither is [RHP02](RHP02.md)'s shared histogram feed (the pinned-memory histogram is the interim source); `worker_heat.h` already has decayed heat and budgeted proposals.

## Stages

**Every stage passes [RHP12](RHP12.md)**, including return to previously recorded layouts.

| Stage | Items | Done when | Depends on |
| --- | --- | --- | --- |
| **1. Live helper re-placement** | [RHP11](RHP11.md) protocol v23; [RHP06](RHP06.md) migration; [RHP04](RHP04.md) stage 1 policy; [RHP07](RHP07.md) controls; [RHP02](RHP02.md) feed | Async host-to-VRAM staging, step-boundary epoch flip, safe old-slot retirement; replay, mailbox/epoch and forced-churn tests prove no lost/duplicate/stale work; paired 8-session prefill, 6-stream decode, 42K step-time A/Bs and quality checks show no degradation | RHP02 evidence; RHP11+RHP06 together; RHP12 |
| **2. Discovery and default-on calibration** | [RHP09](RHP09.md), [RCS10](../build-radiance-components/RCS10.md) | Supported layouts discovered, shared calibration schema, validated costs, topology-keyed cache hit under 0.5 s; default-on with opt-out; 4 actual rows in a 24-row lane within ~3% of 4-row execution | Stage 1, RHP12 |
| **3. Cost-aware solver** | RHP04 stage 3; RHP09 cost model | Disjoint budget-safe ownership with protected experts fixed; minimise layer critical-path max (not sum); bounded swaps, hysteresis and movement cost beat/match static-tail and heat-only A/Bs without quality loss | Stages 1-2, RHP12 |
| **4. Unified allocator** | [RHP13](RHP13.md), managed Radiance `core/place` patch | One allocator budgets engine VRAM, host and helper residency; protected slots and epoch safety preserved; no competing placement planners | Stages 1-3, RHP12 |
| **5. Adaptation** | RHP07 stage 5; RHP05 triggers | Request/time/drift reevaluation uses the same live epochs; residency floors, gain thresholds, move caps and rollback prevent oscillation | Stages 1-4, RHP12 |

## Still-open supporting work

| Item | Role |
| --- | --- |
| [RHP01](RHP01.md) | Prove long-prefill histogram coverage, delayed vs overwritten banks, exact credit and bounded overflow without GPU waits. |
| [RHP03](RHP03.md) | Existing telemetry and bounded snapshots for the **epoch journal, recovery and comparisons**, not restart activation: generations, coverage, budgets, gains, rollback. |
| [RHP08](RHP08.md) | Tail heat largely answered by drift watch and live-histogram comparisons; close with coverage/parity evidence or explicit gaps. |
| [RHP10](RHP10.md) | Stage 1 long-context thrash validation: distinguish KV/attention growth from residency using step time, promotions and host traffic, not token/s alone. |

## Superseded / folded

- [RHP05](../../completed/patching-radiance-heat-placement/RHP05.md): restart activation superseded; **stage A drift watch done and retained**, triggers reused in stage 5.
- `patching-runtime-placement/RPL01` -> RHP09 discovery/costs, RHP04 solver, RHP13 allocator.
- RHP07 tiered host cache -> RHP13; RHP07 retains controls and adaptation.

## Owner decisions (2026-11-01)

Live swap, not restart. Calibration default-on with opt-out. **Stage 1 first**, using current capacity/heat-order costs and proving no degradation before calibration. Layout-agnostic decisions, with no device architecture, ordinal, count or budget hard-coded. Returning to a known layout must pass RHP12 or restore known-good settings. Same-expert execution should be bit-identical; compare KLD when execution path changes.

## Policy boundaries

Deterministic mode keeps placement fixed. Score per **(model-global layer, model-global expert)** with rank/phase/coverage provenance; protected experts follow the runtime slot map. Keep scoring off the critical path. No SQLite/Prometheus-per-expert or raw prompts in persisted heat; bounded aggregates and epoch-journal snapshots suffice. Remote ownership is not a local Radiance tier: readiness, publication and retirement are coordinated across processes.
