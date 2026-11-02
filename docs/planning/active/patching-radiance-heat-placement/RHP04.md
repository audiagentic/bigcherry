---
id: RHP04
order: 12
plan: patching-radiance-heat-placement
state: pending
created-at: '2026-10-15T01:35:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: L
---

# Placement policy: stage 1 heat/bottleneck balancing, stage 3 calibrated critical-path solver

## Description

One placement policy in two milestones, both producing complete validated owner maps applied only through RHP11/RHP06 epochs. Stage 1 (with RHP06/RHP11): make worker_heat.h's propose() produce useful live moves from RHP02 heat with today's inputs. Today it only moves when the destination's scalar cost is strictly lower, so with equal costs it never moves, and it has single moves only. Stage 3 (after RHP09): replace the scalar Device::cost with calibrated cost curves and minimise each layer's predicted critical path for a weighted decode/prefill mix under global VRAM budgets.

## Steps

1. Define per-`(layer,expert)` owner candidates and strict eligibility from `rad0002` **runtime slot map** and `R3_MOE_WORKER_SKIP_OWNED`, protected BF16 experts, MTP layers, `min/max_rows`, full-width/TP geometry, input/output precision and exact helper source file availability. Never propose a helper assignment for an unsupported layer, expert encoding, or step-size fallback.
2. Consume phase-tagged prefill and decode route counts (RHP02); compute rolling/decayed heat and uncertainty with coverage min/age requirements. Separate active tokens per expert per phase, fraction of steps selecting it and distinct expert reads; raw counts alone do not reflect GEMM/reload cost.
3. Calibrate per-device curves from existing `--profile-ops` and helper per-stage timings (`RAD02/RAD05/RAD06`): local R9700 GEMM + host-pool miss, gfx1100 small-lane GEMV, large-lane grouped GEMM, publish/stage/add/fence, VRAM pressure. Include overlapping local and remote work, helper skew, p95 layer wait and transfer costs. Freeze curve version/provenance.
4. Create a deterministic budget-constrained offline optimiser producing a *whole ownership map*: respect helper VRAM limits, engine host+VRAM weights, whole expert bytes, protected/local nonmovable weights; preserve disjoint helper ownership. Compare predicted layer critical path and total request latency; favour low churn and low transfer cost. Explicit `no-change` outcome.
5. Shadow-run candidate calculations on a background CPU task on *copies* of bounded snapshots. No GPU work, blocking hooks, file writes or map mutation in routing/dispatch. Record recommendation and residual ranking error, never apply. Test representative prefill-heavy, decode-heavy, mixed and shifted distributions.
6. Bench existing static heat-based layout versus offline plan with restart/manual ownership selection in balanced ABBA; report calibration error, p50/p95 wait, prefill, decode, MTP accepted tokens, KL, GPU memory and bytes transferred. Gate deployment on real advantage beyond run-to-run variance.

## Detailed Solution & Technical Design

Stage 1: per-(helper, layer) heat load L_h,l = sum of heat of owned experts; score a candidate by the reduction in the per-layer maximum weighted load beyond a deterministic margin; add bounded 2-way swaps as one atomic transaction; respect eligibility, protected experts, budgets, per-layer caps, minimum residency and the move budget; no_change is a normal outcome; static relative-cost factors only as initial weights. One controller fed by interval deltas of coherent RHP02 snapshots. Initial controls (RHP07): evaluate every 5 min, >=1,000 requests or equivalent observation, >=10% heuristic improvement, <=8 experts and <=256 MiB a transaction, one transaction at a time, 30 min minimum residency and failure cooldown; automatic activation off until stage 1 validation passes, then on. Stage 3: deterministic greedy construction from the active map, critical-path relief on the worst weighted layer/phase scenarios, bounded local search (single moves, pair swaps, small bundles) with per-layer aggregates and a global capacity ledger; objective sum over scenarios of predicted layer critical path + movement and uncertainty penalties; scenarios 1/3/6-stream decode, 128-256-row short prefill, 2048-row long prefill; guards e.g. <=2% predicted decode and <=5% TTFT regression for a mixed workload; the tail may own zero experts if that is best; solve budget 1-3 s, cap 10 s; exact oracle on small synthetic models to check the heuristic.

## Code Samples & Guidance



## Files

`engines/radiance/components/libr3/worker/` or a dedicated adjacent policy source with card-free tests; experiment runner `tools/lab/radiance/`; `engines/radiance/components/libr3/component.toml` for experimental default-off knobs; `docs/evidence/` for reports. Reuse RHP02 hook and RHP03 snapshots.

## Validation

Stage 1: card-free determinism, zero/stale heat, equal-cost balancing, per-layer caps, full devices, move and swap, protected experts, hard overflow, tie-breaking, move budget, no duplicate/incomplete owners; exhaustive oracle on a small synthetic model. Stage 3: held-out layer critical-path prediction within 10% p50 / 20% p95 (RHP09), offline gain over static placement on replayed traces, then RHP12 A/Bs.

## Effort & Risk

High: cold experts may incur disproportionately expensive gfx11 prefill kernels; route frequency alone may bias allocation to the wrong helper. Predictions are provisional until independent served A/B.

## Standards

- Source of truth: pinned standalone Radiance 1.3.0 `89cee7ce414486c717aa2085c99e1f522e4c0ea6`; BigCherry `engines/radiance/`. Engine edits MUST be packaged as `engines/radiance/patches/radNNNN_*/{patch.py,patch.toml,SUMMARY.md}` with code-line anchors (patcher strips comments), tests with pinned blobs; helper policy lives in `libr3moe`. Re-check patch IDs before allocation.
- Follow `engines/radiance/recipes.toml` and `components/libr3/component.toml`; runtime profiles set ONLY non-defaults. Use existing BigCherry quality/ABBA/campaign/evidence facilities; no parallel scheduler, telemetry backend, loader, or compatibility alias. Disabled behaviour byte/quality/performance equivalent.
- Related active work: `RAD02` (R9700 trunk + 2 XTX helpers), `RAD03` (GPU mailbox), `RAD06` (prefill MoE), `RAD09` (merge/reconcile helper and rad0002). Integration depends on verified `RAD09` baseline. Do not overwrite another agent's branch or modify a shared patch without merging latest main.

## Acceptance Criteria

Stage 1: valid non-trivial moves when heat drifts, no_change when stable, never an invalid map. Stage 3: applied only when the predicted gain's lower bound is above zero; measured gain beyond paired noise on RHP12 probes with no gate regression.

## Notes

Depends on RHP02–03; no automatic activation before RHP05.

Plan source: GPT staged plan 2026-11-01. The worker_heat.h header ("nothing here changes a live ownership map") is what stage 1 changes.

## Implementation-grade notes (2026-10-15)

### Objective and constraints

Let `r[l,e,p]` be routed rows or demand estimate for phase `p` (prefill/decode). For each layer and device `d`, predict `T[l,p,d]` using measured kernel curves (startup + fixed per-expert weight read + variable rows/tiles + mailbox/merge + possible host-pool misses). Minimise

```text
J(owners) = sum_phase demand_weight[phase]
              * sum_layer max_device(T[layer, phase, device])
          + relocation_penalty(owners,current)
          + uncertainty_penalty(coverage,fit_error)
```

Do **not** sum concurrent engine+helper work as serial time. Model communication on critical path including helper wait, and count a remote expert only once per layer where applicable. For mixed steps, qualify independent curves rather than interpolating unobserved tensor shapes.

Hard constraints: each `(layer,global_expert)` assigned exactly once; `allowed_owners` derives from runtime protected expert map, output semantics, `RAD_MOE_REMOTE_MIN/MAX_ROWS`, source weight location and GPU ISA; per-device resident byte <= measured safe budget including launch scratch, KV/headroom, helper kernels and shadow-transfer workspace; rank parity and top-k routing identity unchanged. `R3_MOE_WORKER_SKIP_OWNED` means a local candidate may need restart reload; no illusion that a bitmap change is free.

### Calibration data and sampling

Record per-layer actual selected expert IDs/rows, per-device active routes, top-k count, per-expert nonzero token rows, hist coverage and duplicate count, occupancy, GEMM shape and padded tile size, helper stage latency (route/publish/gate_up/quant/down/merge), engine local MoE time and time waiting on helper. Cross-link each curve to engine build SHA, libr3 package digest, GPU UUID, ROCm version, model and numerical mode. Fit simple piecewise monotone upper envelopes by phase/shape/device (not high-dimensional ML), with conservative p95 prediction error. If too few samples for a bucket, refuse automatic change.

### Bounded production policy versus oracle

The included `heat_mock.mjs::Planner` enumerates single moves and 2-way swaps on **small synthetic models only** to serve as a correctness oracle. Real Flash-Next has tens of thousands of expert units: use a bounded, stable, deterministic candidate shortlist by `heat * expected net ms` and per-layer helper imbalance, then limited pair or slot-preserving swaps with strict time budget. Stop after no positive marginal improvement, hitting max swaps, or planner budget. Never enumerate O(N²) swaps in the production dispatch path.

### Replay and acceptance matrix

Cold startup profile vs hot-ranked static ownership vs RHP04 recommended; compare same model prompts and routing windows in prefill-only, decode-only, mixed, MTP verify, protected layers and decode helper-disabled. Replay phase-shift mid-run: prefill-heavy to decode-heavy, alternated and uniform. Report predicted/actual total and p95 layer waits, operator estimate error, bytes loaded, churn, source heat coverage, KLD/top-1 and speed ABBA. Proposed 5% predicted threshold is a search parameter; production gate is independent measured benefit with a confidence interval and no material quality regression.

## Revised 2026-10-31 (GPT review of the plan against radiance 1.14.0 and RAD23; source claims checked)

Produce a deterministic whole-map ownership proposal for the current R9700 + gfx1100 + gfx1030 topology in shadow mode, compared with weighted round robin (R3_MOE_WORKER_SHARES) and plan_tail. Seed: worker_heat.h (decayed Heat, greedy hysteresis planner, delta) in this slice; the full optimiser is the per-layer critical path (max over helpers, not sum), not per-route cost.

Steps:
1. Reuse worker_owners.h::select_owned eligibility and exact laid-out sizes; protected and unsupported experts are fixed.
2. Consume decayed heat with explicit coverage and confidence.
3. Calibrate per-helper compute, transfer, queue and synchronisation costs, including the gfx1030 few-rows and grouped-GEMM regimes (RHP09).
4. Minimise predicted per-layer completion subject to budgets and disjoint ownership; add a bounded swap search (a full faster device cannot take an expert without giving one up), movement cost, minimum predicted gain, hysteresis, move cap.
5. Shadow runs against static layouts with ABBA replay and matched MTP work.

Validation: Exact static-tail reproduction (static_tail is plan_tail), deterministic ties, zero-budget helpers, mixed expert sizes, per-layer max-not-sum objective, insufficient-coverage rejection, calibration error, quality comparison.

## Change Log

- 2026-11-01T22:51:22.513641+00:00 (updated-by): Updated: section:title, order=12, priority='P0', section:description, section:detailed_solution, section:validation, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20261101_225732_planning-a-staged-roadmap-for_4092
- 2026-11-01T22:57:35.857193+00:00 (updated-by): Updated: section:ledger-events
- chg_20261102_022722_groundwork-for-live-expert-pla_5619
- 2026-11-02T02:27:34.555558+00:00 (updated-by): Updated: section:ledger-events
