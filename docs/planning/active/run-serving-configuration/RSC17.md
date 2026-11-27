---
id: RSC17
order: 0
plan: run-serving-configuration
state: pending
created-at: '2026-11-27T03:42:00.914197+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P1
---

# Shared KV startup allocation: measured per-rank capacity and native contract

## Description

RSC16 stage A/B owner. Diagnose and deliver exact per-rank shared KV allocation, independent L/S/C intent and versioned admission. Follow docs/design/radiance-shared-kv-capacity-code-guide.md A-D. No scheduler redesign or model-specific exceptions.

## Steps

1. Reproduce archived failure and matched long-profile 256K/512K variants; record actual calibration, per-rank ledger, rad0029 counterfactual. 2. Refactor existing own-split into pure candidate before admission; only extend beyond current algorithm where needed. 3. Typed native capacity, sealed rank/group facts and policy/version, legacy unchanged. 4. Validate final rank budgets before allocation and post-carve geometry; typed drift/refusal. 5. Patch chain and existing VTP33 regression plus synthetic boundary/shortfall; hand off to RSC18/RSC16.

- I1: Reuse native per-rank geometry and proposal; retain independent live verifier and exact pool rounding.
- I2: Trace retained host sizing to TierExec::slot_layout and HostArena; migrate scalar contracts/replay only after caller inventory, preserve caps and reserve/pin gates.
- I3: Version shared-mode calibration and reconcile native/BigCherry facts without duplicate allocator or second resolver.

## Detailed Solution & Technical Design

Code guide sections A-D, H and I1-I3 are normative implementation guidance. Consolidate one native geometry/proposal and independent live verification; reuse rad0029, rad0011 and existing TierExec::slot_layout for retained-host entry geometry. Keep active GPU, streamed mirror, cold retention, checkpoint, recurrent and weight ledgers separate. Migrate calibrated floor only in shared mode; preserve legacy and residency policy.

## Code Samples & Guidance

core/engine_bringup.cpp, mem/vram_budget.cpp, mem/kv.cpp, mem/prefix.cpp, runtime/tierexec.cpp, mem/hostarena.cpp; rad0029 and rad0011; serving resources.cpp, v2/native_measurement.cpp, p1_measurement.cpp, p1_producer.cpp, s8_rules.cpp, s8_replay.cpp, manifest_writer.cpp and contract/serving.toml. See guide I1-I3 for formulas and per-path validation.

## Files



## Validation

Existing config/fleet/mem tests and serving S5, measurement, receipt and cache tests; exact rounded boundaries, COW-independent capacity, unequal ranks, windows/state/streamed/fleet/unified or explicit unsupported refusal. Managed 256K/512K A/B facts; no GPU writes.

Guide I1-I3: test mutated rank facts/BDF/live room, exact pool boundary, replicated/sharded/zero group entry layout, Q-1/Q/Q+1 retained tokens, disabled/streamed/unified, overflow and actual HostArena rounding; host pin/cgroup/reserve one-unit shortfall, cache/contract stale identity, legacy fixture parity.

Guide J1/J3/J4: 65-token target at block 16 needs 5 pages, not 4; fixed==budget, invalid inverse, ID/offset bounds, exact cap/slot and late startup failure; prove preallocation gate, actual carve and ready receipt separately.

## Effort & Risk

Refactor only after measured failure diagnosis. Host entry geometry is required for truthful shared-mode retention; unsupported tier combinations refuse, not approximate. Verify actual production reachability before deleting old arithmetic.

## Standards



## Acceptance Criteria

Exact limiting rank/bytes identified; one full request or actionable refusal; maximal feasible shared geometry with measured guards; no S*L floor in shared mode; same logical rank tables, actual allocation matches sealed plan; legacy unchanged. Report unsupported combinations. Hand off to RSC18 and RSC16.

Shared mode has one authoritative measured geometry and allocation proposal; host retained slots are derived from whole-engine entry layout, not kv_one/L; independent verifier remains fail-closed; no silent calibration change or unqualified tier behavior.

Exact per-group and rank verified capacity; no claim based on approximate token inverse or unbacked address reservation; retained cap not exceeded; target source/build identity and evidence.

## Notes

Second review RSC16 second design review, code guide H: reuse validated rad0029/VTP33 own-split; common rad0011 refit refuses before that path. Instrument exact archived input and matched long profile, selected calibration (seed entries FlashNext only) and complete rank facts. Refactor existing solver before final admission; verify every rank before and after carve, no duplicate writer. Separate managed patch responsibilities, dependency/anchor/composition and legacy tests.

Consolidated cleanup ownership: I1 native allocation/geometry, I2 retained host sizing, I3 calibration and contract migration. RSC12 cache owner in progress; RSC15 remains deferred. No separate plan.

Fourth review (RSC16 guide J): exact target production source identity mandatory; reviewed materialization experimental. Use checked group block vector and carve reservation sequence, not paged_split scalar; preserve native host cap with floor(slot count), separate checkpoint want/floor/actual. Single-engine fixed-backed initial scope includes Qwen hybrid declared state/window groups. Lease for planning that allocates; no power writes.

## Change Log

- 2026-11-27T03:42:00.914197+00:00 (created-by): Created by agent
- 2026-11-27T03:47:49.501797+00:00 (updated-by): Updated: section:notes, section:steps
- 2026-11-27T03:48:39.685143+00:00 (updated-by): Updated: section:notes
- 2026-11-27T04:00:44.567682+00:00 (updated-by): Updated: section:detailed_solution, section:code_samples, section:effort_risk, section:notes
- 2026-11-27T04:01:17.824450+00:00 (updated-by): Updated: section:steps, section:validation, section:acceptance_criteria
- 2026-11-27T04:18:31.945889+00:00 (updated-by): Updated: section:notes, section:validation, section:acceptance_criteria
