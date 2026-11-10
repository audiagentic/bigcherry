---
id: RHP20
order: 0
plan: patching-radiance-heat-placement
state: pending
created-at: '2026-11-10T04:36:03.059770+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Live expert swapping and its observability verified on every card type and layout (gfx1201, gfx1100, gfx1030; head, tail, TP and engine-resident placement)

## Current evidence reconciliation (2026-11-10)

PR #805 rejects incompatible helper arrangements; it is not proof of supported movement on each card. The last-helper-as-tail heuristic must be replaced by the actual placement role.

## Description

Live migration (RHP11) and the policy (RHP16) were developed and measured on flashnext-4gpu: R9700 engine (gfx1201), two 7900 XTX helpers (gfx1100, head) and the RX 6900 XT tail (gfx1030, rows kernel, w4nl64 only, PCIe 3.0 x4). Nothing establishes that swapping and the policy hold on the other layouts (flashnext-2xtx, -2xtx-rdna2-tail, -2xtx-rdna2-head, -tp2xtx-r9700 where the R9700 is a helper and the XTX cards are the engine, -tp2xtx), nor which cards can receive which expert formats (gfx1030 holds w4nl64 only, so a swap between a tail and a head helper must not move fp8 experts to it). Observability must be per card: which device owns what, moves committed and rejected with reasons, bytes and time, and the policy state.

## Steps

1. Matrix: layout x card type x role (head helper, tail helper, engine-resident, TP helper). For each cell record whether live migration is enabled, the formats movable, the budget rule (reserve-mib), and whether a calibration record exists.
2. Run the RHP11 forced-move harness on every cell: swaps within a card type and across card types (gfx1100 <-> gfx1030 for w4nl64 experts), zero-owner destination, MTP and 42K prefill in flight, >=1,000 forced transitions; bit-identical output before and after each move for the same expert; fail closed where a format cannot move.
3. Per-card observability in the manifest and logs (RHP18): owner counts and bytes per device, policy state, last N transactions with outcome and reason, live_policy_state (on/off with reason), and a heat-coverage counter.
4. Calibrate each cell that serves (RHP19) and record the key it is stored under.
5. For layouts where the policy cannot run (engine-only, dense, formats that cannot move), the manifest says so explicitly.
6. Regression gate (RHP12): per-layout step-time A/B and KLD where numerics change; return to a known layout must match its baseline.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

A table in docs/reference/engines/radiance/LIVE_PLACEMENT.md with one row per cell (enabled / disabled with the reason / calibrated key / harness run id / on-off A/B result). Harness logs per cell: transitions, rejections, bit-identity checks. Unit tests for the format-compatibility rule in the owner-map planner (verify_owner_map).

## Effort & Risk



## Standards



## Acceptance Criteria

Every supported layout either runs live swapping with a passing forced-move run and a calibration record, or is marked unsupported with the reason in its manifest; no start can be on or off without saying which; cross-architecture swaps never place an unsupported format.

## Notes

Card types: engines/radiance/archs.toml (gfx1201 libr4d, gfx1100 libr3, gfx1030 libr2 rows-kernel). Layouts: engines/radiance/layouts.toml. Related RHP18, RHP19, RHP12, RHP11.

GATEWAY REVIEW 2026-11-10: (1) worker_heat.h:125-149 treats the LAST helper as the tail (worker_share::plan_tail) - right for flashnext-4gpu and -2xtx-rdna2-tail, wrong for flashnext-2xtx (no tail: the last XTX would become one), -2xtx-rdna2-head (head-share strategy overridden), -tp2xtx-r9700 (single head helper) and -tp2xtx. Until the planner is topology-aware the policy must refuse those layouts at start with a reason. (2) v1/manifest.cpp:384-397 rejects live policy on any TP engine (tp != 1) although live_policy_manifest.cpp:22-25 says the rank-0 histogram supports TP: validate rank-0 heat completeness and both ranks' generation/consumption, then lift deliberately. (3) Destination eligibility is checked optimistically at proposal time (worker_policy_driver.h:245-265); resolve format/arch/budget compatibility before proposing (fp8_e4m3_block cannot move to gfx1030, which holds w4nl64 only) and keep the receiver's authoritative checks. (4) Cross-architecture moves are not bit-identical (different kernels): require the KL/numerical gates, not bit-identity. (5) Engine-resident experts (the engine's 'heat engine' movable units) are NOT candidates of the helper-to-helper policy (worker_policy_input.h:165-175 fixes engine-local owners); engine<->helper movement needs its own ownership/residency design. (6) Existing hardware evidence: 131 paced swaps committed during scoring, 0 aborts, KL passing, on flashnext-4gpu only (RHP06.md); gfx1201-as-helper and TP2 transitions are untested. Matrix rows needed: flashnext-4gpu, -2xtx-rdna2-tail, -2xtx-rdna2-head, -2xtx, -tp2xtx-r9700 (helper-to-helper not applicable), -tp2xtx, engine-only (not applicable).

## Change Log

- 2026-11-10T04:36:03.059770+00:00 (created-by): Created by agent
- 2026-11-10T05:24:04.600666+00:00 (updated-by): Updated: section:notes
