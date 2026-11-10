---
id: RHP20
order: 0
plan: patching-radiance-heat-placement
state: in_progress
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

## Reconciled card-free status (2026-11-10)

The previous text overlooked delivered RHP11 epoch/migration infrastructure and #805 fail-closed derivation. This slice is **partial**: no GPU or lease was used.

1. **Matrix partial.** gfx1201/gfx1100 can handle w4nl64 and fp8_e4m3_block; gfx1030 handles w4nl64 only (archs.toml). #805 rejects unsupported helper arrangements at derivation. The new helper-side guard refuses an explicit live_policy with head-only/roundrobin, single helper, unbalanced/no-tail or TP2 engine configuration, before GPU init. It reads typed manifest handoff strategy and shares, not inferred hardware ordinals; fake tests cover head-head-tail, head-tail, missing tail, malformed shares and TP.
2. **Live moves partial.** RHP11 and RHP06 already delivered real epoch migrations on flashnext-4gpu, including 131 paced scored swaps with 0 aborts. New card-free three-helper tests check that a gfx1030 destination cannot receive an fp8 expert, that owner maps preserve hard per-card budgets, malformed incumbent eligibility fails closed, and engine-local/zero-helper experts remain fixed. Full layout/hardware matrix remains unrun.
3. **Per-card observability open.** There is no joined report keyed by BDF, role, expert format, owner generation and epoch, containing accepted/refused moves, bytes, time, reason and heat coverage. RHP18 runtime status is still incomplete.
4. **Calibrations open** outside the historically measured flashnext-4gpu key; each layout/budget must generate its own RHP19 record.
5. **Unsupported cells partly fail closed.** #805 provides derivation skip provenance, new helper guard also blocks manual explicit policy for unsupported placement/TP. Source-side per-destination format+ISA evidence is still absent: worker_policy_driver.h currently proposes optimistically and learns receiver refusals. The receiver's frozen-slot/format and budget checks are the authority, not the new fake test.
6. **Performance and quality open.** RHP12/KLD against current builds needed for each arm; cross-ISA output is numerical/KL gated, not bit-identical. RHP22 tracks the distinct known live-policy tail p99 KLD excess.

### Card/layout and format matrix (unverified rows remain disabled)

| Layout | Engine | Helpers | Live policy today | Hardware gap |
| --- | --- | --- | --- | --- |
| flashnext-4gpu | gfx1201 | 2x gfx1100 head, gfx1030 tail | Topology permitted, current-key calibration required | 1,000 paced move stress, per-card receipts, fresh KLD/latency |
| flashnext-2xtx-rdna2-tail | gfx1201 | 2x gfx1100 head, gfx1030 tail | Topology permitted, no verified calibration | 2 GiB tail, w4nl64 cross-ISA swaps, KLD |
| flashnext-2xtx-rdna2-tail-combine | gfx1201 | same as preceding, combine | Topology permitted, unverified | Combine on/off, epoch, KLD/latency |
| flashnext-2xtx-rdna2-head | gfx1201 | 2x gfx1100 head, gfx1030 head | OFF, no tail | OFF provenance; forced moves independent of policy |
| flashnext-2xtx, flashnext-2xtx-combine | gfx1201 | 2x gfx1100 head | OFF, no tail | OFF provenance, baseline and forced safety |
| flashnext-tp2xtx-r9700 | 2x gfx1100 TP engine | gfx1201 single head | OFF, no helper-to-helper migration | TP rank/owner checks, baseline |
| flashnext-tp2xtx | 2x gfx1100 TP engine | gfx1201 head, gfx1030 tail | OFF, v1 TP policy refused | Dual-rank ACK/generation/format/KLD proof |
| flashnext-r9700 | gfx1201 | none | NOT_APPLICABLE | Engine-only owner and known tight-VRAM start gate |
| qwen27b-2xtx | 2x gfx1100 dense | none | NOT_APPLICABLE | No-helper startup/TP baseline |

### GPU lease holder hardware acceptance, executable commands

From repository root, with an appropriate GPU lease held, set BUILD to the tested build root, PORT to a free local port, OUT to a fresh output directory, EXPERIMENT_JSON to the corresponding validated v1 forced-churn lab experiment, and KLD_CONFIG to the matching quality campaign. The earlier example EXPERIMENT_JSON shape is ${BC_WORK_ROOT}/runs/rhp06-churn-paced/experiment.json; its expert indices are **not** safe to copy to other layouts without verifying target formats and owner map. Run a separate, valid layout-specific experiment for each forced row; set repeat and workload to actually achieve >=1,000 committed transitions.

| Layout / cell | Start command (one serving start at a time) | Acceptance gate | Expected retained record |
| --- | --- | --- | --- |
| flashnext-4gpu ON | BC_EXPECT_LIVE_POLICY=on bash tools/lab/radiance/launch-flashnext-best.sh start --build "$BUILD" --port "$PORT" --layout flashnext-4gpu --record-period 1 | Derived policy ON/current key, >=1,000 forced or policy epochs in separate stress arm, no generation mismatch | ${BC_WORK_ROOT}/flashnext/manifest-*.json, helper logs, calibration key/store, per-card epoch receipts |
| flashnext-2xtx-rdna2-tail (2 GiB) | bash tools/lab/radiance/launch-flashnext-best.sh start --build "$BUILD" --port "$PORT" --layout flashnext-2xtx-rdna2-tail --experiment "$EXPERIMENT_JSON" --record-period 1 | >=1,000 legal w4nl64 swaps, zero fp8 destination on gfx1030, zero unhandled abort; cross-ISA KLD | Manifest, helper0-2 logs, source/target formats, KLD and RHP12 timings |
| flashnext-2xtx-rdna2-tail-combine | bash tools/lab/radiance/launch-flashnext-best.sh start --build "$BUILD" --port "$PORT" --layout flashnext-2xtx-rdna2-tail-combine --experiment "$EXPERIMENT_JSON" --record-period 1 | Same move/copy gates and combine work counter >0; compare with non-combine | Manifest, helper/epoch logs, combine count, paired KLD/timings |
| flashnext-2xtx-rdna2-head | BC_EXPECT_LIVE_POLICY=off bash tools/lab/radiance/launch-flashnext-best.sh start --build "$BUILD" --port "$PORT" --layout flashnext-2xtx-rdna2-head | Must explicitly state OFF/head-only; a forced-only experiment may be run separately but must not activate derived policy | Manifest placement/provenance, static baseline, forced-move log if tested |
| flashnext-2xtx and combine variant | BC_EXPECT_LIVE_POLICY=off bash tools/lab/radiance/launch-flashnext-best.sh start --build "$BUILD" --port "$PORT" --layout flashnext-2xtx | OFF/no tail, never report second XTX as tail; repeat with layout flashnext-2xtx-combine | Two manifest reasons, per-helper counts, RHP12 baseline |
| flashnext-tp2xtx-r9700 | BC_EXPECT_LIVE_POLICY=off bash tools/lab/radiance/launch-flashnext-best.sh start --build "$BUILD" --port "$PORT" --layout flashnext-tp2xtx-r9700 | OFF/single helper; both TP engine ranks agree owner generation | Manifest, rank0/rank1 epoch digests, gfx1201 helper log |
| flashnext-tp2xtx | BC_EXPECT_LIVE_POLICY=off bash tools/lab/radiance/launch-flashnext-best.sh start --build "$BUILD" --port "$PORT" --layout flashnext-tp2xtx | OFF/TP2; separate forced experiment proves both rank ACKs, format/KLD before policy unblocks | Manifest, both engine-rank logs, gfx1201/gfx1030 helper epoch logs |
| flashnext-r9700 | BC_EXPECT_LIVE_POLICY=not_applicable bash tools/lab/radiance/launch-flashnext-best.sh start --build "$BUILD" --port "$PORT" --layout flashnext-r9700 | NOT_APPLICABLE; no helper-owner migration; separately record known ~1 MiB VRAM preflight issue if encountered | Manifest or capacity-refusal record, fixed engine-resident map |
| Forced in-flight workload on each eligible harness cell | PYTHONPATH=tools python -m bigcherry stream-bench --url "http://127.0.0.1:$PORT" --streams 6 --prompt-tokens 2048 --max-tokens 1024 --reps 12 --temperature 0 --seed 7 --output "$OUT/soak.json" | Must overlap forced swaps and exceed >=1,000 committed, zero unhandled faults; verify MTP/42K separately, not implied by this 2K workload | $OUT/soak.json, migration counters, epoch digest before/after and time series |
| Numerical quality per cross-ISA or combine arm | PYTHONPATH=tools python -m bigcherry.campaign.benchmark --server-config "$KLD_CONFIG" --output "$OUT/kld" --pairs 2 | RHP12/RHP22 KL mean/p99/p99.9/top1 and latency gates, fixed reference corpus; do not require bit-identical gfx1030 results | $OUT/kld, KLD campaign config, each arm manifest and model hash |

Stop after each launcher start using bash tools/lab/radiance/launch-flashnext-best.sh stop and confirm all helper and engine PIDs exited before the next row. The forced work and KLD commands are additional acceptance steps, not evidence that a given layout has already passed. A forced experiment for TP or zero-owner must be independently validated before launch; do not invent unsupported flags to make them run.

The card-free tests in worker_policy_config_test.cpp and worker_heat_test.cpp verify the helper guard and fake eligible owner-map invariants. Record real run IDs, store keys, coverage, per-card owners, and verdicts in docs/reference/engines/radiance/LIVE_PLACEMENT.md before marking this item complete.

## Change Log

- 2026-11-10T04:36:03.059770+00:00 (created-by): Created by agent
- 2026-11-10T05:24:04.600666+00:00 (updated-by): Updated: section:notes
