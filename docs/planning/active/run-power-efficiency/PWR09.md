---
id: PWR09
order: 0
plan: run-power-efficiency
state: in_progress
created-at: '2026-11-10T13:32:42.998106+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# PWR09: transactional Python power controller and supervisor integration

## Description

Python library (tools/bigcherry/hardware/power_transaction.py, power_receipt.py) so the strict apply is implemented once, never in bash: whole-layout preflight against fresh capability probes, selected-BDF only, gpu lease ownership verified (launch.lock is not the lease), durable journal (tmp, fsync, rename, dir fsync) before the first write, deterministic BDF order, exact readback per setting, reverse-order rollback, restore after verified process exit, journal retained on failure, a stale journal blocks later explicit applies but not observe-only starts. Applies exactly the manifest's spec.plan.power (PWR08) and rejects profile/BDF/digest divergence; the final measured manifest's power plan must equal the applied one or the launch aborts and rolls back. Receipt (separate from the manifest) binds to metadata.content_digest and records start/post_apply/pre_restore/stop observations. Until RSC02 exists the shell launcher gets only an adapter that calls this library; an explicit --gpu-power failure aborts before engine start (today it continues, launch-flashnext-best.sh ~385-388). No automatic restore after SIGKILL or host crash is promised: durable recovery instructions only; a watchdog is a later slice (candidate for the Fable session).

## Steps

1. GPU capability probe (PWR07 PR2): per-BDF driver/hwmon by name and parent, OverDrive parsing with device-reported ranges, per-control capability, ambiguous hwmon refused. 2. Receipt types and atomic persistence. 3. Transaction coordinator with an injected fault layer. 4. Route existing gpu-power apply/restore through it (remove the whole-inventory path). 5. Launcher adapter, strict failure, run-specific cleanup. 6. Benchmark per-cell receipt refs (MEN11). 7. Read-only host smoke on this host. 8. Manifest consumption once PWR08 lands.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Fault-injection matrix: second-device preflight failure = zero writes; write rejected after preflight = rollback; readback mismatch = rollback with expected vs observed; rollback failure = journal kept, no success claim; stale journal; unselected GPU never changed; lease not verified = explicit writes refused, observe-only allowed; observe-only receipt IO failure does not stop serving. Host smoke makes no write call.

## Effort & Risk



## Standards

No legacy / no backward compatibility.

## Acceptance Criteria



## Notes

Steps 1-7 do not need v2 and proceed first. Fake sysfs proves portability behaviour, not that a real driver accepts a command. Parent of the remaining PWR04 work.

## Change Log

- 2026-11-10T13:32:42.998106+00:00 (created-by): Created by agent


## Update 2026-11-10 (PWR06 R9700 GFXOFF finding)

Any R9700 idle optimisation proven by PWR06 is applied only through this controller (capability check, lease, rollback, performance validation); nothing in the shell launcher or a global environment flag.


## Implementation status: 2026-11-10

Merged: #815 PowerTransaction, rollback, stale-journal and durable receipt core; #816 routes gpu-power CLI and shell launcher explicit writes through the transaction and aborts on failure; #820 supervisor S6 lifecycle skeleton. S12 card-free wiring adds single-manifest power-plan consumption, pre-start PowerTransaction apply, verified-exit-only restore and digest-bound phase receipts. Production execution remains fail-closed while S11 measured-plan adapter and S9/S10/S13 v2 cutover are pending. Observe-only means zero writes even when a stale explicit journal exists; explicit apply refuses stale journals. No GPU or live power change has been validated by this slice.

## XTX recovery transaction/lease guardrails — 2026-11-11 (proposed runs, not executed)

Use the existing `runs/xtx-undervolt-ab/txn.py` transaction **pattern** for the lead's short single-arm leases; its present `apply` hardcodes -125 mV for both XTXs and cannot be reused unchanged for a clock or cap test. Implement/run-local fake-verified explicit `gpu_power.Setting(sclk_max_mhz=N, vddgfx_offset_mv=V, power_cap_w=P)` parameter selection; resolve only layout XTX BDFs; lease verification, fresh OD/cap min/max probe, durable journal, apply+readback before start, stop engine/helpers, then `PowerTransaction.recover(..., processes_exited=True)` and restore readback within the **same lease**. In amdgpu OverDrive, stage `s 1 N` then `vo V` then `c` once (committing SCLK can reset the offset); caps write `power1_cap` in microwatts and must restore the exact prior value. A recorded pre-arm cap of zero in `D1.json` is an ambiguity to investigate before any cap experiment, not a validated writable cap.

Do **not** write `power_dpm_force_performance_level`, `pp_dpm_sclk` or `pp_power_profile_mode` with this helper: they have no snapshot/rollback support in `PowerTransaction`. Observe first; add capability/restore/failure fakes before authorizing an arm. No live phase switching of OD values while kernels execute; phase-specific controls need an explicit quiescent-boundary supervisor design. Each candidate arm 8–12 min lease (15–20 min if isolated prefill/idle), fixed-build ABBA across four **separate** leases with separate decode/prefill regression gates and integrated phase J/token; failure/refusal always leaves or recovers durable evidence. Rank, knobs, measurements and no-regression thresholds: `docs/reference/power-tuning-evidence-2026-11.md`. No new setting is validated by this step; PWR09 production v2 S9/S10/S13 dependencies remain.


## Harness follow-up: corrected short-lease sequence (2026-11-11)

**No new live runs by the docs agent.** Audit of archived `xtx-undervolt-ab/{arm.sh,arm2.sh,watts.sh,summarise.py,txn.py}`: `arm.sh` has no deliberate benchmark warmup, `arm2.sh` only one 64-token warmup; both lack thermal matching and during-decode OD readback. Both lack fail-fast bench checks and unconditionally write `status=ok` after attempts; mixed-window watts in `arm.sh` cannot certify prefill energy, while `summarise.py` does not calculate W/J/token. The recovery wrapper passes `processes_exited=True` without independently verifying engine/helper PIDs. Correct all before future qualification, using fake-process and transaction tests, not unguarded sysfs writes.

**Owner-run arm schedule:** 0A stock `pwr01-baseline.sh 40` (no write); 0B -125 mV and same script with protected transaction/readback; 0C clock-only per-BDF legal OD max at measured current-stock decode clock, and 0D identical clock plus -125 mV, all separate short leases, with ABBA replication. Stage `s 1 N` then `vo -125` then `c` once, read back after warmup/within decode and after, verify all processes stopped, restore via PowerTransaction before lease release; reject zero/unreadable cap preflight, out-of-range SCLK, stale journal, bench JSON `ok=false`, output mismatch or failed restore. Keep original PWR01 prompts, phase order and AMD-SMI sensor, but log BDF-mapped hwmon average/input **simultaneously** as a read-only sidecar. Default `pwr01-baseline.sh` selects `rad_build`, so record and enforce the same compiled-build digest and launcher layout for each arm (no accidental rebuild between arms). An old 600-s-per-phase soak requires a long lease, not a short arm. Detailed gates and sequencing in November power reference lead addendum.

## 2026-11-11 — advisory GFX12 GFXOFF host check

PWR06's new sysfs-only detector is an observation, **not** a PowerTransaction setting. The serving supervisor captures the per-engine-BDF no-process floor just before engine launch and compares it after engine readiness, before the first request. It prints an advisory active/absent/ambiguous/unavailable line; probe and receipt failures do not affect serving or existing transaction restore ordering. Its result is attached to the pre-existing PWR07 `Receipt.observations["gfxoff"]` and run record. The live v1 shell launcher calls the same Python probe at the equivalent boundaries and keeps JSON in its run directory. Serving-manifest v1 remains closed; Python supervisor production activation is still gated on RSC11. See `docs/reference/serving-host-prerequisites.md`.

## 2026-11-11 sweep outcome, transaction boundaries and follow-ups

The new `runs/pwr-sclk` evidence includes per-arm **two-XTX OD readback** and immutable workload run IDs. At equal 2700 MHz SCLK, -125 mV appears promising, but **no apply-mode profile is qualified**, and `-125 mV` at *uncapped* SCLK remains withdrawn. Proposed `xtx-cap2700-uv125` must wait for completed 40-minute soak, independent output-identity check, host amdgpu error/reset inspection (container `dmesg` unavailable), replicated same-build stock/SCLK-only/combination controls, and readback + **verified process exit and restore** under PowerTransaction. The `soak27uv125` folder contained no completed status/result at inspection. No R9700 or 6900 XT control was changed by these arms.

The MCLK follow-up (reported XTX OD_MCLK 97–1250 MHz range) and pp_power_profile_mode tests (current BOOTUP_DEFAULT vs COMPUTE/POWER_SAVING) are **not writable via the present reversible Setting contract**: extend snapshot, capability/preflight, exact readback, fault-injection rollback and receipt support **before** any owner hardware experiment. R9700 minimum 210 W board-cap trial must separately verify real power-cap supported range and safe restoration. Record **phase-isolated decode and prefill** W/speed and true energy if sampled; the current pwr-phases `sysfs:power1_average` medians / same-workload tps are *GPU-only J/token proxies*, with no CPU energy or integration. See `docs/reference/power-tuning-evidence-2026-11.md`.
