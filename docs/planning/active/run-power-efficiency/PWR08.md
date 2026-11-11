---
id: PWR08
order: 0
plan: run-power-efficiency
state: in_progress
created-at: '2026-11-10T13:32:38.583551+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# PWR08: named power profiles in the serving manifest (bigcherry/v2)

## Description

Owner decision (2026-11-10): the selected power profile is part of the serving manifest. GPT review (req_main_eb983900e21b) established that serving-manifest v1 is closed (spec.md section 8: any added field is bigcherry/v2; helper runtime schema is closed; the strict reader rejects unknown fields), so this is a v2 change, migrated producer/readers/plugins/fixtures/docs together, no v1 translator. Minimal v2 surface: spec.intent.power {profile, overrides by role} (operator intent, class policy) and spec.plan.power {mode observe|apply, profile, resolved per-role/BDF settings} (derived by the C++ resolver from layout roles, profile, host overrides and, for selection=calibrated, a qualified calibration record). Not in the manifest: live sensors, applied values, receipts (run evidence, PWR07/PWR09). runtime.power_save stays where it is with its meaning. Named reusable profiles live in config/power-profiles.toml (stock/efficient/performance/quiet, role-level, no BDFs); host calibration by BDF stays in environment.local.toml; measured records under BC_WORK_ROOT/power-calibration keyed by BDF, driver, topology, workload, build. A cached measurement never activates itself: selection needs explicit policy intent.

## Steps

1. Decide scope with the other known v1 gaps (RSC09 launcher literals, RSC10 margin refit) so there is ONE v2 migration, not several. 2. Registry successor of serving-v1.toml, generated schema/codec, field reference. 3. Profile file schema and role resolution in the C++ resolver (engine, helper head, helper tail from tier; per-BDF overrides; unknown role = error; duplicate BDF = error). 4. Provenance records through the existing manifest_writer pointer mapping. 5. Shared reader, semantic validation, fixtures, negative tests. 6. State the memory-plan cache key's relationship to power (power does not change memory geometry; performance calibration keys include the resolved power plan). 7. Migrate every consumer in one change (bc-serve, engine, plugins, launcher plan-check, tools tests, docs).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Card-free: the same resolved power plan appears in unmeasured and measured manifests; changing the profile file after manifest generation changes nothing that is applied; unknown/duplicate roles and missing BDFs fail resolution; observation mode has an explicit empty settings map per role; registry fingerprints regenerate and CI generator-diff passes; all consumers reject v1 and accept v2 (no translator).

## Effort & Risk



## Standards

No legacy / no backward compatibility: v2 replaces v1, no aliases.

## Acceptance Criteria



## Notes

Depends on PWR07 (probe) and feeds PWR09. Challenged against GPT's proposal: keep spec.facts.power out of the first v2 unless the resolver needs a typed capability fact to refuse a profile (decide in step 1; stable driver/range metadata only, never an authority to write: the controller re-probes immediately before applying). CPU EPP/governor and non-AMDGPU intent can be designed in the schema but stay observe-only until a CPU-policy controller and ownership model exist. The v2 cost is high (producer, readers, plugins, bindings, fixtures, docs together): do it once, with every other pending v1 gap.

## Change Log

- 2026-11-10T13:32:38.583551+00:00 (created-by): Created by agent


## Update 2026-11-10 (PWR06 R9700 GFXOFF finding)

Kernel and firmware are host prerequisites, never manifest settings (PWR06: the GFXOFF fix is an amdgpu driver change). A proven runtime optimisation for the R9700 idle floor, if any, becomes a profile control here.


## Implementation status: 2026-11-10

Merged: #824 named profile file/reader with stock, efficient, quiet, performance and the data-only XTX -125 mV candidate; #825 inactive per-BDF resolver with qualified-selection and fresh-probe gates. Both are observe-first. The XTX -125 mV candidate needs a same-session A/B and explicit qualifying record before apply; no R9700 or 6900 XT optimum is validated. Still blocked: S9 v2 registry/generator, S10 C++ resolver/writer integration and S13 single active v2 migration. The Python supervisor S12 only consumes a validated v2 power plan and does not activate profiles.

## Superseding qualification decision for XTX -125 mV — 2026-11-11

#824's “data-only XTX -125 mV candidate” is historical provenance only. After #832 and the raw `runs/xtx-undervolt-ab/` ABBA (current build **190–191 W stock vs 192–196 W UV**, unchanged 325–326 decode tok/s), **-125 mV is not a validated current-build power-saving profile**, must not be an apply-mode override or default, and must not be promoted by a cached October soak. The names stock/efficient/quiet/performance remain **observe-first**, with no non-stock XTX winning values. The docs/reference evidence appendix supersedes the earlier “needs a same-session A/B” line: that A/B was **done and negative**, so the next gate is a *new setting and new qualifying A/B*.

Ranked owner-run candidates: fresh phase-isolated stock baseline → SCLK max 2900/2800/2700 MHz at 0 mV → same speed-safe SCLK with -125 mV vs clock alone → conditional 290/280/272 W cap when actually near a measured cap → read-only DPM diagnostics → deferred per-phase policy. All require per-device preflight, speed-equivalence for **both** decode and prefill, better integrated J/token, second-session replication and explicit selection plus matching v2 plan/receipt/build/driver evidence. No new candidate is qualified by this documentation slice. Dependencies S9/S10/S13 v2 cutover remain unchanged.


## Qualification ordering after lead XTX power addendum (2026-11-11)

The initial validation gate is **not** another clock sweep on an unmatched six-stream workload. First investigate stock power: October PWR01 had one-stream 154–158 W/card and current six-stream stock 190–191 W/card. Run the same **October PWR01 single-stream decode/long-prefill/idle protocol with AMD-SMI `SOCKET_POWER` on the current build** in stock and -125 mV separate-lease A/B arms, then a legal stock-decode-clock pinned **clock-only** arm paired with **clock+UV**. The source October launcher already enabled MTP depth 3 and 22528 MiB per XTX; added tail, source changes, concurrency and sensor require isolation. No October-only “-125 validated” profile may become apply-mode: current-build A/B, separate decode and prefill gates, sensor concordance and exact during-decode offset and clock readbacks are prerequisites. Existing S9/S10/S13 v2 cutover blockers remain.

## 2026-11-11 candidate named XTX profiles — evidence before values

The current-build `-125 mV` **offset alone** is still withdrawn as a power-saving *apply* setting; the newer sweep shows a **different** mechanism under a fixed SCLK maximum. Proposed, **inactive/not qualified** name: `xtx-cap2700-uv125` (here “cap” means max **OD_SCLK 2700 MHz**, **not** board power cap) for the **two measured XTX BDF roles only**, offset -125 mV. Discovery row IDs from `runs/pwr-sclk/c2700uv125`: `pwr-phases@1=587b81b5fa3b4835`, `prefill-doc-44k@1=b69a3764952f49b2`, `decode-6stream@1=d9177d830ee64d89`. Matched SCLK-only `s2700` comparator rows: `9331ee86e4b34adc`, `8e2d6ec68b3f4196`, `706d606817fe4288`. **These are discovery runs and cannot serve as a qualifying profile-selection record.** No profile file or v1/v2 resolver is activated by this docs slice.

Power-profile rule for S9/S10/S13: each per-BDF profile setting must be **sourced from a separate explicit validating run**, not copied from this survey, historical October offset or third-party evidence; bind qualification to actual device identity, model/build/kernel, preflight driver/probe, both OD readbacks, per-phase decode and prefill no-regression and energy, output identity, repeated control arms, soak, host amdgpu log and restored state. Until then `stock/efficient/quiet/performance` remain observe-first, no automatic XTX settings. R9700 and 6900 XT did not change in the sweep and receive no new candidate apply values. The 2700/-125 40-minute soak at `runs/pwr-sclk/soak27uv125` is **pending**, as are output identity and host dmesg. See the complete separate-phase table and V/F stress medians in `docs/reference/power-tuning-evidence-2026-11.md`.
