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

# Native serving power profiles: per-device resolution, qualification and replay

## Description

Complete the missing native production power-profile resolver in the current serving architecture. Reuse the existing generated v3 PowerSettings contract and native Radiance YAML as the sole authored model configuration. Preserve the 2026-11-10 decision (#824/#825): named reusable profiles, host calibration by physical BDF, separate qualification and fresh capabilities, no cached measurement authorizing itself. The historical v1-to-v2 cutover has shipped and must not be repeated. The C++ owner applies the resolved settings via RSC02; PWR07 owns probe and receipts.

## Steps

1. Connect existing named profile/site inputs to the C++ owner; seal one normalized policy snapshot. 2. Resolve concrete controls independently for each selected physical BDF, including mixed TP3; validate qualification and fresh capability. 3. Preserve existing contract, bind effective policy identity into cache and replay, and record provenance without duplicating native model options. 4. Exercise measuring, measured and stored starts with the same admitted policy; integrate RSC02 transaction and PWR07 receipts. 5. Qualify PWR10 minimum-cap profile, then other candidates through PWR02. Explicit promotion decision, no automatic historical offsets.

## Detailed Solution & Technical Design

Native recovery (supersedes obsolete cutover instructions): (1) Inspect existing --site/--power-profile sources and installed bundle; reuse config/power-profiles.toml and the existing per-BDF host policy semantics, with one explicit source authority and deterministic rule/override precedence. Confirm native launch selector support rather than assume a YAML key; if required review only a minimal typed profile-name reference in existing x-bigcherry extension. Do not duplicate native model fields. (2) Thread a sealed, typed policy snapshot through WriterInputs, s10_inputs and the native owner measuring/measured writers and stored replay. Use the existing C++ LinuxHardware probe and trusted qualification records, not a parallel Python runtime resolver. Resolve each selected physical BDF independently, preserving TP rank/helper order, including two gfx1100 and one gfx1201 with distinct settings. Resolve min-cap selector against fresh capabilities to concrete integer W. Keep the existing v3 PowerSettings fields and reader; preserve compatible profile digest semantics, bind effective settings/source and qualification identity in owner cache/receipts where possible; do not silently change wire digest meaning. Stop for design review if the current wire contract truly cannot express it. (3) Exact per-device qualification includes model/build, kernel/driver/firmware, topology, workload, actual settings, output/quality and phase energy evidence. Compare identities, not mere nonempty strings. Stock absent selection observes; explicit nonstock apply requires qualification, --allow-power-apply and verified owner lease; no silent fallback. Research-only arms may use existing fenced transaction under explicit authorization without claiming production qualification. (4) Bind effective policy across measurement, final plan and replay; invalidate performance calibration when settings change under same name; memory-plan reuse only with fresh admission and compatibility. Delegate application/restore to existing RSC02 transaction; do not expand engine hardware controls. Explicit operator decision required before any default-on policy.

## Code Samples & Guidance



## Files

engines/radiance/components/serving/{main.cpp,session.cpp,session.h,manifest_writer.cpp,stored_manifest_cache.cpp,stored_manifest.cpp,v2/s10_inputs.h,v2/s10_inputs.cpp,v2/s10_resolver.h,v2/s10_resolver.cpp,v2/p1_producer.cpp}; config/power-profiles.toml; tools/bigcherry/hardware/power_profiles.py (reference parser); existing host/site input; docs/design/radiance-native-config-x-bigcherry.md; docs/reference/engines/radiance/SERVING_MANIFEST.md. Contract/serving.toml is existing authority, not a new schema task.

## Validation

Card-free: the same resolved power plan appears in unmeasured and measured manifests; changing the profile file after manifest generation changes nothing that is applied; unknown/duplicate roles and missing BDFs fail resolution; observation mode has an explicit empty settings map per role; registry fingerprints regenerate and CI generator-diff passes; all consumers reject v1 and accept v2 (no translator).

## Effort & Risk



## Standards

No new manifest, manifest version, parallel model configuration, launcher or supervisor. Existing v3 contract unchanged; no repeat cutover, no legacy fallback. Native YAML authority retained. Any indispensable wire change requires separate design review.

## Acceptance Criteria

**Transferred PWR04 acceptance (independent gate):** Carry original PWR04 config and validation semantics into the one v2 named-profile/power-plan authority, not a second environment-only or v1 power schema. Named profile and optional role/device overrides cover XTX VDDGFX offset, SCLK max, R9700 cap/offset and future CPU EPP as *observational intent only* until a tested CPU policy controller/lease ownership exists. Per-BDF qualification record must bind exact device, driver/kernel, model/build and workload, stability soak, unchanged output and measured joules/token, separately for decode and prefill; missing evidence cannot activate a setting. stock remains the absent-settings default. Unknown or duplicate BDF/role, unqualified calibration and stale profile file refuse application before writes, with explicit provenance in spec.plan.power. No hardcoded -125 mV or model defaults; the old observed negative XTX -125 mV result is not approval. Preserve PWR08 S9/S10/S13 v2 migration gates and keep measurements/receipts outside the manifest.

**Transferred PWR05 acceptance (independent gate):** PWR05 default-selection acceptance: any proposed flashnext-4gpu default power profile must be a named PWR08 v2 intent and qualified PWR07/PWR09 receipt-bound controller action; a generic launcher --gpu-power flag is not a substitute. The old PWR05 claim that -125 mV saved ~24 W per XTX is superseded by the later same-current-build negative comparison (190-191 W stock vs 192-196 W UV at ~325-326 decode tok/s), so do not activate -125 mV or claim efficiency. Requalify each XTX card and R9700 separately, hold model/compiled binaries/MTP/KV/layout/concurrency/driver/firmware constant, require phased 42K+8-session power and performance ABBA, identical output/quality, readback of real effective clocks/voltage and capability/restore success with integrated GPU+CPU joules/token, no regressions beyond measured CI. Only a newly demonstrated speed-safe/quality-safe energy win and an explicit per-role/BDF selection permit default-on; otherwise keep observe-only stock and record the negative no-promotion verdict. PWR02 owns power-cap sweeps; PWR06 owns gfx1201 kernel idle; PWR07/PWR09 own probe and transaction. No code defaults or second profile registry.

Native recovery gate: strict production nonstock profile source; mixed TP3 and two same-arch devices with different settings; no unknown/duplicate BDF/role or unsupported controls; apply with qualification matching actual identities and fresh probe, observe has no writes; gfx1100 offset requires paired clock ceiling; changing source/setting under same profile name invalidates stale effective policy and performance calibration; same sealed policy on fresh measurement and measured/replay start. Existing v3 schema unchanged, native YAML unchanged for native options, installed runtime Python-free. Explicit nonstock native foreground serve/stop under lease verifies exact apply and restore in RSC02/PWR10; historical settings not automatically activated. Card-free parser/resolver/session/cache/contract tests plus real native receipts, output and performance gates. Preserve all preceding acceptance until satisfied.

## Notes

Depends on PWR07 (probe) and feeds PWR09. Challenged against GPT's proposal: keep spec.facts.power out of the first v2 unless the resolver needs a typed capability fact to refuse a profile (decide in step 1; stable driver/range metadata only, never an authority to write: the controller re-probes immediately before applying). CPU EPP/governor and non-AMDGPU intent can be designed in the schema but stay observe-only until a CPU-policy controller and ownership model exist. The v2 cost is high (producer, readers, plugins, bindings, fixtures, docs together): do it once, with every other pending v1 gap.

CONSOLIDATION 2026-11-22 POWER-LIFECYCLE: PWR04's original residual requirement is now owned here. PWR04's 2026-11-10 design explicitly delegates named profiles and configuration to PWR08; retaining an independent earlier host-profile schema would duplicate the v2 manifest authority. Neither this transfer nor previous observational power results satisfy live device, rollback or v2 migration acceptance.

CONSOLIDATION 2026-11-22 POWER-BASELINE: PWR05's unique acceptance is retained under this established owner; no hardware changes or runtime activation are implied.

2026-11-25 native recovery audit: the current generated bigcherry/v3 envelope already carries all five controls; native Radiance YAML remains the single authored model config. #891 left v2/s10_inputs.cpp:82–86 stock-only/observe, #895 supplied lease-held automatic settings in the old launcher, #1131 retired that launcher before nonstock resolution was connected; #1167 did not cause the original gap. Recent native serving receipts show stock/observe and empty per-device settings. s10_resolver.cpp power() also rejects differing settings on mixed-architecture TP engine ranks via a role-keyed map. PWR09 is superseded by the native owner (#1400); RSC02 now owns remaining transaction acceptance, WSL08 owns detached lease lifetime. Keep the historical evidence and old plan sections as provenance, not instructions to repeat the v2 migration. The historical 2900 MHz/-75 mV host selection and 2700/-125 discovery are not current qualifications. PWR10 is the first deployment candidate; PWR02 owns wider sweeps. No new manifest/version or duplicate model schema, no revived launcher.

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

## Reviews

- RV4546
- 2026-11-22T11:42:25.163641+00:00 (updated-by): Updated: section:notes, section:acceptance_criteria
- RV4550
- 2026-11-22T11:49:05.252354+00:00 (updated-by): Updated: section:notes, section:acceptance_criteria
- RV4576
- 2026-11-25T01:01:53.754975+00:00 (updated-by): Updated: section:title, section:notes, section:detailed_solution, section:files, section:acceptance_criteria
- 2026-11-25T01:10:09.371431+00:00 (updated-by): Updated: section:description, section:steps, section:standards
