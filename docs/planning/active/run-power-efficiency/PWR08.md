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

# Owner-only YAML power profiles: per-device resolution, qualification and measured reuse

## Description

Finish missing nonstock production policy in the no-launch-manifest architecture. Native deployment x-bigcherry names profile; one reusable profile YAML and site YAML own authored settings/physical overrides. Reuse existing PowerPlan/Tuning/SelectedGpu and S10 pure per-device math, not v3 launch schema. RSC02 applies via existing transaction, PWR07 owns probe/receipt, PWR10 qualifies minimum cap. No interim manifest-based implementation.

## Steps

Migrate config/power-profiles.toml and serving DevicePower/site inputs to single YAML authority with RSC01, migrate all readers, retire --power-profile override in favor of YAML selector. Resolve each selected BDF independently with actual capability and matching trusted qualification; preserve voltage/clock/memory/cap controls and reversible gfx1201 offset only after backend gate. Bind effective policy to owner session and performance/calibration/cache identity, fresh admission for named starts. RSC02 applies before measure and all GPU children, restores exact original under lease. PWR10 first qualified cap-only profile, paired clock/UV and broader PWR02 later; no default activation without decision.

## Detailed Solution & Technical Design

docs/design/radiance-native-yaml-launch.md sections 2, 5 and 6. Extract existing S10 power resolution from Manifest and fix role-keyed mixed TP rejection. Return typed power::PowerPlan directly to SessionPower; no manifest-digest authorization, no new launch format. Exact per-BDF/profile/driver/kernel/firmware/model/build/topology/workload/actual settings qualification; stock observe default; apply intent plus explicit allow and verified AGENT_INSTANCE lease. Qualification evidence not YAML boolean. Same admitted plan for measurement/final, named cache reuse revalidates current effective policy; changed same-name settings invalidate relevant performance identity. PWR07 positive cap/original/readback and per-board support. Preserve PWR04/05 transferred gates; historical offsets not automatically applied. CPU observe-only until host controller. Lab research fenced and separate from production promotion.

## Code Samples & Guidance



## Files

engines/radiance/components/serving/{main.cpp,session.cpp,session.h,xbigcherry.cpp,serving.cpp,v2/s10_inputs.*,v2/s10_resolver.*,manifest_power.*,power_transaction.*,power_linux.*,power_records.*,plan_cache.*}; config/power-profiles.toml -> YAML replacement; tools/bigcherry/hardware/power_profiles.py and core/environment.py DevicePower readers; existing site policy, configs, design docs and tests. No new launch schema/contract.

## Validation

Reuse existing profile parser/resolver/transaction tests for selected BDFs, unsupported/unqualified settings, stock no writes and exact restoration. Add missing mixed-TP and same-name effective settings cases. One shared real PWR10 minimum-cap native launch/readback/restore under lease and relevant output/performance check; no full control sweep, historical retuning or new qualification framework.

## Effort & Risk



## Standards

No launching manifest, no duplicate YAML/TOML registry, no shim or legacy backport, no new supervisor/engine hardware writer. Run capture output only. Native YAML authority, explicit authorization and exact restore.

## Acceptance Criteria

**Transferred PWR04 acceptance (independent gate):** Carry original PWR04 config and validation semantics into the one v2 named-profile/power-plan authority, not a second environment-only or v1 power schema. Named profile and optional role/device overrides cover XTX VDDGFX offset, SCLK max, R9700 cap/offset and future CPU EPP as *observational intent only* until a tested CPU policy controller/lease ownership exists. Per-BDF qualification record must bind exact device, driver/kernel, model/build and workload, stability soak, unchanged output and measured joules/token, separately for decode and prefill; missing evidence cannot activate a setting. stock remains the absent-settings default. Unknown or duplicate BDF/role, unqualified calibration and stale profile file refuse application before writes, with explicit provenance in spec.plan.power. No hardcoded -125 mV or model defaults; the old observed negative XTX -125 mV result is not approval. Preserve PWR08 S9/S10/S13 v2 migration gates and keep measurements/receipts outside the manifest.

**Transferred PWR05 acceptance (independent gate):** PWR05 default-selection acceptance: any proposed flashnext-4gpu default power profile must be a named PWR08 v2 intent and qualified PWR07/PWR09 receipt-bound controller action; a generic launcher --gpu-power flag is not a substitute. The old PWR05 claim that -125 mV saved ~24 W per XTX is superseded by the later same-current-build negative comparison (190-191 W stock vs 192-196 W UV at ~325-326 decode tok/s), so do not activate -125 mV or claim efficiency. Requalify each XTX card and R9700 separately, hold model/compiled binaries/MTP/KV/layout/concurrency/driver/firmware constant, require phased 42K+8-session power and performance ABBA, identical output/quality, readback of real effective clocks/voltage and capability/restore success with integrated GPU+CPU joules/token, no regressions beyond measured CI. Only a newly demonstrated speed-safe/quality-safe energy win and an explicit per-role/BDF selection permit default-on; otherwise keep observe-only stock and record the negative no-promotion verdict. PWR02 owns power-cap sweeps; PWR06 owns gfx1201 kernel idle; PWR07/PWR09 own probe and transaction. No code defaults or second profile registry.

Native recovery gate: strict production nonstock profile source; mixed TP3 and two same-arch devices with different settings; no unknown/duplicate BDF/role or unsupported controls; apply with qualification matching actual identities and fresh probe, observe has no writes; gfx1100 offset requires paired clock ceiling; changing source/setting under same profile name invalidates stale effective policy and performance calibration; same sealed policy on fresh measurement and measured/replay start. Existing v3 schema unchanged, native YAML unchanged for native options, installed runtime Python-free. Explicit nonstock native foreground serve/stop under lease verifies exact apply and restore in RSC02/PWR10; historical settings not automatically activated. Card-free parser/resolver/session/cache/contract tests plus real native receipts, output and performance gates. Preserve all preceding acceptance until satisfied.

Native YAML selected profile resolves exact independent per-BDF settings, trusted qualification matches actual identities, no stale/unsupported writes, observe no writes, paired gfx1100 SCLK/voltage safety; same sealed effective policy through measurement/final, named cache revalidates. Existing C++ owner alone applies and restores exact original; real native --config launch with min cap under verified lease and output/performance proof, no old manifest input. All transferred PWR04/PWR05 gates above remain semantic requirements; provenance now typed owner/receipt, not launch schema.

## Notes

Depends on PWR07 (probe) and feeds PWR09. Challenged against GPT's proposal: keep spec.facts.power out of the first v2 unless the resolver needs a typed capability fact to refuse a profile (decide in step 1; stable driver/range metadata only, never an authority to write: the controller re-probes immediately before applying). CPU EPP/governor and non-AMDGPU intent can be designed in the schema but stay observe-only until a CPU-policy controller and ownership model exist. The v2 cost is high (producer, readers, plugins, bindings, fixtures, docs together): do it once, with every other pending v1 gap.

CONSOLIDATION 2026-11-22 POWER-LIFECYCLE: PWR04's original residual requirement is now owned here. PWR04's 2026-11-10 design explicitly delegates named profiles and configuration to PWR08; retaining an independent earlier host-profile schema would duplicate the v2 manifest authority. Neither this transfer nor previous observational power results satisfy live device, rollback or v2 migration acceptance.

CONSOLIDATION 2026-11-22 POWER-BASELINE: PWR05's unique acceptance is retained under this established owner; no hardware changes or runtime activation are implied.

2026-11-25 native recovery audit: the current generated bigcherry/v3 envelope already carries all five controls; native Radiance YAML remains the single authored model config. #891 left v2/s10_inputs.cpp:82–86 stock-only/observe, #895 supplied lease-held automatic settings in the old launcher, #1131 retired that launcher before nonstock resolution was connected; #1167 did not cause the original gap. Recent native serving receipts show stock/observe and empty per-device settings. s10_resolver.cpp power() also rejects differing settings on mixed-architecture TP engine ranks via a role-keyed map. PWR09 is superseded by the native owner (#1400); RSC02 now owns remaining transaction acceptance, WSL08 owns detached lease lifetime. Keep the historical evidence and old plan sections as provenance, not instructions to repeat the v2 migration. The historical 2900 MHz/-75 mV host selection and 2700/-125 discovery are not current qualifications. PWR10 is the first deployment candidate; PWR02 owns wider sweeps. No new manifest/version or duplicate model schema, no revived launcher.

2026-11-25 owner direction: no launch manifest, all authored serving configuration YAML, no compatibility shims or legacy backports. docs/design/radiance-native-yaml-launch.md supersedes earlier manifest-launch instructions; completed RUM04/05/06/09/12 evidence is historical, not proof of the new launch path. Existing capture manifests may remain output-only. No new programme or second resolver. Supersedes #1544-specific v3 contract-preservation and writer/replay implementation instructions. The stock-only and mixed TP findings, historical negative uncapped UV, exact qualification, PWR07 telemetry and RSC02 recovery remain. No new power plan owner.

2026-11-25 cross-programme bump review: docs/design/radiance-bump-integration.md is the shared implementation-order and compatibility specification; MEN24 owns sequence and timing, domain implementation remains here. No duplicate checker, bump driver, manifest launch or legacy fallback. Effective per-device power and qualification identity binds candidate runtime acceptance, performance/calibration keys and named measurement compatibility; do not invalidate unrelated compile cache. Prior profile evidence not silently accepted on changed context; stock no-write every bump, required nonstock profile fresh compatible qualification or explicit refusal, no fallback. Owner-only typed plan from YAML; PWR10/RSC11 exact candidate real restore evidence. Do not repeat every historical sweep or introduce manifest producer.

2026-11-25 second integration review: docs/design/radiance-integration-review-20261125.md is the exact-source review and controls conflicting earlier wording. Shipping pin remains b6b8751 (2.2.0), upstream observed main 1c104de4 is a development candidate, 119 commits and 327 changed files; no automatic repin or backport. No new compatibility layer, launch manifest, bump driver or identity machinery. Candidate native memory/queue changes affect performance and measured room, not power authority. Owner-only YAML profile and native PowerPlan independent of manifest/Fleet; no new hashes, retain existing receipt and actual build/profile/board compatibility. Required profile requalifies if context changed or refuses; stock no writes. No universal nonstock sweep on routine bump.

2026-11-25 serving/configuration closure sweep: docs/design/radiance-serve-config-review-20261125.md. Existing policy_calibration_inputs keys profile name only; same name can resolve changed actual caps/clocks/voltage/BDF overrides. Supply resolved ordered control settings and verified readback to RHP16 existing calibration eligibility, not merely profile label. Memory plan key only actual resource-affecting values, time/energy key separate. YAML selected profile replaces active CLI --power-profile, Gate E stock YAML and PWR01 lab arm edits; explicit apply consent/lease separate. Migrate private local device intent explicitly; no automatic reactivation.

Owner scope correction 2026-11-25: docs/design/radiance-native-yaml-launch.md section Scope and acceptance rule governs. Preserve behavior; smallest native launch and YAML policy change, no generalized machinery. Reuse existing tests/evidence, one shared dense/helper installed acceptance plus materially different affected paths only; cheap parser checks across approved YAMLs. Independent outstanding work stays with its owner, not a blanket S18 prerequisite. Keep existing voltage/clock/cap capability; use existing qualification and actual settings/readback, not extra security machinery. Paired undervolt research and any default promotion remain separate.

docs/design/radiance-native-yaml-launch.md §4.2 (code seam review, 2026-11-25): main.cpp constructs ManifestPower and session.cpp:1213–1307 admits power through Hooks.admit_power before any measuring child, including carried plan path. Keep lifecycle ordering and power_transaction implementation. Replace ManifestPower::begin(path,use)/verify_serving(path) and writer-derived power with existing typed PowerPlan selected once from YAML/actual BDFs; verify same effective settings on final launch, no reapply on host-pool convergence/cache hit. Fix stock-only s10_inputs.cpp and role-keyed mixed TP resolver in this extraction, not by extending manifest. Use real probe/qualification and exact readback, retain allow-power-apply and verified lease/children-exit; preflight inspector is device-free. One shared min-cap test with PWR10, no new power framework.

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
- RV4595
- 2026-11-25T01:39:04.091027+00:00 (updated-by): Updated: section:title, section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes
- RV4615
- 2026-11-25T02:09:24.099602+00:00 (updated-by): Updated: section:notes
- RV4646
- 2026-11-25T02:47:36.773505+00:00 (updated-by): Updated: section:notes
- RV4662
- 2026-11-25T03:28:39.913911+00:00 (updated-by): Updated: section:notes
- 2026-11-25T04:30:48.155358+00:00 (updated-by): Updated: section:validation, section:notes
- RV4675
- 2026-11-25T05:41:53.148203+00:00 (updated-by): Updated: section:notes
