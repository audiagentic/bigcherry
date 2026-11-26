---
id: PWR10
order: 0
plan: run-power-efficiency
state: superseded
created-at: '2026-11-25T00:42:29.487523+00:00'
breadth: ''
skill: intermediate
created-by: lead
work: M
priority: P1
---

# Qualify native minimum-cap launch profile and paired clock/voltage efficiency

## Description

A trial on 2020.11.4 (qwen27b, 3 decode requests and 3 prefills of 2,723 tokens per phase): the lowest writable power caps (272 W on the XTX and 6900 XT, 210 W on the R9700, against stock 303 W and 300 W) cut the XTX mean power from about 293 W to 267 W (peak about 305 W to 276 W) with decode 126.1 to 124.7 tokens a second (-1.1 percent, one run each) and prefill unchanged (2,492 to 2,511); the R9700 peak clipped from 228 to 215 W with no speed change. The cap resets at reboot and nothing applies it at start.

## Steps

1. After PWR08/RSC02 input and lifecycle recovery, select minimum-cap profile per launch using each selected device's reported minimum, restore prior caps at stop. 2. Repeat stock, XTX-only, R9700-only and all-selected minimum-cap trials, at least three runs per arm. 3. Qualify clock-only and paired voltage/cap controls under independent leases using PWR02/PWR07 evidence protocol, including quality and restoration; R9700 offset requires backend. 4. Deploy first qualified native cap-only profile without waiting for other research. Host idle investigation is PWR11, CPU PWR03, GFXOFF PWR06.

## Detailed Solution & Technical Design

PWR08/RSC02 deliver one native profile using device-reported minimum cap, only selected BDFs; no hardcoded caps or new manifest. PWR10 owns qualification, not a second writer. Repeat exact current dense workload, three runs per arm minimum, randomized/ABBA or BAAB blocks with matched warmup/thermal/cache conditions; separate XTX-only, R9700-only and all-selected minimum caps. Use existing leased lab path for research, and native owner for deployment acceptance. Then clock-only XTX 2900/2700/2400/2100 at 0 mV, paired voltage at matched clock/cap after prior stability pass (conservative steps, per physical card; retain historical -75/-125 as candidates, not defaults). R9700 offset only after reversible backend proof; broad MCLK/other architectures belong PWR02. Record separate decode/prefill throughput/energy, actual per-BDF clock/voltage/cap, latency, output/quality, kernel errors and restore. PWR01 instrumentation, PWR02 wider sweep. No idle savings claimed from cap; platform idle separate.

## Code Samples & Guidance



## Files



## Validation

Launch and stop leave caps at their previous values; decode and prefill within the noise of stock; power measured with amd-smi.

## Effort & Risk

Small. Risk: the cap shifts other agents' measurements while applied; apply only under the GPU lease.

## Standards



## Acceptance Criteria

Minimum-cap profile selectable per launch with measured watts and speed, caps restored at stop.

First deployable slice: native foreground launch and replay select qualified minimum-cap profile, independently resolve each participant, apply before model load/measurement, verify readbacks, leave unused GPU unchanged, stop/reap/restore original caps under lease. Repeat trial with phase-separated energy and stable identity, three runs per arm or more; 1.1% decode loss is not zero, require lead-approved non-regression/tolerance and quality before promotion. Preserve evidence if negative. Qualify clock/voltage combinations only with matched controls, independent per-card output/quality, thermal soak and recovery; no default-on without explicit decision. Do not require this research to block a qualified cap-only deployment.

Native YAML selected qualified cap-only profile applies before measurement/model load, independently resolves device minimum and exact readbacks, unused devices unchanged, exact restore before lease release. Named cache reuse must revalidate current policy and never launch from run manifest. Preserve previous repeated/phase-integrated, quality, tolerance and thermal/UV evidence gates; no default-on without explicit decision.

Additional gate for new clock/voltage settings only: stock-reference numeric integrity (actual same BDF and provider, deterministic repeated full output, no silent skip or unchecked failure), safe observed thermal envelope with normal cooling, transient and sustained production role/replay and memory integrity, matched measured phase energy and throughput, no unexpected mismatch/reset/timeout, exact restore before lease release. Thermal stop or insufficient coverage is inconclusive, never a pass. Existing cap-only gate remains independent; no external tuner dependency or automatic undervolt.

## Notes

Historical note (superseded by the clarification below): the then-active voltage offset was 0 mV, and the withdrawn undervolt was not to be re-added. The withdrawal applies to the unqualified automatic setting, not the supported voltage control or matched-clock research.

2026-11-25 correction: previous note withdrawing undervolt refers to unqualified historical settings, especially uncapped -125 mV which increased power; it does not remove supported voltage control or prohibit qualified matched-clock research. The native production resolver is stock-only and mixed TP per-role resolution rejects differing arch settings; PWR08/RSC02 own this prerequisite, PWR07 probes. Preserve single native YAML and existing v3 contract. No automatic promotion of historical 2900/-75 or 2700/-125. Owner trial: stock/min/stock decode 126.1/124.7/126.0; prefill 2492/2511/2494; active GPU mean sums 786/735/829 W; 72.5 W midpoint saving, 1.1% decode cost, 43 W stock drift. One run/arm, mixed windows, not integrated phase energy. Caps restored per owner. Raw samples and identity not independently recovered.

2026-11-25 owner direction: no launch manifest, all authored serving configuration YAML, no compatibility shims or legacy backports. docs/design/radiance-native-yaml-launch.md supersedes earlier manifest-launch instructions; completed RUM04/05/06/09/12 evidence is historical, not proof of the new launch path. Existing capture manifests may remain output-only. No new programme or second resolver. #1544 trial and qualification plan retained, but first deployable minimum-cap profile uses YAML selector and owner-only typed PowerPlan, not manifest. Native --config real foreground serve/stop and named measurement reuse, no --manifest input. Do not implement intermediate manifest-based nonstock serving. Preserve repeated controls, per-board clock/UV research, quality and exact restore.

2026-11-25 bump integration: docs/design/radiance-bump-integration.md and MEN24 gate order. Qualification receipts link exact candidate pin/build/bundle, native YAML, selected BDFs, kernel/driver/firmware, actual admitted controls and workload. MEN17/RSC11 use required candidate-compatible profile evidence (stock no-write for every bump; nonstock only where required/qualified), do not repeat every historical sweep. Actual native owner applies before measure/load and restores exact prior state after children exit; no implicit stock fallback on requested apply. If context changes and evidence no longer qualifies, refuse and requalify. Keep repeated min-cap and paired voltage/clock/quality gates and no default promotion.

2026-11-25 second integration review: docs/design/radiance-integration-review-20261125.md is the exact-source review and controls conflicting earlier wording. Shipping pin remains b6b8751 (2.2.0), upstream observed main 1c104de4 is a development candidate, 119 commits and 327 changed files; no automatic repin or backport. No new compatibility layer, launch manifest, bump driver or identity machinery. Preserve current discovery trial as separate cohort. First qualified native cap profile test on exact approved target and actual YAML/BDF, native reserve/queue changes can alter workload and power. Stock no-write every bump; nonstock only for intended required profile with compatible evidence or explicit refusal. No default activation, no replay all historical sweeps, exact child exit+restore under lease.

2026-11-25 serving/configuration closure sweep: docs/design/radiance-serve-config-review-20261125.md. Migrate PWR01 --power-profile CLI to one intended YAML change via arm_yaml, retain actual stock/readback/profile/build/model and explicit allow-power-apply/lease. Workload power phase receipts identify actual owner/session and endpoint, no cmdline manifest discovery. No qualification from unknown/stale owner or profile label alone; preserve existing cap and clock/UV gates and verified restore.

2026-11-26 roadmap reconciliation at 165d6f724: External reference: Sweet-Spot Tuner v3.5.15, commit a2a31ce63cc1d40af72997de621115eaa2dcd54d (https://gitlab.com/spawnitems-group/sweetspot-tuner/-/tree/v3.5.15); see docs/reference/power-tuning-evidence-2026-11.md, hot AI qualification section. First deployable cap-only slice unchanged; do not block native serving on undervolt research. Before qualifying any clock/voltage (or combined clock/voltage/cap) candidate, add a short fail-fast AI arithmetic screen and final sustained plus burst/transient and memory correctness under actual safe operating envelope. Use matched installed ROCm headers/libs and existing rad-kbench/replay/unit harness/actual kernels; optional focused 768x768 FP16 FP32-accum hipBLASLt test with fixed algorithm, only if supported; reuse existing lab HIP diagnostic pattern, do not add downloaded binaries or clone proprietary/hand-declared ABI. Generate deterministic reference at verified stock on the same physical card, validate stock repeat and independent CPU/reference where feasible; compare candidate full outputs to unchanged stock reference, check all HIP return codes, positive completed iterations/time and actual BDF, fail closed on skip. This catches consistent corruption the external self-reference misses. Use actual IQ8R/IQR/MoE/GDN/attention and transfers for roles, not FP16 alone, and check actual near-resident VRAM patterns using existing replay/copy checks with safe headroom; no arbitrary fill or OC. Screen only promising paired settings under lease, one bounded representative full inference/quality and warm/transition soak for finalist, exact restore. No fixed universal undervolt reserve or automated search to crash edge. Thermal limit -> stop/cool/qualified bounded repeat or inconclusive, never raise limit/fans to claim normal-policy stability. Do not promote from synthetic GFLOPS or stable games; require zero unexpected mismatches/faults and agreed throughput/latency/energy, per-card receipt, no defaults changed.

2026-11-26: the first real native minimum-cap apply passed on the installed 2020.21.1 (lab script run.sh with a self-checking result.txt; the evidence page carries it): the qualifying apply (--allow-power-apply --qualify-power under the gpu lease) set 272/272/210 W on the two XTX and the R9700 while serving (readback exact, 6900 XT untouched at 303 W), stop restored 303/303/300 W exactly, the receipt recorded apply, restored, 3 applied + 3 restored events and no failure events; a second start without --qualify-power was accepted by that receipt and applied/restored again; a start with a different cap (215 W on the R9700) was refused naming the missing receipt, nothing written. Found and fixed on the way: an installed release resolves NAME.yaml from its bundle before --configs, so the lab script now passes --native-config (the first attempt measured the bundled stock config); and a missing card-free end-to-end test of YAML profile + site file + allow flag reaching the admitted plan (#1664). Not run, by decision: the 3-arm decode/prefill A/B and the clock/voltage qualification (the research part of this item stays open).

Evidence: docs/evidence/2026-11-26-pwr10-rsc14-native-acceptance. Item stays open for its research part only (steps 2-4: stock/XTX-only/R9700-only/all-selected minimum-cap A/B with three runs per arm, clock-only and paired voltage qualification); the native apply/readback/restore path itself is accepted.

2026-11-27 consolidation (superseded): Native minimum-cap apply/readback/restore accepted #1673; transfer residual matched cap, clock and voltage research to PWR16. No duplicate campaign or claim of full pass. All earlier results remain historical; see power-session-closeout inventory.

## Change Log

- 2026-11-25T00:42:29.487523+00:00 (created-by): Created by lead
- 2026-11-25T00:42:52.701764+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria, section:notes

## Reviews

- RV4580
- 2026-11-25T01:04:09.048902+00:00 (updated-by): Updated: section:title, priority='P1', work='M', section:notes, section:detailed_solution, section:acceptance_criteria
- 2026-11-25T01:10:22.073565+00:00 (updated-by): Updated: section:steps, section:notes
- RV4597
- 2026-11-25T01:40:03.796527+00:00 (updated-by): Updated: section:notes, section:acceptance_criteria
- RV4624
- 2026-11-25T02:14:26.375451+00:00 (updated-by): Updated: section:notes
- RV4647
- 2026-11-25T02:48:07.107725+00:00 (updated-by): Updated: section:notes
- RV4663
- 2026-11-25T03:29:11.902957+00:00 (updated-by): Updated: section:notes
- 2026-11-25T22:03:05.477798+00:00 (updated-by): Updated: section:notes
- 2026-11-25T22:05:39.356405+00:00 (updated-by): Updated: section:acceptance_criteria
- 2026-11-26T05:43:32.075041+00:00 (updated-by): Updated: section:notes
- 2026-11-26T08:35:13.183700+00:00 (updated-by): Updated: section:notes
- 2026-11-26T08:36:15.464511+00:00 (state-transition): State: pending → in_progress
- RV4685
- 2026-11-26T23:16:05.519382+00:00 (updated-by): Updated: section:notes
- 2026-11-26T23:17:34.283147+00:00 (state-transition): State: in_progress → superseded
