---
id: MEN17
order: 0
plan: run-multi-engine
state: pending
created-at: '2026-10-17T09:09:32.756838+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Post-bump hardware acceptance against stored baselines

## Description

Owner-approved (2026-10-17): after a bump, run each engine's declared acceptance configs (radiance: helper-ab speed, helper-kld and 27b-native-kld quality; llama.cpp: its production A/B) and compare against stored baselines with tolerances from measured noise, instead of reading numbers by eye. Includes named, stored KLD references per model/layout (no self-recorded reference in the same run) so the gate does not compare a control against its own reference.

## Steps

1. Baseline store: docs/reference/engines/<engine>/baselines/*.json (metric, value, noise band, run id, pin). 2. Engine declares acceptance configs + baselines in engine.toml. 3. `bigcherry --engine <name> accept` runs them through ab-benchmark and gates against baselines; writes an evidence record. 4. Stored KLD references (path + recorded-on pin/layout) used by quality configs. 5. pin-bump/pin-release can require a passing accept record.

## Detailed Solution & Technical Design

Reuse campaign.acceptance.prepare/run_entry/accept and existing engine adapter. Pass immutable candidate target pin/source/selection/existing build_plan_id, verified binary and runtime home, installed plugin versions from pin_bump.run_phase_accept; each engine.toml entry declares exact lane/experiment, distinct helper lane resolved/built for same target and verified composition, no inherited RAD_BIN/RAD_HOME or old default. Explicit override validated/recorded. Missing/ambiguous build or independent KLD ref refuses before lease; independent stored ref model/layout/tokenizer/corpus/provenance, no same-run reference or regeneration on failure. Add required native dense single/TP2/TP3, helper, quality and actual installed canary/stop, resource and named cache reuse, matched stock paired performance with RHP12 noise/repetitions; qualified nonstock PWR10 apply/restore only where profile intended, no automatic historical tuning. RSC11 current supported YAML roster preflight and full S18 migration matrix; routine release Gate E plus affected paths, RCS23 family gates distinct. Use existing capture/lifecycle owner, not direct proxy or second executor. Explicit build-only/unqualified vs qualified release proof; required skipped/no GPU/missing ref/build mismatch never pass. RSP01/pin-release requires candidate-bound evidence for qualified promotion; preserve deliberate release.

## Code Samples & Guidance



## Files



## Validation

radiance accept on the 1.7.0 pin passes against the 1.3.0 baselines (2,050 pp4096 / 124 tg512 / KLD 0.064); a degraded arm fails.

MRC08: offline schema/privacy sentinel tests, missing/wrong model, provider, policy, KV, workload or provenance rejects comparison; identical cohort across different pins compares; degraded metric fails; independent reference mismatch fails; failed/aborted or interrupted session never passes; concurrent and retry export does not rerun GPU or overwrite other work. Real matched previous-release and candidate records for representative dense IQ8R/IQ6R and FlashNext IQ4R/HCBF16 routes, durable portable summaries and linked local run IDs. Report missing references explicitly.

## Effort & Risk



## Standards



## Acceptance Criteria

**Transferred MEN23 acceptance (independent gate):** MEN17 acceptance entries must declare validated lane + experiment/build identity and fail closed with declaration error on missing, incompatible or ambiguous metadata; fresh-checkout bigcherry --engine radiance accept builds or locates the exact declared lane and sets engine binary/home per entry itself, not from inherited RAD_BIN/RAD_HOME or a stale stock build. A user-specified override must be explicit, validated and recorded, never default. pin-bump --accept reuses precisely the same acceptance-resolver/executor, including helper-ab's rad0002/moe-remote handoff build and 27B/native/KLD reference identity. Add fake-lane offline tests for selected/incorrect/missing lane, environment independence, cache provenance and helper-ab failure before GPU, update Radiance accept docs. Only close when a named real helper-ab entry on the qualified handoff lane succeeds from a clean checkout without manually exported RAD_BIN/RAD_HOME and quality/baseline verdict is attributable; old v1.8.0 helper OOM from using stock build remains historical evidence, not a pass. MEN17's baseline/noise/ABBA post-bump quality gates remain independent and unmet until measured.

Offline fake exact lane and missing/wrong/ambiguous/stale pin, runtime home, binary or provider and inherited env, independent KLD wrong/missing, degraded arm and no GPU/skip refusal. Real helper-ab on qualified handoff lane without manually exported env, independent quality and dense/mixed/helper native lifecycle against exact candidate, RHP12 matched cohorts and PWR10 profile evidence. Actual acceptance run receipts link candidate build and actual policy. pin-bump --accept and standalone accept share executor; required evidence verified by release, no closure on existing code or prior release record. Existing MEN23 acceptance remains.

### Enhancement parity integration (2026-11-27)

[Enhancement parity guide](../../../design/radiance-enhancement-parity.md): reuse small fixture/discrete proof for active enhancement contract, including optional/diagnostic paths; actual full-engine test only for unresolved cross-boundary behaviour, combined with final installed model canary where possible. Missing/changed active capability requires explicit disposition and proof or owner-approved contract change before promotion. No production promotion of untested diagnostic work. No duplicate measurements for already proven paths.

### Minimal integrated acceptance (2026-11-27)

[Fast-gate review](../../../evidence/2026-11-27-radiance-230-bump/FAST-GATES.md): no engine tests per kernel edit; use discrete evidence. At release, reuse one matched canonical lane and existing installed canaries; exercise each model and affected path with the shortest real generation/quality/capacity check, combine in one load, reuse exact unchanged proofs, and run paired repeats only where performance changed or verdict uncertain. All YAML preflight is not 18 full serving benchmarks. Measure actual group carve/admission where capacity changed; long prompt only if needed to resolve a specific capacity/scheduler boundary. Keep independent baseline, no self-reference or skipped required proof. RSC16 new mode and XDP11 converter qualification are separate.

### Required 2.3.0 affected-model acceptance (2026-11-27)

Current engine.toml has only helper-ab and helper-kld, the latter a same-run reference; run_phase_accept does not explicitly bind candidate binary/home. Fix under existing MEN17 before qualified promotion. Preserve independent references and baseline noise; baseline actual old model/config and compare matched candidate, no self-reference. Cover IQ8R TP2/TP3 (group256), IQ6R TP3 (U6/U7, mixed head and drafter), both Flash-Next containers (single R9700, TP2 XTX), remote MoE including RDNA2 head/tail and combine, and BF16/FP8/16-8 cache, pipelined/mixed steps and long-context admission. See candidate MODEL-COMPATIBILITY.md. Every approved YAML must map to a real passing affected cohort or documented identical resolved behavior; not a Cartesian product. Exercise actual generation, accepted speculative tokens (all 18 request 3), prefill/decode, quality, capacity, lifecycle and helper execution; a successful start without drafts/helper use is insufficient. Bind exact candidate lane, config, model/drafter identity, actual row/provider/device and independent baseline to existing accept evidence. Missing evidence blocks qualified release; no model or performance concession without explicit owner acceptance.

### MRC08 bump comparison integration (owner request 2026-11-27)
Include reviewed MRC08 (run-measurement-records) in this bump; draft 99e2ee554 is not qualified. Reuse explicit stream-bench record, existing immutable local run store, allowlisted portable export and MEN17 comparator, not a new store. Match old/candidate cohorts on model/drafter content and layout, resolved profile, KV dtype/cache state, TP/helper topology and architecture classes, power policy, workload revision, token targets, concurrency, depth, seed, repeats and warm/cold order. Record source pin, build plan, binary/plugin identity and actual effective routing. Compare matching cohort keys excluding deliberately changed pin/build; missing provenance or mismatch is INCOMPARABLE, not PASS. Store independent run IDs, both build identities, per-trial samples, median/dispersion, delta, noise threshold and verdict for prefill, decode, TTFT, throughput, cache and errors. Keep independent KLD reference and typed quality evidence. Export controlled identifiers, hashes and typed metrics only, never local paths, device locators, URLs or raw text. Retain failed/aborted runs and retry export without GPU rerun. Existing short smokes are not baselines. Capture paired old and 2.3 measurements before qualified promotion; use shortest useful matched cohort set.

## Notes

BACKLOG TRIAGE 2026-11-04 absorbed items (superseded into this one; their notes keep the full history): MEN11: Merge target MEN17/MEN26 exist and are open; radiance measurement/evidence paths are used by patch-promote (#543/#546).



CORRECTION 2026-11-04: the earlier note that MEN11 were absorbed into this item is withdrawn. An independent revalidation found each either still has open scope of its own (reopened) or was delivered outright. This item's scope is unchanged.

CONSOLIDATION 2026-11-22 PHASE-5: receive MEN23. MEN23 describes an acceptance entry's build-lane selection defect in MEN17's only engine-aware accept/pin-bump--accept command; independent plan ownership would duplicate the acceptance executor. Original acceptance still applies; the transferred gate is unfinished. Source is superseded for duplicate planning ownership, not completed on implementation, benchmark or live-hardware criteria. Where Vulkan is involved, implementation remains paused by owner directive.

2026-11-25 cross-programme bump review: docs/design/radiance-bump-integration.md is the shared implementation-order and compatibility specification; MEN24 owns sequence and timing, domain implementation remains here. No duplicate checker, bump driver, manifest launch or legacy fallback. Existing MEN23 transfer exact-lane gap remains; run_phase_accept currently passes layout to campaign.acceptance.accept without explicit candidate binding; engine.toml helper-ab/helper-kld comments manual RAD_BIN/RAD_HOME, helper KLD self-records reference, 27b-native not entry. No claim of actual wrong candidate observed in this review.

2026-11-25 second integration review: docs/design/radiance-integration-review-20261125.md is the exact-source review and controls conflicting earlier wording. Shipping pin remains b6b8751 (2.2.0), upstream observed main 1c104de4 is a development candidate, 119 commits and 327 changed files; no automatic repin or backport. No new compatibility layer, launch manifest, bump driver or identity machinery. Current Gate E evidence 2020.11.5 only two install/canary/stop layouts, not exact candidate hardware comparison or quality. Full S18 all supported installed YAMLs (18 currently) once, normal releases existing two Gate E plus affected configs/required cohorts. Exact candidate real provider/row and native ABI/layout, independent KLD reference and actual binary/home binding; no optional gate omission promoting qualified release. Native provider A/B/C same-source, not cross-pin speed attribution.

2026-11-25 serving/configuration closure sweep: docs/design/radiance-serve-config-review-20261125.md. Bind acceptance and bench metadata to actual verified owner/child/endpoint, not /proc --manifest or newest record. Stock Gate E uses named stock YAML, nonstock intended lab arms use one YAML change with separate consent. Existing run evidence and independent KLD ref, no stale/missing provenance pass. S18 full supported config matrix distinct routine affected release gates.

Owner scope correction 2026-11-25: docs/design/radiance-native-yaml-launch.md section Scope and acceptance rule governs. Preserve behavior; smallest native launch and YAML policy change, no generalized machinery. Reuse existing tests/evidence, one shared dense/helper installed acceptance plus materially different affected paths only; cheap parser checks across approved YAMLs. Independent outstanding work stays with its owner, not a blanket S18 prerequisite. Existing exact candidate binding and independent reference correctness remain essential. Use same actual installed dense/helper cohort and affected workload checks as RSC11/RSP01; do not require separate duplicated runs. Keep unrelated existing acceptance tasks independent.

2026-11-26 next execution roadmap (docs/planning/NEXT_EXECUTION_ROADMAP.md; main 725bd8951). Assumes native YAML serving cutover and initial min-cap qualification accepted; does not claim they are done. Stage 0 source finding: 2020.17.0 overlay parity evidence could not use normal baseline comparator because two launches in one measure share run_id; tools/bigcherry/core/results.py:70 intentionally requires distinct launch IDs. Correct existing producer to allocate distinct existing run IDs with common parent/campaign context (reuse core/workspace.py and campaign/planner.py patterns); preserve comparator independent-launch check, no new ID/receipt scheme. Card-free two-launch regression, replay existing receipts only where independent identity actually known. Bind actual candidate/installed home and independent reference, no redundant GPU measurement.

Owner requires VTP59 medium-prompt prefill/decode batching fix before bump promotion. Current 2.2 baseline: four streams at 4096 prompt and 128 output collapse to 124.02 engine tok/s; interleave 0 restores 410.16 but p95 event gap rises to 974 ms. Require matched quality, latency and throughput validation of a balanced fix on old release and candidate; do not blindly turn interleave off.

VTP59 expanded stability gate: audit all resolved profile defaults with field provenance before and after pin change; sparse prompt/output/concurrency boundary and arrival screening, then matched repeats on anomalies; qualify IQ8R/IQ6R TP2/TP3, long KV variants, FlashNext IQ4R/HCBF16 helper/native classes. Independent MRC08 cohort records must include realized tokens, effective scheduler settings and separate prefill/mixed/decode metrics; do not treat short-prompt kernel gates as concurrency qualification. No promotion until VTP59 fix and per-class checks accepted.

RV4754 baseline-first: consume MRC08/MEN29, not duplicate their record system. Current helper-ab uses legacy flags, diagnostic/observe, ambient RAD_BIN/RAD_HOME; helper-kld records same-run reference. These cannot qualify promotion. Fail closed before GPU for wrong/missing candidate, missing independently identified reference or diagnostic-only record; use existing benchmark/lab records and native owner. Required affected existing-model/format/topology cohorts, scheduler regression and quality; broad optimization and new shared KV mode after baseline.

First baseline acceptance implementation: candidate plan/binary SHA/source attestation verified; pin-bump passes saved lane, standalone accepts existing candidate-state-dir; bound runtime environment cannot inherit an ambient engine. Legacy diagnostic, unadmitted performance and same-run/missing independent quality references refuse. Current helper-ab/helper-kld remain diagnostic, so no qualified GPU verdict. Next reuse native managed run records and independently identified quality references for affected cohorts, not force old diagnostic config through the gate.

Integrated baseline repair validated: 251 Python tests and 34 subtests passed with host environment explicitly selected; actual candidate 927a24bf85c2ac29d3dbe0b779063323 source attestation and launcher SHA verified, actual ComponentGate.plugin_abi 71/71 passed. Six owned DSO hashes differ from earlier build, so do not reuse all old component proof. See integrated-offline-gates.json. GPU occupied by independent Gate E; no new GPU acceptance. Final managed record comparisons/independent quality, current bundle component gates, formal bump and installed lifecycle still open.

Actual integrated candidate 927a24bf85c2ac29d3dbe0b779063323 completed 93/93 declared offline component tests (5 libr3, 48 MoE, 40 serving); engine, owner and 71 plugins unchanged after test build. Independent historical BF16 Qwen reference (8240 scored positions, 4 docs) scored using installed bc-radiance-2020.24.1 2.2 and enhanced 2.3, same native YAML, FP8 KV, one seq, speculation and prefix off. IQ8R mean KLD 0.0013790939135882837/top1 97.22087378640776%, IQ6R-V4 0.0015840392022430626/96.90533980582524%, both exact aggregate, per-doc and byte-identical per-position row files across versions; four clean owner exits. Evidence baseline-qwen-quality.json. No formal promotion claim, references predate MRC08. Native lab identity now shares installed-release verification (49 tests passed), no second verifier.

Matched TP2 regression screen: released 2.2 bc-radiance-2020.24.1 vs candidate 927a24bf85c2ac29d3dbe0b779063323 default vs tuned pipe on/min256 batch4096 interleave off mixed250; three repeats at 2/4 streams, 4096/128 and 512/128; 36 rounds, all exact prompt hashes and per-stream token counts matched, paired native receipts and six portable exports, zero preemptions. Long aggregate 2/4: old49.662/42.315, default50.790/42.763, tuned49.998/56.798; four-stream worst TTFT 11.0095/10.9684/7.7688s, p95 gap28.606/27.465/273.160ms. Short aggregate 148.286/229.950,153.690/232.900,159.728/218.523. Existing inversion persists at default; tuned restores scaling but materially worsens latency, not promote. Need narrower latency-safe scheduling, repeat ABBA and mixed/quality qualification. Evidence baseline-matched-tp2.json. Failed lab setup and record attempts preserved; no GPU lease left. Native record validity still unverified, no formal acceptance claim.

Historical raw recovery: 18 ab-tp2 VTP36 records indexed. Seed1 2020.8.5 W (TP2 wht6) and T (TP3 exact) both have 6 successful eight-stream rounds across two launches. Content-addressed records verified, all eight prompt SHA256 and actual prompt counts match existing 2.3 512/192 screen exactly; all generated counts 192. TP2 historical seed1 median327.1015 t/s/2496.67ms max TTFT/36.5655ms p95 gap vs tuned355.173/2017.25/37.732. TP3 360.8315/2053.02/38.5305 vs tuned350.280/1987.82/40.298. Same cohort inputs now established, but historical engine/model provenance remains legacy and launch spread differs; not independent formal acceptance. Eight older 4K/256 RUM06/RUM09 records recovered with same seed1 prompt family; different output length than 4K/128 or 512, distinct tuning. See historical-cohort-reconciliation.json. No need to repeat all historical short benchmarks.

Focused TP3 governor screen on exact candidate 927a24bf85c2ac29d3dbe0b779063323: 8 streams, 4096/512 then 512/128, 2 reps each, FP8 KV exact wire, pipeline min256, batch2048, interleave off, target75. Three verified fresh-start policies arrival; budget factor1.0/remaining/512 exemption/120000ms; cap2/remaining/512/120000ms. All 12 rounds complete, prompt hashes and actual counts match, paired tuning receipts, six portable exports, zero preemptions, clean stops. Long throughput / max TTFT / p95 event gap: arrival180.3305/15.596s/154.171ms; budget125.088/27.277s/162.6465ms; cap114.489/30.842s/158.798ms. Short throughput281.9865/278.9115/275.8155. End mixed-target misses/steps142/144,347/351,440/444, learned cap64 across all (one geometry quantum); source busy_cap raises effective allowance for participant fairness and backfill, so target is soft and reducing numeric target alone is unlikely to fix latency. Restricted admission worsens 8K-context burst performance; reject these arms, preserve long-profile governor and existing defaults. Evidence tp3-governor-screen.json. No new engine patch or qualified production setting. Next isolate per-mixed-step row/participant floor vs decode cost using native metrics/host scheduler simulation; only change scheduler if a latency-safe fair policy can be proven. Baseline release gate remains open; avoid repeating already covered screens.

## Change Log

- 2026-10-17T09:09:32.756838+00:00 (created-by): Created by agent
- 2026-11-04T03:47:51.848107+00:00 (updated-by): Updated: section:notes
- 2026-11-04T04:45:43.413487+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4528
- 2026-11-22T11:08:28.715849+00:00 (updated-by): Updated: section:acceptance_criteria, section:notes
- RV4617
- 2026-11-25T02:10:29.120798+00:00 (updated-by): Updated: section:notes, section:detailed_solution, section:acceptance_criteria
- 2026-11-25T02:15:19.358080+00:00 (updated-by): Updated: section:acceptance_criteria
- RV4652
- 2026-11-25T02:50:48.994436+00:00 (updated-by): Updated: section:notes
- 2026-11-25T02:53:31.547189+00:00 (updated-by): Updated: section:detailed_solution
- RV4668
- 2026-11-25T03:31:54.393313+00:00 (updated-by): Updated: section:notes
- 2026-11-25T04:30:40.599872+00:00 (updated-by): Updated: section:notes
- RV4679
- 2026-11-25T18:51:09.347812+00:00 (updated-by): Updated: section:notes
- RV4702
- 2026-11-27T09:21:10.868081+00:00 (updated-by): Updated: section:validation
- 2026-11-27T09:22:03.040708+00:00 (updated-by): Updated: section:acceptance_criteria
- 2026-11-27T09:45:30.038198+00:00 (updated-by): Updated: section:notes
- 2026-11-27T10:09:00.551273+00:00 (updated-by): Updated: section:notes
- 2026-11-27T15:12:13.103651+00:00 (updated-by): Updated: section:notes
- 2026-11-27T15:24:06.385547+00:00 (updated-by): Updated: section:notes
- 2026-11-27T15:24:47.511828+00:00 (updated-by): Updated: section:notes
- 2026-11-27T20:10:02.027053+00:00 (updated-by): Updated: section:notes
- 2026-11-27T20:17:40.797946+00:00 (updated-by): Updated: section:notes
- 2026-11-27T20:33:10.303400+00:00 (updated-by): Updated: section:notes
- 2026-11-27T20:38:02.794957+00:00 (updated-by): Updated: section:notes
