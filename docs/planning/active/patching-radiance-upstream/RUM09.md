---
id: RUM09
order: 9
plan: patching-radiance-upstream
state: in_progress
created-at: '2026-11-18T01:28:48.655352+00:00'
breadth: ''
skill: advanced
created-by: codex
priority: P0
work: M
---

# Prove the minimum working transition and publish the migrated runtime

## Description

Run only the matrix needed for the changed production configuration: composition/build/SDK, launch/stop/admission, helper-off control, the selected helper-on TP1 canary, short quality and paired performance checks. Qualify mixed-rank profiles only if they are included in the candidate roster. Tie evidence to exact source/build/data identities. Switch pin and publish after gates pass; retain the previous verified release as a recovery artifact, not a legacy code path.

## Steps

- Freeze the exact release candidate source/build/model/config identities and explicit defaults. Retain the previous verified release artifact for rollback.
- Gate A: run canonical production composition resolution and ordered apply/reapply, one same-SDK build, plugin ABI/layout inspection, and focused host/component tests for the selected targets. Reuse the build; do not repeat it for documentation-only edits.
- After Gate A and RUM07's feature migration gates pass, update the candidate recipe pin/version, all six component built-against declarations, native config metadata, and package roster together. Promote every retained, historically tested BigCherry feature to the bumped production composition. Record historical behavior evidence as carried forward and new-base compatibility evidence separately. This source promotion does not mean the runtime/release gates below have passed.
- Gate B: one card-free named fresh-start -> measurement -> stored replay -> stop path using identical sealed native YAML bytes/digest/root and explicit ordered Config.devices. Check stale build, changed digest/root/topology, wrong-level native key and fresh-room refusal before any child/power mutation. Keep this gate on the fake fixture.
- Gate C: acquire the GPU lease. Run the documented helper-on TP1 canary with the exact E4M3 format and fixed MTP/policy settings, plus the helper-off quality control. Check admission/stop and one bounded helper-death/drain fixture. Qualify any mixed-rank profile only if it is included in the candidate runtime roster.
- Gate D: run the existing short quality sample and thresholds, then three interleaved old/new performance pairs on the same helper-on deployment. Repeat only to resolve a failure or material noise. This remains the final regression/performance gate requested by the user.
- Gate E: enforce the actual runtime bundle roster and mode allowlist, then install/relocate through existing checks and verify the qualified profiles, uniform plugin ABI, health/start/stop.
- Publish only after Gates B-E pass, the ledger is synchronized, and required CI succeeds. Do not describe source/package promotion as runtime qualification or as completion of the whole release item.

## Detailed Solution & Technical Design

Required proof is a working transition on the actual deployed path, not exhaustive validation of every upstream feature. Reuse existing callback/admission/KV/session/SDK tests and prior accepted thresholds. A clean patch check is a source gate, compilation is an interface gate, and one hardware canary proves activation; none substitutes for the short quality/resource gate.

The performance comparison must use identical prompts/contexts/sequence count, weight files, topology and power policy. Record prefill/decode, TTFT/ITL, acceptance, memory peak and obvious transport stalls, but do not invent new instrumentation if existing receipts already provide them. Unexplained quality/memory regressions block that configuration; noise alone triggers one targeted investigation rather than the whole suite.

Keep the previous verified release bundle as an operational recovery artifact. Do not retain old internal API code paths, a second config parser or old plugin binaries in the new bundle. The production switch should happen only after the candidate bundle is identifiable and installable.

During this planning/detail task run only document/plan consistency, link checks and repository-required CI. All native builds and hardware gates above are future implementation acceptance, not work to execute while writing the plan.

## Code Samples & Guidance

Minimum acceptance record (one row per selected configuration):
```text
source/build/model/config digests | layout/roles | enabled KV/spec/controller
apply + build + SDK | launch/admit/stop | active helper/replay markers
short quality/noise gate | paired prefill/decode/TTFT/ITL | memory headroom
bundle install/relocate | result + explicit unsupported modes
```
Use existing release, KLD and receipt tools; do not create a new campaign framework for this table.

### Deep scan implementation refinement

Add an improvement receipt: changed upstream attachment sites, handwritten projections, actual runtime branches removed, moved source lines, generated lines and retained code. Do not impose arbitrary LOC quotas. For subsequent bumps regenerate the seam overlap report with existing FilePatch/registry APIs and run focused changed-seam checks plus one admitted production canary. Existing unrelated atomic writer consolidation is not a bump gate.

Source evidence and removal conditions: docs/reviews/radiance-upstream-2026-11-18/CONSOLIDATION.md. Use patch-footprint.json to inspect every overlapping patch before changing an attachment site.

## Files

- engines/radiance/recipes.toml
- engines/radiance/engine.toml
- engines/radiance/components/serving/sdk_check.cpp
- engines/radiance/components/serving/release_install.cpp
- docs/reference/build/RELEASES.md
- docs/reviews/radiance-upstream-2026-11-18/README.md

## Validation

- Gate A and RUM07 migration evidence: exact production closure resolves to validated states; all retained cache packages apply and reapply byte-identically on the frozen 2.1.0 source; focused host format/runtime/geometry/prefill tests, selected target object builds, complete component build, ABI44 checks, and serving/libr3 component tests pass.
- Reuse historical BigCherry device, codec, KLD, serving and performance evidence for behavior already established on the old base. Record new-base port checks as compatibility evidence, not as a claim that old behavior was untested.
- Latest-main additions are covered by focused stored-manifest and runtime-receipt tests and parity harness tests.
- Gates B-E remain required for runtime/release completion: immutable-config lifecycle and refusals, leased hardware activation, short quality gate, paired regression/performance comparison, and installed/relocated runtime roster.
- Run each gate once on the frozen candidate. Re-run only if an input changes or a result is unresolved. No broad model-family or long-soak sweep.

## Effort & Risk

M: uncontrolled defaults or model/layout drift makes A/B inconclusive. Freeze identities and run a bounded reused gate set; expand only after a real regression or unresolved noise.

## Standards

Shared agent workflow; tooling ownership and evidence boundaries. No deep testing during plan authoring.

## Acceptance Criteria

One explicitly enumerated runtime roster passes source, same-SDK build/plugin ABI+layout, strict config/admission/lifecycle, selected TP1 helper-on activation, helper-off functional quality, short quality, bounded paired performance and install/relocate gates. The package actually withholds every unqualified profile/mode. Recipe, component declarations, native config and runtime roster refer to one exact candidate. No production pin change occurs before all gates pass.

## Notes

Imported stable roadmap identity from the reviewed design. Prerequisites: RUM03, RUM04, RUM05, RUM06, RUM07, RUM08. Review: docs/reviews/radiance-upstream-2026-11-18/README.md; proposal: MANIFEST-AND-EXTENSIONS.md. This initial item is the first-pass execution skeleton; detailed code guidance follows after the plan merge.

Prerequisites RUM03-RUM08. RUM10 fleet and later placement optimization are not blockers. Refer to RHP12/RAD39/ENG02/RSP01 for existing acceptance owners and thresholds; do not duplicate their campaign engines.

Astra source review ASTRA-CONSOLIDATION.md adds installed normal, measure/KL and stored-start proof: native option absent from BC registry passes via named launcher; wrong-level native key fails; identical immutable bytes and matching parser/build defaults reach planning/serving/replay; changed digest refuses before helper/power mutation. Cached starts still require fresh room; partial helper start rolls back. Explicitly support, pin or withhold every other shipped configuration before changing a common pin. Count actual handwritten/attachment deletions separately from moved/generated lines. No new bundle-index or acceptance framework.

Stage Gate C: first run the TP1 Flash-Next production canary with disjoint heterogeneous helpers. This is first launch/parity evidence only. Run required mixed-rank qualification separately before any common-pin release claiming that support; if unavailable retain the previous release for those lanes. The helper-off profile is functional/KL control, not a throughput comparator; compare same layout and workload across old/new builds. Exact gate recipes: docs/reviews/radiance-upstream-2026-11-18/IMPLEMENTATION-HANDOFF.md.

RV4240: original Gate C required a helper-on mixed-TP canary despite the selected first lane being TP1; packer profile allowlist also needed enforcement. The corrected sequence qualifies the named TP1 lane first and makes mixed-rank proof conditional on inclusion in the candidate roster. RUM09 still owns release execution. The current runtime asset packer must be changed before claiming other names are withheld. Policy-off baseline file is an old-pin workload reference, not candidate evidence. See docs/reviews/radiance-upstream-2026-11-18/IMPLEMENTATION-HANDOFF.md and docs/evidence/radiance-latest-bump/README.md.

User direction (2026-11-19): migrate all retained, historically tested BigCherry features onto the new Radiance base and include them in the production recipe. After Gate A and RUM07's exact compatibility gates pass, update the candidate pin and promote the packages together; do not leave them candidate-only while waiting for the final paired performance comparison. The final RUM09 deployment/runtime qualification and performance comparison remain pending and must gate release/publish. The latest origin/main commits through 31bd5b9e9 are merged into the task branch; their stored-manifest replay, boot-placement receipt, and parity warmup changes are part of the candidate.

Main integration checkpoint 99ae7bfea incorporates origin/main d611e7997, with ledger acknowledgement 25dee8511 on the isolated integration branch. The merged v2 serving, stored manifest, and feature state checks passed 3/3. FlashNext parity harness passed 18/18. The integration branch is ready to combine with the native startup checkpoint. These checks establish compatibility of the merge, with the candidate release gates still open.

A full 2.2 v3 runtime was staged for the ec28406a2 checkpoint. Engine, worker and serving use identical wire fingerprints. All ten kernel and 49 architecture plugins plus one quantizer reported ABI44 fingerprint 5cc71b9c5bb7ab40. A stock-power helper-on Flash-Next lifecycle check is queued on the GPU lease with fresh, named-cache and explicit stored replay stages. No GPU result yet. The strict recipe gate still requires retained E4M3 rad0046 qualification. Run final hardware quality and matched old/new parity on the final integrated runtime.

The pool-fallback engine produced a live plan. At an explicit diagnostic trial of 2247 MiB, writer convergence succeeded and all three helpers and the serving engine launched. After loading Flash-Next, stock Radiance 2.2 refused managed serving because its Engine had no attached server Front. Health, named replay and explicit replay did not complete. Automatic remeasurement checkpoint be89849c0 passed the card-free convergence, lookup and replay test and is running separately under the GPU lease without the manual pool override.

Live automatic remeasurement at be89849c0 produced two distinct engine plan receipts and converged from 9216 to 2247 MiB unchanged YAML, then launched serving. Front integration 7ef6186a2 attaches the stock HTTP front on managed online paths and leaves measurement/offline unchanged. Exact 2.2 patch replay, five manifest cases and 105 native front host checks pass. A front-enabled live run is queued separately for health, fresh start and both replay paths.

Live checkpoint 77fd71cf6 converged and passed health and default-name routing. The first completion reached Flash-Next layer 0 and failed with mailbox publish operand 3 handle 0. Operand 3 is the required E4M3 activation slice; the HIP op also requires its scales and base. Stock 2.2 can use an I8 local activation lane without materializing the E4M3 pair. Port the producer planes while preserving the local lane and the strict runtime check. No working completion or replay yet.

At producer checkpoint e36f584f0, the live run passed health and the prior required buffer-handle check. The first completion reached GPU execution, then gfx1201 reported an HSA queue exception at mailbox_seal_kernel. The owner received no HTTP reply and ended the session. Preserve the engine fault log and diagnose the mailbox pointers and queue ordering before another live attempt.

Fail-closed diagnostic checkpoint d466f696a reproduced the first request on leased hardware without losing the error to a queue trap. Parent 1 had matching ticket/epoch/slot (1/1/0) but 139 published and 0 consumed. Host proof rejected it and returned HTTP 500 without emitting consumption ACKs. Investigate finish operation issue/replay and counter reduction before the next functional canary.

Live direct per-slot accounting at bb1b8a190 reproduced exactly 139 published and 0 consumed. This comparison rules out the workgroup reduction as the sole explanation. The GPU session exited cleanly and released its lease. Next diagnostic distinguishes first-capture dispatch from recorded replay and checks whether any device consumed records were written before seal.

Packet trace captured 102 publish and 51 finish dispatches for parent 1. Both old and inverse-aware publish hooks had applied to the 2.2 graph because a multiline negative guard matched after the capability marker. Checkpoint 03f587211 uses a whole-file applicability guard and exact one-publish fixture. The leased helper-on Flash-Next checkpoint generated 24 tokens in each fresh, named and explicit stored session and used one exact manifest with only two measurement attempts. Explicit replay used the original site inputs after the authored YAML was removed. Final integrated runtime build, bounded quality/performance and release/install gates remain.

## Change Log

- 2026-11-18T01:28:48.655352+00:00 (created-by): Created by codex
- 2026-11-18T01:40:08.541378+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:acceptance_criteria, section:notes
- 2026-11-18T01:43:42.600655+00:00 (updated-by): Updated: section:effort_risk
- 2026-11-18T01:57:19.521344+00:00 (updated-by): Updated: section:code_samples
- 2026-11-18T02:20:14.331286+00:00 (updated-by): Updated: section:notes
- 2026-11-18T02:24:08.970577+00:00 (updated-by): Updated: section:notes
- 2026-11-18T02:27:05.771137+00:00 (updated-by): Updated: section:notes
- 2026-11-18T03:40:35.424079+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4240
- 2026-11-19T03:00:58.377857+00:00 (updated-by): Updated: section:steps, section:validation, section:acceptance_criteria, section:notes
- RV4243
- 2026-11-19T04:28:00.455200+00:00 (updated-by): Updated: section:description
- RV4245
- 2026-11-19T07:30:08.804115+00:00 (updated-by): Updated: section:steps, section:validation, section:notes
- 2026-11-19T07:35:45.850777+00:00 (state-transition): State: pending → in_progress
- 2026-11-19T21:41:03.879740+00:00 (updated-by): Updated: section:notes
- 2026-11-19T23:05:36.031221+00:00 (updated-by): Updated: section:notes
- 2026-11-20T00:13:25.180057+00:00 (updated-by): Updated: section:notes
- 2026-11-20T00:23:43.215660+00:00 (updated-by): Updated: section:notes
- 2026-11-20T00:43:44.078640+00:00 (updated-by): Updated: section:notes
- 2026-11-20T00:55:22.935951+00:00 (updated-by): Updated: section:notes
- 2026-11-20T01:03:47.081769+00:00 (updated-by): Updated: section:notes
- 2026-11-20T01:12:49.299909+00:00 (updated-by): Updated: section:notes
- 2026-11-20T02:05:25.873512+00:00 (updated-by): Updated: section:notes
