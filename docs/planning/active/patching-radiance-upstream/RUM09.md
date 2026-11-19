---
id: RUM09
order: 9
plan: patching-radiance-upstream
state: pending
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

- Freeze one release candidate with exact upstream/local patch, component, SDK, model, power/layout and native/extension configuration identities. For the old-pin control, use the current-main policy-off baseline only as workload/input reference; it does not qualify the candidate. Use the same explicit defaults on both arms.
- Gate A: run canonical dependency-resolved patch composition against the candidate, then one build/SDK/component pass for the actual selected production target. Reuse the build to inspect every packaged plugin's ABI version and exported layout against libr4d; run only touched component unit targets. Do not repeat the full build after documentation-only changes.
- Gate B: one card-free named fresh-start -> measurement -> stored replay -> stop path using identical sealed native YAML bytes/digest/root and explicit ordered `Config.devices`. Check stale build, changed digest/root/topology, wrong-level native key and fresh-room refusal before any child/power mutation. Keep this gate on the fake fixture.
- Gate C: acquire the GPU lease. First activation is the documented helper-on TP1 `flashnext-4gpu` lane with disjoint heterogeneous helper cards and the exact E4M3 format, fixed MTP depth and policy settings in the handoff. Run the helper-off `flashnext-r9700` arm only as a functional/KL quality control. Check admission/stop and one bounded helper-death/drain fixture. Do not require a mixed-engine-rank canary for this TP1-only release. If any TP2/TP3 or custom mixed-rank profile is included in the actual runtime roster, that profile must pass its own rank-order/code-object/collective and activation gate before packaging.
- Gate D: run the existing short quality sample and thresholds, then three interleaved old/new performance pairs on the same helper-on TP1 deployment, prompts, contexts, weights, topology, speculation, interleave, heat and power. Use current policy-off comparison inputs. Repeat only to resolve a failure or material noise.
- Gate E: make the actual runtime bundle enforce the candidate roster and mode allowlist. Install and relocate it through existing RSP01 checks; verify exactly the qualified config names/formats are shipped, plugin ABI generation is uniform, and health/start/stop work. The current packer requires both shipped TOML profiles, so add an explicit roster input and migrate its callers/tests together before claiming unqualified profiles withheld.
- Only after Gates A-E pass, update recipes pin/version, all six component built-against declarations, native config metadata and package roster together. Record exactly supported names/modes; keep old release artifacts available for rollback. Never silently include an unqualified profile.

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

Run each gate once on the frozen candidate and exact runtime roster. Re-run only when a code/config input consumed by that gate changes or its result is unresolved. Required evidence is: canonical patch report; matched SDK build, plugin ABI/layout and component tests; fake immutable-config round trip and refusal checks; leased TP1 helper-on activation plus helper-off quality arm; current quality threshold result; three paired same-topology performance comparisons; and installed/relocated Python-free package inventory. No full model-family, quantization or long-soak sweep for a TP1-only release. A mixed-rank profile in the actual candidate roster adds its specific compatibility and activation gate.

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
