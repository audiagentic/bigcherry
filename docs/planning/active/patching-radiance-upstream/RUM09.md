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

Run only the matrix needed for the changed production configuration: composition/build/SDK, launch/stop/admission, helper-off control and representative helper-on mixed TP, short quality and paired performance checks. Tie evidence to exact source/build/data identities. Switch pin and publish after gates pass; retain previous verified release as recovery artifact, not a legacy code path.

## Steps

- Freeze one release candidate with exact upstream/local patch, component, SDK, model, power/layout and native/extension configuration identities. Use the same control workload and explicit defaults.
- Gate A: dependency-resolved patch apply plus one build/SDK check for actual production targets. Run only touched local unit targets; don't rebuild all targets repeatedly for documentation changes.
- Gate B: one fake/CPU launch-stop case, exact stored snapshot replay, stale-build/topology/digest refusal and fresh resource admission. Use existing session fixtures.
- Gate C: on leased hardware run the helper-off control and one currently deployed helper-on mixed-TP layout with short prefill/decode and fixed-depth speculation. Check helper death/drain with an existing bounded fixture, not a long chaos run.
- Gate D: run one short paired quality sample using current project thresholds/noise handling and a small matched performance comparison (three interleaved pairs). Add repetitions only if a failure/noise question remains; otherwise stop.
- If the selected path passes, update recipes pin/version and actual runtime launch/package metadata together. Remove retired patch set entries and dead code after coverage is established; keep history/evidence.
- Build/install/relocate one Python-free runtime bundle through existing RSP01 release checks, verify one plugin ABI generation and health/stop, then publish the version.
- Record exactly which formats/layouts/modes passed. Broader optional configurations are explicitly not qualified by this cutover and cannot quietly enter production.

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

## Files

- engines/radiance/recipes.toml
- engines/radiance/engine.toml
- engines/radiance/components/serving/sdk_check.cpp
- engines/radiance/components/serving/release_install.cpp
- docs/reference/build/RELEASES.md
- docs/reviews/radiance-upstream-2026-11-18/README.md

## Validation

Gate A-D above, each once on a fixed candidate. Re-run a gate only after code/config affecting it changes or a result remains unexplained. No full model-family, layout, quantization or long-soak matrix is required for the first single-model bump.

## Effort & Risk

M: uncontrolled defaults or model/layout drift makes A/B inconclusive. Freeze identities and run a bounded reused gate set; expand only after a real regression or unresolved noise.

## Standards

Shared agent workflow; tooling ownership and evidence boundaries. No deep testing during plan authoring.

## Acceptance Criteria

One documented production configuration passes source/build/ABI, strict launch/admission/lifecycle, activated helper/KV/spec quality and bounded paired performance gates, and its runtime bundle installs cleanly. Pin and source metadata match that candidate; incomplete optional work stays pending.

## Notes

Imported stable roadmap identity from the reviewed design. Prerequisites: RUM03, RUM04, RUM05, RUM06, RUM07, RUM08. Review: docs/reviews/radiance-upstream-2026-11-18/README.md; proposal: MANIFEST-AND-EXTENSIONS.md. This initial item is the first-pass execution skeleton; detailed code guidance follows after the plan merge.

Prerequisites RUM03-RUM08. RUM10 fleet and later placement optimization are not blockers. Refer to RHP12/RAD39/ENG02/RSP01 for existing acceptance owners and thresholds; do not duplicate their campaign engines.

## Change Log

- 2026-11-18T01:28:48.655352+00:00 (created-by): Created by codex
- 2026-11-18T01:40:08.541378+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:acceptance_criteria, section:notes
- 2026-11-18T01:43:42.600655+00:00 (updated-by): Updated: section:effort_risk
