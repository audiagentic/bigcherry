---
id: RUM08
order: 8
plan: patching-radiance-upstream
state: in_progress
created-at: '2026-11-18T01:28:40.856862+00:00'
breadth: ''
skill: advanced
created-by: codex
priority: P1
work: M
---

# Reuse upstream MTP replay and reconcile scheduling policy

## Description

Compare upstream recording against rad0013 before retirement; select one scheduler controller and preserve rank-consistent draft/history handling. Keep adaptive/traversal/suffix modes outside the initial acceptance matrix unless explicitly enabled.

## Steps

- Compare target Qwen4exp pass_key and recorded draft/verify/decode behavior with rad0013's exact intended fix, including split cap and QSA exactness/grid boundary.
- Instrument only enough to show the selected fixed-depth production draft and verify paths hit the expected replay keys. Retire rad0013 if target coverage is equivalent; otherwise keep a small semantic fix rather than its old full override.
- Inspect target scheduler prefill-interleave auto defaults, mixed-step timing controller, preemption and prefix adoption against rad0036. Select one policy authority and map its settings through native YAML.
- Preserve rad0028 rank-consistent draft decisions and rad0029 ownership behavior with actual verify rows. Keep suffix/adaptive/traversal modes disabled unless they are part of the chosen production configuration.
- Remove superseded governor/config parsing and associated dead tests/aliases once all callers are migrated. Keep unrelated diagnostic experiments out of production composition.
- Use upstream defaults only after an explicit decision; initial comparison pins the old production behavior so policy drift cannot hide behind a version bump.

- Extend ABI43 tests using existing `tests/arch_test.cpp` exact-QSA-bound coverage and `tests/runtime_test.cpp` effective keyed-bound coverage; inspect actual selected/effective launch arguments, not only the row key.
- Prove whether a reachable wide verify can receive a missing/nonpositive summary. The architecture key maps missing and above-bound summaries to the same selector bit, while issue code substitutes the keyed bound only for a proven positive summary. If reachable, keep that case live with exact geometry or a narrow semantic patch; do not restore rad0013 wholesale.
- Preserve rad0028 rank-0 proposal agreement and rad0029 per-rank resource bounds. Choose one explicit production interleave/depth authority; pin it for comparisons and scope or withhold long profiles whose governor semantics differ from the upstream mixed-step timing controller.

## Detailed Solution & Technical Design

Upstream commit 7f5218ee already records Qwen4exp draft/verify/decode passes; subsequent pass keys include split geometry and indexer exactness decisions. Reapplying the local override wholesale can discard those newer correctness distinctions. Review the current function and compare semantic key cases instead of restoring a removed textual comment.

Latest target commits add token/block/traversal verification, adaptive depth and suffix proposals. They are available features, not compulsory work for the minimum bump. Fixed depth and the current verification mode are enough for initial acceptance. A variable-row future mode must fit admission, ownership and cache replay; do not assume testing fixed depth proves it.

rad0034/35 are optional policy experiments, rad0036 is the production governor. Port the current production objective with the existing upstream controller when equivalent; retain a small typed scheduler policy hook only for a demonstrated semantic gap. Never let two independent controllers adjust the same cap. Put timing observation at scheduler completion, not inside GPU kernels.

Astra refinement: target pass_key keys the exact QSA-bound side and slot-indexed state for decode/verify/MTP; mixed remains live. Prefer removing rad0013's full local definition after one boundary-crossing fixed-depth comparison and one ineligible mixed shape. rad0036's aged-long-prompt reservation, remaining-token order and admission budget differ from native time control. Pin prefill-interleave/mixed-step-target-ms explicitly. Keep a demonstrated required long-profile decision or withhold that profile; do not claim native equivalence from option names. rad0034/35 were already superseded locally, not new deletions.

ABI43 candidate replay already omits the production rad0013 override because Qwen4exp pass keys and core issue carry keyed decode/verify/MTP draft geometry. This does not by itself prove launch equivalence. Reuse candidate tests `qsa_reads_the_exact_host_bound_and_keys_its_side_of_the_exact_bound`, `the_prefill_geometry_key_reuses_equal_grids_and_separates_changed_grids`, and the runtime effective-bound test. Add only the smallest missing case to assert the selector output and actual arguments around budget/ratio minus one, at the boundary, and plus one in the same rounding bucket. Distinguish same keyed side with changed exact bound, narrow/wide split geometry, mixed rejection, and absent summary. Existing rad0034/35 are superseded; rad0036 has a separate long-Qwen policy with remaining-token order, 120-second age reserve and an admission budget that differs from upstream's mixed-step controller.

## Code Samples & Guidance

Minimal replay key cases to inspect:
```text
same rows + same split cap + same QSA/grid side => same key
changed split/grid/exactness decision          => distinct key or explicit live path
unknown/unrecordable shape                    => explicit no-replay
```
The actual upstream key algorithm remains authoritative; these are proof cases, not a new hashing implementation.

### Deep scan implementation refinement

Main net-deletion candidates are rad0013 replay override after target semantic coverage proof and rad0036 governor after policy equivalence. Remove production entries, obsolete option/state mapping and superseded tests in the same slice. Diagnostics rad0010/39/43/44 may share route identity and bounded sink plumbing, but histogram/stats/per-token records retain different contracts and off-path costs.

Source evidence and removal conditions: docs/reviews/radiance-upstream-2026-11-18/CONSOLIDATION.md. Use patch-footprint.json to inspect every overlapping patch before changing an attachment site.

## Files

- engines/radiance/patches/rad0013_mtp_pass_replay/patch.py
- engines/radiance/patches/rad0028_rank_consistent_block_drafts/patch.py
- engines/radiance/patches/rad0029_rank_own_split/patch.py
- engines/radiance/patches/rad0036_prefill_governor/patch.py
- engines/radiance/recipes.toml
- engines/radiance/components/serving/config_loader.cpp

- engines/radiance/patches/rad0047_mtp_verify_replay_summary/patch.py
- engines/radiance/patches/rad0047_mtp_verify_replay_summary/patch.toml
- tools/tests/patch/test_radiance_rad0047.py

## Validation

One focused replay-key case table, one rank-agreement/accepted-history fixture, one mixed prefill plus decode/prefix-preemption case using the chosen controller. One fixed-depth canary in RUM09. Do not sweep all depth/verification/suffix combinations while proving the basic bump.

Use candidate host tests for QSA boundary/key and runtime effective-bound behavior, plus the BigCherry rank-consistent draft and per-rank room tests. Test fixed-depth production replay with helper active and inactive, including exact effective launch geometry for decode, verify and MTP draft. Confirm the missing-summary path is unreachable for accepted geometry or assert its live fallback. Keep the production comparison interleave and fixed-depth settings identical between baseline and candidate. Do not use the v1.17-only fixture as evidence of ABI43 behavior.

## Effort & Risk

M: upstream replay/governor overlap can silently change behavior. Pin one production mode and test key distinctions; leave new adaptive/suffix/traversal features outside initial qualification.

## Standards

Shared agent workflow; tooling ownership and evidence boundaries. No deep testing during plan authoring.

## Acceptance Criteria

The production mode has one controller and correct replay shape discrimination; no duplicate override loses target semantics. All ranks agree on admitted rows/history and the explicit default comparison is documented. Additional upstream modes stay unqualified until separately selected.

No unnecessary rad0013 production override remains for ABI43; the selected fixed-depth path demonstrably replays with the same effective launch arguments for all admitted shapes. Missing summary does not silently change grid behavior. Rank proposal agreement and per-rank resource bounds remain enforced. Exactly one documented scheduler/interleave policy is used by the initial production composition.

## Notes

Imported stable roadmap identity from the reviewed design. Prerequisites: RUM04, RUM06, RUM07. Review: docs/reviews/radiance-upstream-2026-11-18/README.md; proposal: MANIFEST-AND-EXTENSIONS.md. This initial item is the first-pass execution skeleton; detailed code guidance follows after the plan merge.

Prerequisites RUM04/RUM06/RUM07. Reuse ENG02 measurement ownership for later depth tuning; no broad tuning campaign in this migration.

Astra source review ASTRA-CONSOLIDATION.md: resolve native replay/controller choices before porting their hunks; no new generic scheduler interface. Integrated qualification remains RUM09.

C10 (2026-11-19): on the exact ABI43 source with HIP disabled, existing arch_test qsa_reads_the_exact_host_bound_and_keys_its_side_of_the_exact_bound passed 30 checks; runtime_test an_arch_keyed_program_issues_decode_passes_at_the_keyed_bound passed 6. These cover exact-bound selector and effective keyed pass bounds only. Sol identified a remaining edge: the architecture key may coalesce missing and above-bound summaries while issue uses a keyed context only for a proven positive summary. Prove that wide verify cannot receive missing summary or add one focused selected/effective-argument test before retiring rad0013; RUM08 remains open.

2026-11-19 Sol follow-up to RV4239: source tracing narrows the C10 summary edge. BatchBuilder allocates and attaches kv_used_max_h from per-group h_used and seqused_h; an accepted nonempty full-KV decode/verify therefore has a positive summary. A missing pointer alone can be recomputed by issue.cpp for a true full-group operand. The wide QSA verify path above the bound uses private qsa_sequ, so both summary cases use the same model-bound fallback there. Do not change pass-key encoding based only on an absent pointer. Verify effective arguments on selected fixed-depth paths; if an accepted dense path can carry genuinely missing/nonpositive summary, make it fail closed or encode presence explicitly. See completed RV4239.

C13 (2026-11-19): added candidate-only `rad0047_mtp_verify_replay_summary`. Against the frozen ABI43 commit, exact apply and idempotent replay passed; the compiled pass-key fixture confirms the wide speculative decode case returns 0 for a nonpositive full-KV summary and retains a nonzero key for positive 1504. Focused test: `tools/tests/patch/test_radiance_rad0047.py` (2 passed). Keep the patch unselected/untested in production recipe until the candidate prefix and effective live fallback are validated; v1.17 `rad0013` remains unchanged.

2026-11-19 Sol review: ABI43 rewrites an absent prefill_interleave to auto:1; the matched v1.17 parity manifests omit the option and therefore use 0. Pin prefill_interleave=0 on flashnext-4gpu (inherited by its alias) to preserve the measured production setting. The existing v2 registry maps it to --prefill-interleave; assert explicit zero survives the shared native argv projector and the resolved named profile. Confirm the matched parser sets Config.prefill_interleave_set=true in the native core fixture before the runtime canary. num_speculative_tokens was explicitly 3 in historical manifests and the current BC defaults resolve to 3; do not change it without a separate historical/effective-value comparison.

2026-11-19: historical parity launch manifests show num_speculative_tokens=3 and no prefill-interleave. ABI43 turns absent interleave into auto:1, so the flashnext-4gpu profile now pins 0 (alias inherits it). Resolver and native argv projection tests assert the named value and exact `--prefill-interleave 0`; the rad_core native-YAML fixture asserts explicit zero sets the option/provenance. Review and update the two affected layout-manifest golden hashes after running bc_serving_test. Keep depth 3 and rad0036 governor policy unchanged.

## Change Log

- 2026-11-18T01:28:40.856862+00:00 (created-by): Created by codex
- 2026-11-18T01:40:00.933069+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:acceptance_criteria, section:notes
- 2026-11-18T01:43:35.850667+00:00 (updated-by): Updated: section:effort_risk
- 2026-11-18T01:57:12.077870+00:00 (updated-by): Updated: section:code_samples
- 2026-11-18T02:20:05.918412+00:00 (updated-by): Updated: section:detailed_solution, section:notes
- 2026-11-18T02:23:54.480453+00:00 (updated-by): Updated: section:notes
- 2026-11-18T02:26:51.232442+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4239
- 2026-11-19T02:22:09.889177+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:validation, section:acceptance_criteria
- 2026-11-19T02:54:52.556661+00:00 (updated-by): Updated: section:notes
- 2026-11-19T03:06:34.707493+00:00 (updated-by): Updated: section:notes
- 2026-11-19T03:16:06.097468+00:00 (updated-by): Updated: section:files, section:notes
- 2026-11-19T08:59:55.589700+00:00 (updated-by): Updated: section:notes
- 2026-11-19T09:00:55.776384+00:00 (state-transition): State: pending → in_progress
- 2026-11-19T09:02:24.106483+00:00 (updated-by): Updated: section:notes
