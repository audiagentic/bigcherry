---
id: PVPS02
order: 0
plan: patching-validation-package-standard
state: pending
created-at: '2026-09-11T06:35:00.394840+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P1
---

# Generic per-patch A/B benchmark harness (replace per-RD hardcoded CLI gates)

## Description

tools/bigcherry/patch/validation_campaign.py's paired-benchmark evidence producers (run_rd04_benchmark_evidence, run_rd08_validation_lanes, the RD58/RD73 equivalents) are each genuinely reusable, generic functions -- confirmed by direct inspection 2026-09-11: run_rd04_benchmark_evidence(control_binary, subject_binary, model, hip_path, amdgpu_targets, bench_prompt, bench_gen, bench_repetitions, ...) takes NO RD04-specific parameters. It builds isolated control/subject worktrees, runs experiment.execution.run_paired_lane() for a decode and a prefill phase, and writes performance.json. Nothing in the function signature or body references RD04 by name.

What makes it RD04-only is a single hardcoded gate in the CLI dispatcher (validation_campaign.py's run(), ~line 3007): `if descriptor.experiment_contract != "RD04-BF16-FLASH-ATTN-TILE": raise PatchCampaignError(f"{args.patch}: --run-rd04-benchmark is RD04-only today")`. The same shape of restriction exists for --run-rd08-lanes/--run-rd08-contract, --run-rd58-state-restore, --run-rd73-contract: each is the SAME underlying 6-check campaign plan (apply, build, performance, activation, correctness, controls) with one patch's own correctness/activation producer wired in, then gated to that one patch id by name.

Real consequence hit live during this session's own work (2026-09-11): validating RD04 (a real, already-promising patch -- prior evidence +3.39% decode tg128 non-overlapping, reconfirmed fresh at the current pin as +3.67% non-overlapping, prefill flat at +0.30%) required using a flag that literally will not run against any other patch id. Promoting/testing the NEXT patch (any of the untested RD/HI/NRO/GP11 patches) would require writing a new --run-rdNN-* flag, a new gate, and copy-pasting the wiring, rather than pointing a generic command at the new patch id.

Also hit live: --run-rd04-benchmark has no device-visibility enforcement (silently inherited whatever HIP_VISIBLE_DEVICES happened to be set in the shell, which was unset the first time and caused a real SIGSEGV crash from ambiguous multi-GPU enumeration across 4 heterogeneous real devices), while --run-rd58-state-restore DOES fail closed on missing/malformed HIP_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES before touching hardware. The per-patch gates aren't just redundant, they're inconsistently safe.

Desired outcome: a single generic paired-benchmark entry point (e.g. `--run-performance-benchmark`, taking any patch id whose experiment/contract wiring already exists in recipes.toml/external-sources.toml) that exposes the already-shared engine directly, with ONE enforced device-visibility contract applied uniformly (not per-patch). Correctness/activation proof legitimately stays bespoke per patch (a kernel's own correctness claim SHOULD differ patch to patch) -- this item is scoped to the performance-pairing harness only, not to unifying correctness proof shapes.

Same topic as PVPS01 (the standard Patch Qualification Matrix, 4 arms x N architectures) -- this item is specifically about the paired-benchmark EXECUTION harness underlying that matrix becoming patch-generic rather than one hardcoded flag per patch; coordinate with PVPS01 rather than duplicating its scope.

## Steps

1. Add DeviceVisibility/require_device_visibility (extracted from RD58's existing guard) + tests. Wire RD58 and all three RD73 lane functions to it. No other behavior change.
2. Extract run_paired_llama_benchmark from the duplicated RD04/RD08 execution logic; keep run_rd04_benchmark_evidence/run_rd08_validation_lanes as compatibility wrappers over it. Existing RD04/RD08 tests must keep passing unmodified at this step.
3. Add the validation.toml wiring resolver (benchmark-executor/benchmark-extra-args on the performance check); add real metadata to RD04 and RD08's validation.toml.
4. Add --run-performance-benchmark itself (generic dispatch on wiring, never on patch id/contract name); build the standard model/architecture matrix with explicit skip-with-reason for inapplicable cells.
5. Add tierM-ministral14b-q4km to config/models.toml with real inspected file metadata. Add the tensor-split benchmark-only projection for tierL-qwen27b-q8.
6. Add --run-rd04-benchmark/--run-rd08-lanes as deprecated aliases onto the generic path; leave --run-rd08-contract/--run-rd58-state-restore/--run-rd73-contract bespoke, only adopting the visibility primitive.
7. Migrate/rewrite the regression suite per the plan above (replace brittle source-string assertions with behavioral tests).
8. Run the full real-hardware merge gate on Brutus (7 real-hardware checks listed above) before considering this item done.

## Detailed Solution & Technical Design

GPT design (2026-09-11, fresh session ses_83c745d8f4ad454a, req_da3d91bf014d419e, read the real d0ec4578 implementation before designing):

CORRECTION to this item's own problem statement: run_rd04_benchmark_evidence() is not parameter-generic as originally claimed -- it hardcodes RD04's `-fa on -ctk bf16 -ctv bf16` flags and RD04-labelled log/context strings; run_rd08_validation_lanes() additionally binds Experiment Contract roles/evidence refs. The reusable part is the paired execution/statistics/env/logging SHAPE, not the functions unchanged as-is.

**1. Generic CLI/wiring**: new `--run-performance-benchmark` flag, plus `--model-root`, repeatable `--benchmark-model`/`--benchmark-architecture` (default: architecture intersection of recipes.toml's platform.linux-multi targets and the patch's own validation-architectures), and required repeatable `--device-map ARCH=ID[,ID...]` (never auto-inferred). Patch benchmark wiring lives on the existing required `performance` `[[check]]` block in validation.toml (not a new top-level table -- schema 1 rejects unknown top-level keys today) via two new opaque validator-config fields: `benchmark-executor = "paired-llama-bench-v1"` and `benchmark-extra-args = [...]`. A patch is generically benchmarkable iff exactly one required capability="performance" check declares a recognized executor; zero/ambiguous/unknown is a configuration error. No patch ID or contract ID appears anywhere in dispatch logic.

**2. Core execution API**: one generic primitive `run_paired_llama_benchmark(control_binary, subject_binary, model, hip_path, architecture, device_count, extra_args, pairs)` returning `PairedBenchmarkResult`, built on a central `llama_bench_lane(workload)` mapping (decode/prefill -> argv + regex, same shape as today) and calling the EXISTING `experiment.execution.run_paired_lane()` unchanged -- it already does alternating A/B order, nonzero-arm failure, metric parsing, block-bootstrap stats, and already accepts an `execution_identity` parameter (pass a real `ExecutionIdentity(backend="ROCm", architectures=(architecture,)*device_count)`) for real per-process architecture/device-count attestation, which is a stronger check than the ad hoc `_require_real_gpu_execution()` used today. `run_rd08_validation_lanes()` stays as a thin adapter (it legitimately does something non-generic: turns measurements into positive/control LaneEffects + Contract evidence refs -- Contract semantics must NOT move into the generic executor). RD73 does not use this executor at all (server-based MTP/decode lanes, shares only the device-safety primitive). RD58 must never be presented as a performance benchmark (it's a correctness/reliability contract with no performance claim).

**3. Device-visibility contract**: extract RD58's existing inline guard into a shared `require_device_visibility(context, env=None, exact_count=None, minimum_count=1) -> DeviceVisibility`, fail-closed on missing/blank/malformed/mismatched/duplicate/too-few HIP_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES (strictly improves on today's RD58 check, which doesn't reject `0,,1`/`0,1,` style malformed lists). Mandatory call sites: the new run_paired_llama_benchmark (exact_count=cell.device_count), RD58's state-restore evidence (minimum_count=2), and all three RD73 lane functions (exact_count=expected device count) -- and for RD73 specifically, copy the validated values explicitly into the attested server session's env_overrides rather than just validating ambient state and assuming inheritance. The matrix dispatcher sets both env vars from --device-map per cell; never auto-infers a device index from architecture.

**4. Refactor/migration order** (safety first, no behavior break mid-sequence):
  1. Add DeviceVisibility/require_device_visibility + tests; move RD58's inline guard into it; wire RD73. Behavior otherwise unchanged.
  2. Extract run_paired_llama_benchmark from the duplicated RD04/RD08 clean-env/runner/command/raw-log/paired-run logic. Keep both existing public functions as compatibility wrappers initially -- current RD04/RD08 tests keep passing untouched. RD04's hardcoded flags move into validation.toml metadata.
  3. Add the wiring resolver (parse benchmark-executor/benchmark-extra-args, fail closed on zero/ambiguous/unknown); add RD04/RD08 metadata.
  4. Add --run-performance-benchmark itself: dispatcher checks wiring, never `descriptor.experiment_contract == "RDNN-..."`. Build one control/subject binary per applicable compile-target set, execute device-scoped matrix cells. Artifact records planned/executed/skipped cells (skipped cells need an explicit reason, never silently omitted), model IDs, architecture, selectors, build identities, commands, raw logs, paired stats.
  5. Compatibility aliases: --run-rd04-benchmark and --run-rd08-lanes become deprecated aliases for the generic path (RD04's alias preserves today's single-model behavior when legacy --model is supplied); --run-rd08-contract's performance phase calls the generic executor but the flag itself stays as the bespoke full-qualification wrapper; --run-rd58-state-restore and --run-rd73-contract stay fully bespoke (only adopt the visibility primitive) -- removing them would violate this item's own explicit scope (correctness/activation/resource proof legitimately stays patch-specific). Do not delete legacy aliases until callers/docs have migrated -- the first PVPS02 change must never be a flag deletion.
  Generic-mode artifact semantics: `passed` means "every applicable cell executed with valid evidence and finite statistics", never "met a Contract threshold" -- generic mode must not populate contract_promotions.

**5. Models**: add `tierM-ministral14b-q4km` to config/models.toml with the REAL relative path/exact size-bytes/verified mtp value (confirmed 2026-09-11: 7.7GB) -- do not invent the metadata, inspect the real file. Reuse tierB-qwen9b-q6k and tierL-qwen27b-q8 as-is (both already exist and are already documented for exactly this purpose). For tierL-qwen27b-q8, reuse the existing production-dual-xtx tensor-split policy but do NOT blindly inject that runtime profile's full server-args into llama-bench (RD73's own docs already record real server-vs-bench CLI differences) -- add/reuse an explicit benchmark-only projection containing just the valid tensor-split args (principally -sm tensor). On Brutus's real current topology the applicable default cells are: single-GPU Ministral/Qwen9B on each of gfx1030/gfx1100/gfx1201, plus Qwen27B on the two-device gfx1100 pair -- other combinations (e.g. Qwen27B on gfx1030 alone, which doesn't fit) must be persisted as skipped/not-applicable with a reason. Keep this benchmark characterization strictly distinct from each patch's own Experiment Contract qualification (RD04's contract names tierA-qwen4b-q6k + long_context; RD08's names tierM-gptoss20b-q6k) -- running the new standard matrix must never be treated as silently satisfying those existing contract bindings.

**Regression plan**: existing tests needing migration/retention: test_patch_validation_campaign_va04.py (RD04), _va14b.py/_va14_final.py (RD08), _va05.py (RD58), _va06b.py/_va06c.py (RD73), _gpu_execution_guard.py, _va15_wiring.py (its regexes explicitly enumerate the legacy special modes and need updating, not just extending). New tests needed: strict visibility parsing (every malformed-list shape, mismatch, duplicate, too-few, wrong exact-count), exact selector preservation into child env, wiring-resolver success/zero/duplicate/unknown, RD04 metadata reproducing today's exact argv, RD08 empty-extra-arg path reproducing today's argv/results, wrong process architecture/count rejected via ExecutionIdentity, architecture-intersection ordering, standard-model matrix incl. explicit inapplicable-cell handling, legacy-alias equivalence, and exact Ministral models.toml path/byte-size validation. Replace brittle source-string assertions ("contains the RD04 contract literal") with real behavioral tests -- several existing tests encode implementation details this refactor intentionally removes.

**Real-hardware merge gate on Brutus** (do not merge without running all of these for real): (1) unset/mismatched/wrong-exact-count device selectors must all fail BEFORE process launch; (2) single-device generic A/B on gfx1030/gfx1100/gfx1201 with Ministral and tierB-qwen9b-q6k, every process attesting its real architecture/count; (3) deliberately map one architecture to the wrong device -- per-process attestation must reject it; (4) tierL-qwen27b-q8 on the real dual-gfx1100 mapping via the existing tensor-split profile, verifying no concurrent control/subject GPU collision; (5) RD04 once via legacy alias and once via the generic entry point on the same arch/model -- commands and evidence schema/stats must match exactly; (6) RD08 generic lanes once to prove its Contract adapter still receives identical LaneEffects; (7) RD58 dual-GPU state-restore and RD73 full-qualification smoke, to prove extracting the visibility primitive didn't regress either bespoke path. Only merge once performance.json records architecture/model/device-visibility for every planned applicable cell and an explicit reason for every skipped cell.

## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria

All items in the real-hardware merge gate pass for real on Brutus. performance.json records architecture/model/device-visibility for every planned applicable cell and an explicit skip-reason for every inapplicable one. RD04 produces identical commands/evidence via the legacy alias and the new generic entry point on the same arch/model. RD08's Contract adapter receives identical LaneEffects post-refactor. RD58/RD73 regression-tested unaffected by the visibility-primitive extraction. Full offline test suite green. Generic-mode artifacts never populate contract_promotions.

## Notes

This item is being handed to dev-gpt-agent for a full design + implementation + refactor/regression plan BEFORE any code is written -- do not implement ad hoc. GPT should read the real current implementation (tools/bigcherry/patch/validation_campaign.py's run_rd04_benchmark_evidence/run_rd08_validation_lanes and the run() dispatcher's --run-rdNN-* gates, tools/bigcherry/experiment/execution.py's run_paired_lane, and the RD58 HIP_VISIBLE_DEVICES enforcement block for the safety-contract shape to generalize) before proposing an API.

This item is being handed to dev-gpt-agent for a full design + implementation + refactor/regression plan BEFORE any code is written -- do not implement ad hoc. GPT should read the real current implementation (tools/bigcherry/patch/validation_campaign.py's run_rd04_benchmark_evidence/run_rd08_validation_lanes and the run() dispatcher's --run-rdNN-* gates, tools/bigcherry/experiment/execution.py's run_paired_lane, and the RD58 HIP_VISIBLE_DEVICES enforcement block for the safety-contract shape to generalize) before proposing an API.

Confirmed 2026-09-11: Ministral-3-14B-Instruct-2512-Q4_K_M.gguf is 7.7GB on disk -- comfortably fits all three real architectures single-GPU (including 16GiB gfx1030), same sizing logic as the existing tierB-qwen9b-q6k cross-architecture reference lane. Needs a real config/models.toml entry (not yet registered) before the generic harness can reference it by model id rather than raw path.

## Change Log

- 2026-09-11T06:35:00.394840+00:00 (created-by): Created by agent
- 2026-09-11T06:35:12.354368+00:00 (updated-by): Updated: section:description, section:notes
- 2026-09-11T06:48:37.813026+00:00 (updated-by): Updated: section:acceptance_criteria
- 2026-09-11T06:48:51.301259+00:00 (updated-by): Updated: section:notes
- 2026-09-11T07:17:49.001543+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:acceptance_criteria
