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

tools/bigcherry/patch/validation_campaign.py's paired-benchmark evidence producers (run_rd04_benchmark_evidence, run_rd08_validation_lanes, the RD58/RD73 equivalents) share a common EXECUTION SHAPE (paired control/subject worktrees, experiment.execution.run_paired_lane, decode+prefill measurement) but are NOT parameter-generic as-is -- run_rd04_benchmark_evidence hardcodes RD04's -fa/-ctk/-ctv flags and RD04-labelled context; run_rd08_validation_lanes additionally binds Experiment Contract roles/evidence refs. (Corrected 2026-09-11 after GPT's design pass read the real code -- an earlier version of this description wrongly claimed RD04's function was already parameter-generic; it is the SHAPE that's reusable, not the function unchanged.)

What makes each one single-patch-only today is a hardcoded gate in the CLI dispatcher (validation_campaign.py's run(), ~line 3007): `if descriptor.experiment_contract != "RD04-BF16-FLASH-ATTN-TILE": raise PatchCampaignError(...)`. Same shape for --run-rd08-lanes/--run-rd08-contract, --run-rd58-state-restore, --run-rd73-contract.

Real consequences hit live during this session's own work (2026-09-11): validating RD04 required a flag that errors on any other patch id; and --run-rd04-benchmark has no device-visibility enforcement (unlike RD58's mode, which fails closed on missing/malformed HIP_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES) -- this caused a real SIGSEGV from unscoped 4-GPU heterogeneous enumeration.

Desired outcome: a single generic paired-benchmark entry point exposing the shared execution shape directly, with ONE enforced device-visibility contract applied uniformly, and standard architecture/model coverage (gfx1030/gfx1100/gfx1201; Ministral-3-14B, tierB-qwen9b-q6k, tierL-qwen27b-q8 with tensor-split). Correctness/activation proof stays bespoke per patch -- out of scope here.

Same topic as PVPS01 (standard Patch Qualification Matrix) -- coordinate rather than duplicate.

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

SECOND-ROUND VALIDATED DESIGN (2026-09-11, same GPT session, self-critique pass, req_5d71b1ec01094e31). Verdict: core architecture sound, NOT implementation-ready without these amendments -- real gaps found, not a rubber stamp.

**Honesty correction (important -- do not overclaim in implementation or evidence records)**: `experiment.execution.run_paired_lane()` really does accept `execution_identity` and really does fail closed on missing attestation / backend mismatch / device-count mismatch / architecture mismatch. BUT `parse_rocm_attestation()` (what llama-bench's own attestation produces) yields `ObservedDevice(..., locator=None)` -- locator is only checked when `expected.locators is not None`, which it never will be here. So the proposed `ExecutionIdentity(backend="ROCm", architectures=...)` proves architecture + device COUNT, not physical identity between e.g. the two real gfx1100 cards. HIP/ROCR selector propagation proves launch INTENT, not physical-device execution attestation. Evidence records and docs must say exactly this, not overclaim physical-device proof.

**Metadata placement**: putting benchmark-executor/benchmark-extra-args on the existing performance `[[check]]` block (vs. a new schema-2 top-level table) is the LOWEST-RISK choice (avoids a schema migration) -- not the only technically possible one. Confirmed validation.toml schema 1 really does rejects unknown top-level keys but passes unknown per-check fields through to CheckSpec.config, which benchmark/autotune-campaign validators currently ignore -- safe today.

**REVISED migration order** (the original 8-step order had a real blast-radius problem: step 1 wired RD58/RD73 -- both currently-working bespoke paths -- into the new safety helper before the new generic path existed or had any tests of its own):
1. Extract + unit-test `require_device_visibility()` ONLY. Do NOT touch RD58/RD73 behavior yet -- RD58's guard currently lives in run(), while run_rd58_state_restore_evidence() stays directly callable WITHOUT selector validation, and several existing tests depend on exactly that separation; RD73's hardware-free lane tests likewise call functions without HIP/ROCR selector setup. Immediate enforcement at this step would break both.
2. PURE semantics-preserving extraction of the shared execution shape into `run_paired_llama_benchmark()`. Do NOT yet enable execution_identity/attestation here -- existing RD04 test fakes emit `found 1 ROCm devices` without the per-device `Device 0 ... gfx...` line parse_rocm_attestation() requires, so turning attestation on immediately breaks currently-passing tests. Do NOT move RD04's flags to validation.toml yet (the resolver doesn't exist until step 3) -- keep them in the compatibility wrapper. The wrappers (run_rd04_benchmark_evidence/run_rd08_validation_lanes) need REAL translation work, not thin pass-through: RD04's wrapper must still inject its flags, build today's exact performance.json/finite-stat passed/campaign-build metadata/return shape; RD08's wrapper must still produce LaneEffects/Contract evidence refs/validation-lanes.json. Split the generic function's command inputs into `patch_args` and `runtime_args` (RD04 orders patch flags before -ngl; RD08's topology extras go after -ngl -- one undifferentiated extra_args underspecifies this).
3. Add the validation.toml wiring resolver (benchmark-executor/benchmark-extra-args); add real RD04/RD08 metadata. Update each step's own tests as it lands -- do NOT defer test migration to a later "step 7", every step must land green.
4. Add --run-performance-benchmark itself with STRICT visibility+attestation from the start (this is where execution_identity/device-visibility actually turn on for the new path only). Requires a real CLI/build early-path (see omission #1 below) -- cannot simply be inserted into the existing RD04 dispatch block, since main()/run() today hard-require --model/--manifest/--amdgpu-targets and build tune/replay/stock/control/subject around one target before any RD dispatch runs; generic mode must skip all of that and build one fat control/subject binary for the joined applicable targets instead.
5. Model/topology resolution (see omissions #2 and #3 below -- this does not exist yet, is not just "add one models.toml row").
6. Migrate legacy aliases: --run-rd04-benchmark onto the generic path (preserving today's behavior via legacy --model). --run-rd08-lanes MUST invoke the generic executor THROUGH RD08's own adapter, not generic matrix semantics directly -- otherwise its Contract model/ref/artifact behavior silently changes.
7. THEN, as its own dedicated regression-tested step, migrate RD58/RD73 onto the visibility helper -- keep enforcement at the orchestration/preflight boundary for RD58 (pass DeviceVisibility.document() into the existing observed_devices= argument rather than forcing the low-level evidence producer to read globals); for RD73, the validated HIP/ROCR values can safely be copied into AttestedServerSession.env_overrides since ServerRunner starts from ambient env then applies overrides. For the new generic path, validate the FINAL child env directly rather than mutating global os.environ between matrix cells.

**Three real omissions found, must be resolved before implementation, not during**:
1. CLI/build architecture: generic mode needs its own early path that skips legacy --model/--manifest/--amdgpu-targets requirements and the unrelated tune/replay/stock campaign builds -- detailed above.
2. Model resolution doesn't exist: today's code only exposes `known_model_ids_from_models_registry()` (returns IDs), not a `--model-root` + id resolver, size verification, or topology resolution. Step 5 needs a real ModelSpec/registry loader with file-size verification, not merely a new TOML row.
3. Device-map/topology semantics are undefined: define `--device-map gfx1100=0,1` as an ORDERED device pool (an N-device cell consumes the first N IDs; order is intentional -- `gfx1100=1,0` picks card 1 for a single-GPU cell and that tensor-split order for a 2-GPU cell); fail/skip if the pool has fewer than N devices. Define model benchmark topology structurally too (a registry key: single vs tensor-2, owning device_count + runtime args) -- otherwise Qwen27B becomes yet another hardcoded model-ID branch; the existing model note saying "dual-GPU/-sm tensor" is prose, not executable config.

**Contract-boundary guard, reconfirmed**: `_builtin_benchmark()` today validates only bound-artifact + nonempty-metrics + passed -- it does NOT inspect model/workload identity. The generic matrix may populate diagnostic performance evidence, but Contract promotion must stay fully separate, as originally intended -- this still holds.

**REVISED real-hardware merge gate** (the original 7-check gate was correctly identified as far too expensive to be a casual pre-merge checklist -- the default matrix alone is 7 cells x 3 pairs x 2 phases = ~84 measured llama-bench launches before ANY RD04/RD08/RD58/RD73 regression checking, and RD73's MTP lane starts a fresh server per arm/request with 2 warmup + 10 measured pairs by default. Treat this as a RESERVED BRUTUS QUALIFICATION WINDOW requiring exclusive use of the gfx1100 pair, not a routine command.) Consolidated real-hardware checks:
- Selector malformed/unset cases: stay OFFLINE (unit-testable) except ONE real fail-before-launch smoke.
- Wrong-architecture device mapping: the negative prelude immediately before the one valid matrix run (not a separate hardware session).
- The Qwen27B dual-gfx1100 check IS one matrix cell -- don't duplicate it as a separate check.
- The RD04 generic run counts as its own matrix cell; only the LEGACY alias counterpart needs to run separately for comparison (not both full re-runs).
- Use pairs=1 for the cross-architecture merge-wiring smoke, plus ONE representative multi-pair cell to verify alternation/statistics -- this gate proves the harness works, it is not a performance-promotion claim.
- Replace "full RD73 qualification" with MINIMAL real execution through each changed RD73 lane (warmup=0/measured=1, decode pairs=1, minimal resource burst) -- since PVPS02 only changes RD73's visibility plumbing, not its lane logic, full qualification tests an unchanged seam at needless cost.

## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria

All items in the real-hardware merge gate pass for real on Brutus. The generic --run-performance-benchmark entry point writes performance-matrix.json (workdir-scoped, distinct from RD04's own per-run performance.json evidence artifact) recording architecture/model/device-visibility for every planned applicable cell and an explicit skip-reason for every inapplicable one. RD04 produces identical commands/evidence via the legacy alias and the new generic entry point on the same arch/model. RD08's Contract adapter receives identical LaneEffects post-refactor. RD58/RD73 regression-tested unaffected by the visibility-primitive extraction. Full offline test suite green. Generic-mode artifacts never populate contract_promotions.

(Corrected 2026-09-11 per GPT review req_e608313764834497, which flagged the original wording -- "performance.json" -- as ambiguous against the real implementation's performance-matrix.json; this is a plan-doc correction, not an implementation change, since the matrix JSON legitimately differs in shape from RD04's own per-patch evidence artifact and should not be conflated with it.)

## Notes

This item is being handed to dev-gpt-agent for a full design + implementation + refactor/regression plan BEFORE any code is written -- do not implement ad hoc. GPT should read the real current implementation (tools/bigcherry/patch/validation_campaign.py's run_rd04_benchmark_evidence/run_rd08_validation_lanes and the run() dispatcher's --run-rdNN-* gates, tools/bigcherry/experiment/execution.py's run_paired_lane, and the RD58 HIP_VISIBLE_DEVICES enforcement block for the safety-contract shape to generalize) before proposing an API.

This item is being handed to dev-gpt-agent for a full design + implementation + refactor/regression plan BEFORE any code is written -- do not implement ad hoc. GPT should read the real current implementation (tools/bigcherry/patch/validation_campaign.py's run_rd04_benchmark_evidence/run_rd08_validation_lanes and the run() dispatcher's --run-rdNN-* gates, tools/bigcherry/experiment/execution.py's run_paired_lane, and the RD58 HIP_VISIBLE_DEVICES enforcement block for the safety-contract shape to generalize) before proposing an API.

Confirmed 2026-09-11: Ministral-3-14B-Instruct-2512-Q4_K_M.gguf is 7.7GB on disk -- comfortably fits all three real architectures single-GPU (including 16GiB gfx1030), same sizing logic as the existing tierB-qwen9b-q6k cross-architecture reference lane. Needs a real config/models.toml entry (not yet registered) before the generic harness can reference it by model id rather than raw path.

Recommended status from the second-round validation: KEEP the design, amend before coding -- do not redesign the core idea. The central decisions (validation-adapter wiring for benchmark metadata, one shared paired-execution primitive, strict fail-closed selector safety, preserved bespoke correctness/activation paths per patch) hold up under critical review. What needed fixing was sequencing safety and explicit orchestration contracts (CLI early-path, model resolution, device-map topology semantics), not the architecture itself.

This item is now design-validated twice (initial design + self-critique). Before implementation begins, decide whether a third pass is wanted or whether this is sufficient to start coding against -- the user's own call, not to be assumed.

Recommended status from the second-round validation: KEEP the design, amend before coding -- do not redesign the core idea. The central decisions (validation-adapter wiring for benchmark metadata, one shared paired-execution primitive, strict fail-closed selector safety, preserved bespoke correctness/activation paths per patch) hold up under critical review. What needed fixing was sequencing safety and explicit orchestration contracts (CLI early-path, model resolution, device-map topology semantics), not the architecture itself.

This item is now design-validated twice (initial design + self-critique). Before implementation begins, decide whether a third pass is wanted or whether this is sufficient to start coding against -- the user's own call, not to be assumed.

IMPLEMENTATION PROGRESS (2026-09-11):
Step 1 DONE (commit 1162f617): require_device_visibility()/DeviceVisibility extracted into tools/bigcherry/experiment/execution.py, 16 new unit tests, nothing wired to RD58/RD73 yet (deliberately, per the revised migration order). Full offline suite clean (only the 3 confirmed pre-existing/unrelated failures remain). Pushed to main/planning-refactor, synced to Brutus.

Remaining steps per the revised order: 2 (pure semantics-preserving extraction of the shared execution shape into run_paired_llama_benchmark(), with real translation-preserving compatibility wrappers for run_rd04_benchmark_evidence/run_rd08_validation_lanes and split patch_args/runtime_args), 3 (validation.toml wiring resolver), 4 (--run-performance-benchmark CLI with its own early path, execution_identity turned on here for the first time), 5 (real model/topology resolution -- ModelSpec/registry loader + Ministral models.toml entry + ordered device-pool semantics), 6 (legacy aliases, RD08's through its own adapter), 7 (RD58/RD73 wired onto the visibility helper as its own dedicated regression-tested step), then the consolidated real-hardware merge gate on Brutus (reserved GPU-exclusive session, not routine).

Recommended status from the second-round validation: KEEP the design, amend before coding -- do not redesign the core idea. The central decisions (validation-adapter wiring for benchmark metadata, one shared paired-execution primitive, strict fail-closed selector safety, preserved bespoke correctness/activation paths per patch) hold up under critical review. What needed fixing was sequencing safety and explicit orchestration contracts (CLI early-path, model resolution, device-map topology semantics), not the architecture itself.

IMPLEMENTATION PROGRESS (2026-09-11):
Step 1 DONE (commit 1162f617): require_device_visibility()/DeviceVisibility extracted, 16 tests. Nothing wired to RD58/RD73 yet (deliberate).
Step 2 DONE (commit 875e6db2): run_paired_llama_benchmark()/_paired_llama_bench_command() extracted from RD04/RD08's duplicated logic, both existing functions now byte-identical-behavior compatibility wrappers, patch_args/runtime_args split preserving each caller's historical argv order, 8 new direct tests, 1 brittle source-inspection test fixed as the design predicted.
Step 3 DONE (commit 6f9a365d): resolve_benchmark_wiring() reads benchmark-executor/benchmark-extra-args from a patch's required performance validation.toml check, fail-closed on zero/ambiguous/unknown wiring. Real metadata wired into RD04 and RD08's actual validation.toml files. 10 new tests covering every failure mode plus both real patches.

All three steps: full offline suite clean (only the 3 confirmed pre-existing/unrelated failures -- telemetry x2, HI104 -- remain), patch-lint/check clean, pushed to main/planning-refactor, synced to Brutus after each step.

Remaining steps per the revised order: 4 (--run-performance-benchmark CLI with its own early path bypassing legacy --model/--manifest/--amdgpu-targets requirements, execution_identity/attestation turned on here for the first time), 5 (real model/topology resolution -- ModelSpec/registry loader + Ministral models.toml entry + ordered device-pool semantics -- does not exist yet, ​not just a TOML row), 6 (legacy aliases, RD08's through its own adapter), 7 (RD58/RD73 wired onto the visibility helper as its own dedicated regression-tested step), then the consolidated real-hardware merge gate on Brutus (reserved GPU-exclusive session, not routine).



STEPS 4 & 5 DONE (commits 228c785d, 4dab7efe) plus a real-hardware-caught bugfix (commit 2923cef0), 2026-09-11:
Step 4: --run-performance-benchmark CLI added (own early path, skips legacy --model/--manifest/--amdgpu-targets requirements; --amdgpu-targets changed from required=True to conditionally-required so the legacy flow still enforces it). execution_identity/require_device_visibility turned on for the first time in this module. _run_performance_benchmark() materializes control/subject source once, builds one control/subject llama-bench per applicable architecture, iterates model cells with explicit skip-with-reason for inapplicable ones, writes performance-matrix.json.
Step 5: ResolvedBenchmarkModel/resolve_benchmark_model() (real file-existence + size-match verification, topology never defaulted), resolve_device_pool()/parse_device_map() (ordered device-pool semantics), tierM-ministral14b-q4km added to config/models.toml with benchmark-topology="single", benchmark-topology added to tierB-qwen9b-q6k ("single") and tierL-qwen27b-q8 ("tensor-2").

BUG FOUND ON FIRST REAL-HARDWARE SMOKE TEST, FIXED (commit 2923cef0): ran --run-performance-benchmark for real on Brutus (gfx1100, tierM-ministral14b-q4km, single device) immediately after step 4 landed -- both control and subject llama-bench binaries built successfully for real, then hit NameError: require_device_visibility used in _run_performance_benchmark() but never imported (only `from bigcherry.experiment import attestation` and `from bigcherry.patch import source as psi` were present). None of the 9 CLI tests written for step 4 caught this -- all 9 were parser-only or source-inspection-only, none actually reached that line. Fixed the missing import, then added OrchestrationLogicTests (3 new tests) that mock build_tree/generate_registry/source-materialization/resolve_benchmark_wiring/resolve_benchmark_model so the real orchestration logic -- device resolution, require_device_visibility()/ExecutionIdentity construction, skip-cell handling -- actually runs hardware-free, closing this coverage gap for future changes to this function. Full offline suite clean (only the 3 confirmed pre-existing/unrelated failures), patch-lint/check clean.

Re-ran the real-hardware smoke test on Brutus after the fix: full end-to-end success -- both binaries built, require_device_visibility()/ExecutionIdentity resolved correctly, 1 cell executed with 0 skipped, real decode/prefill bootstrap stats captured (decode ~+3.7%, prefill ~-7.5% ci geometric_effect_pct for this single-pair informal smoke -- NOT a promotion-grade measurement, pairs=2, purely a plumbing confirmation). Pushed to main/planning-refactor, synced to Brutus.

Remaining steps per the revised order: 6 (legacy aliases --run-rd04-benchmark/--run-rd08-lanes onto the generic path, RD08's through its own Contract adapter), 7 (RD58/RD73 wired onto require_device_visibility as its own dedicated regression-tested step), then the consolidated real-hardware merge gate on Brutus (reserved GPU-exclusive session, not routine).

## Change Log

- 2026-09-11T06:35:00.394840+00:00 (created-by): Created by agent
- 2026-09-11T06:35:12.354368+00:00 (updated-by): Updated: section:description, section:notes
- 2026-09-11T06:48:37.813026+00:00 (updated-by): Updated: section:acceptance_criteria
- 2026-09-11T06:48:51.301259+00:00 (updated-by): Updated: section:notes
- 2026-09-11T07:17:49.001543+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:acceptance_criteria
- 2026-09-11T07:28:55.652156+00:00 (updated-by): Updated: section:description, section:detailed_solution, section:notes
- 2026-09-11T07:59:38.514248+00:00 (updated-by): Updated: section:notes
- 2026-09-11T08:18:36.275347+00:00 (updated-by): Updated: section:notes
- 2026-09-11T09:12:57.017255+00:00 (updated-by): Updated: section:notes

## Ledger-events

- chg_20260911_091302_fixed-a-crash-in-the-new-gener_9057
- 2026-09-11T09:13:02.211977+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-11T09:30:41.738140+00:00 (updated-by): Updated: section:acceptance_criteria
