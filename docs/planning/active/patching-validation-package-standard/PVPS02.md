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



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

This item is being handed to dev-gpt-agent for a full design + implementation + refactor/regression plan BEFORE any code is written -- do not implement ad hoc. GPT should read the real current implementation (tools/bigcherry/patch/validation_campaign.py's run_rd04_benchmark_evidence/run_rd08_validation_lanes and the run() dispatcher's --run-rdNN-* gates, tools/bigcherry/experiment/execution.py's run_paired_lane, and the RD58 HIP_VISIBLE_DEVICES enforcement block for the safety-contract shape to generalize) before proposing an API.

## Change Log

- 2026-09-11T06:35:00.394840+00:00 (created-by): Created by agent
- 2026-09-11T06:35:12.354368+00:00 (updated-by): Updated: section:description, section:notes
