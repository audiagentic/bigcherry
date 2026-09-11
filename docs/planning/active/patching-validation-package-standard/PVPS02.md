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

Additional real requirement (user, 2026-09-11): the generic harness must exercise ALL applicable real architectures, not just one. Brutus's real, confirmed hardware (rocminfo, 2026-09-11): gfx1030 (RX 6900 XT, 16GiB), gfx1100 (RX 7900 XTX x2, 24GiB each), gfx1201 (32GiB). config/recipes.toml's platform.linux-multi already declares targets = ["gfx1100", "gfx1201", "gfx1030"] -- the generic harness should default to this same set (or whatever subset a given patch's own validation-architectures narrows it to), not a single hardcoded arch.

Standard model set (user-specified, 2026-09-11), with tensor-split where the model requires more than one device:
- Mistral family: Ministral-3-14B-Instruct-2512-Q4_K_M.gguf (used directly by path in the RD04 bench so far; NOT YET registered in config/models.toml -- needs a real models.toml entry, e.g. tierM-ministral14b or similar, following the existing tierA/tierB/tierL id convention). ~8-9GB Q4_K_M, should fit all three real architectures single-GPU (including the 16GiB gfx1030) -- confirm this against real GGUF file size before assuming, don't guess.
- Qwen 9B: already registered as tierB-qwen9b-q6k (Qwen3.5-9B-Q6_K.gguf) -- an existing, deliberately-designed cross-architecture reference lane (its own models.toml note explains it was sized specifically to fit gfx1030/gfx1100/gfx1201 with headroom, dense + MTP-capable, stock Q6_K to avoid an unusual quant as a hidden variable). Reuse this lane as-is, do not create a duplicate.
- Qwen 3.8 27B: already registered as tierL-qwen27b-q8 (Qwen3.8-27B-Q8_0.gguf) -- the real production dual-XTX model, requires tensor split (-sm tensor) across 2+ devices per the project's own dual-XTX baseline memory; too large for gfx1030 alone (29GB vs 16GiB). Use the existing production-dual-xtx runtime-profile's tensor-split config, do not invent a new one.

Documentation requirement (user, 2026-09-11): if a patch's validation needs a specific model/quant not already covered by the standard set above, document the exact model/quant requirement in that patch's own docs (README.md/SUMMARY.md), and source it preferring unsloth or another mainstream GGUF publisher on Hugging Face -- do not invent a bespoke/obscure quant source without documenting why the standard set doesn't cover the need.

## Notes

This item is being handed to dev-gpt-agent for a full design + implementation + refactor/regression plan BEFORE any code is written -- do not implement ad hoc. GPT should read the real current implementation (tools/bigcherry/patch/validation_campaign.py's run_rd04_benchmark_evidence/run_rd08_validation_lanes and the run() dispatcher's --run-rdNN-* gates, tools/bigcherry/experiment/execution.py's run_paired_lane, and the RD58 HIP_VISIBLE_DEVICES enforcement block for the safety-contract shape to generalize) before proposing an API.

This item is being handed to dev-gpt-agent for a full design + implementation + refactor/regression plan BEFORE any code is written -- do not implement ad hoc. GPT should read the real current implementation (tools/bigcherry/patch/validation_campaign.py's run_rd04_benchmark_evidence/run_rd08_validation_lanes and the run() dispatcher's --run-rdNN-* gates, tools/bigcherry/experiment/execution.py's run_paired_lane, and the RD58 HIP_VISIBLE_DEVICES enforcement block for the safety-contract shape to generalize) before proposing an API.

Confirmed 2026-09-11: Ministral-3-14B-Instruct-2512-Q4_K_M.gguf is 7.7GB on disk -- comfortably fits all three real architectures single-GPU (including 16GiB gfx1030), same sizing logic as the existing tierB-qwen9b-q6k cross-architecture reference lane. Needs a real config/models.toml entry (not yet registered) before the generic harness can reference it by model id rather than raw path.

## Change Log

- 2026-09-11T06:35:00.394840+00:00 (created-by): Created by agent
- 2026-09-11T06:35:12.354368+00:00 (updated-by): Updated: section:description, section:notes
- 2026-09-11T06:48:37.813026+00:00 (updated-by): Updated: section:acceptance_criteria
- 2026-09-11T06:48:51.301259+00:00 (updated-by): Updated: section:notes
