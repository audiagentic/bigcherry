---
id: RWQ03
order: 3
plan: patching-radiance-wire-quality
state: pending
created-at: '2026-10-15T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: L
---

# Select wire precision by operation family, layer and phase without rank divergence

## Description

Global --tp-wire wht6 is too coarse. Radiance composes fused exact and WHT6 operators at model declaration time, and some exact fused forms have narrower row bounds than lossy forms. A transport-only environment toggle cannot safely turn every fused six-bit operation into exact at run time. Create an explicit deterministic graph policy: exact, WHT8, WHT6 on distinct collective sites. Keep exact the default for Flash-Next until actual KL passes.

## Steps

1. Enumerate every declaration and issue path using arch/qwen35_fp8/qwen35_fp8.cpp, arch/common/rad_arch.h, rad_block_hc.h, rad_block_moe_fp8.h, rad_block_mtp_hc.h and standalone reductions. Produce a CSV/op map keyed by semantic site, layer and execution phase (attention, GDN, MoE, HC, MTP, embedding).
2. First implement family-wide exact-vs-WHT6 ablations with known op handles to find sensitivity; use no new approximate codec in this causal baseline.
3. Design a single canonical policy parser, validated at process initialization, from a profile-specified compact map, e.g. exact default with explicit exceptions for prefill-only operation families/layer ranges. Do not parse untrusted strings or make host per-token policy decisions in GPU dispatch.
4. If layer/phase IDs are unavailable, expose immutable integer op tags in a pinned Radiance patch (new radNNNN package). Declare supported op variants once when building a graph; choose only among declared handles with exact matching input/output contracts. The two ranks must derive the same choice from model layer ID, op family, T and policy ID; reject incompatible policy hashes before sharing scratch.
5. For exact large HC/MoE paths where the fused kernel cannot serve the declared shape, use a correct standalone all_reduce + nonfused HC/gather chain; record the lost fusion explicitly in performance results. Do not issue zero/unsupported op handles.
6. Add policy-coverage tests for every layer 0..47 plus draft/vision, M={1,2,4,8,64,512,1024,2048,4096}; verify helper-on and off, row gating, static graph replay; then ABBA validate in RWQ05.

## Detailed Solution & Technical Design

Proposed selector inputs: model family/version, phase={prefill,decode,verify}, semantic op family, layer index (or -1 for non-layer ops), row M, word width K, rank_count, supports_exact, supports_wht6, supports_wht8. Output is an enum with a guaranteed advertised kernel and a decision reason. Precedence: exact mandated for disjoint embedding gathering, unsupported shapes, decode/verify small messages, protected operations, or unrecognised policy; next a pinned per-layer override; next a per-family override; finally safe model default (Flash-Next exact). Policies resolve identically on all ranks; no runtime per-rank quantisation error sampling to select policy. Variant-specific scratch capacity is checked before launch.
Potential design in graph declaration: each semantic site has EXACT/WHT8/WHT6 handles where supported, site_id (phase|layer|family), and runtime issues one handle after ar_wire_is_exact/size gate. Uniform layer/phase routing avoids dependence on rank-local activations. Existing helpers in rad_block_hc must be adapted so a gather's consumer knows which wire owns the reduction and never doubles or skips an all-reduce.

## Code Samples & Guidance

Example policy contract (illustrative, not an accepted default):
model=qwen38-flash-next; default=exact; min_rows=512; allowed_only=prefill; overrides=[{family:ar_rmsnorm_quant_fp8, layers:0-47, mode:wht8}].
Parser must reject unknown layers/modes, overlapping conflicting rules, missing a kernel variant and mismatch of model identity. Decision metadata must state why fallback exact occurred.

## Files

New engine patch package under engines/radiance/patches/ (post-inventory ID); engines/radiance/components/libr3/native/r4d_ar_entry.hip.rw and component.toml; engines/radiance/recipes.toml (experiment, not production); tools/tests/patch/test_radiance_patch_packages.py; new policy unit tests; RWQ03 coverage artifact.

## Validation

Offline matrix enumerating all 48 layers and all phase/shape/supported-mode combinations and verifying two simulated rank decisions identical. Engine patch PIN_BLOBS hash, idempotence, missing-anchor atomicity, full pinned tree compile tests. Runtime trace every site once for each width; real Brutus comparison exact-vs-exact and isolated site ablations; ensure full model does not accidentally use WHT6 or WHT8 under an 'exact' policy.

## Effort & Risk

High: graph declaration/selection can silently skip/add collectives; fused/nonfused replacement changes arithmetic; wrong rank policy deadlocks. Start with per-family selector, then only add layer granularity if profiling justifies it. If every sensitive site is exact and speed benefit disappears, reject this line.

## Standards

- Target Radiance 1.3.0 pinned at 89cee7ce414486c717aa2085c99e1f522e4c0ea6; no unpinned edits.
- Engine edits are explicit Radiance patch packages; gfx1100 kernels and settings belong to the libr3 component. Keep opt-ins OFF unless independently promoted with quality evidence.
- Use engine-specific recipes, source/build identities, runtime profiles containing only settings different from declared defaults, and existing bigcherry ab-benchmark/engine_quality evidence facilities. No second campaign or loader.
- Follow docs/reference/tooling/TOOLING.md, docs/reference/engines/RADIANCE_RUNBOOK.md, patch mechanics, relevant device checker, and main-branch code review requirements.
- Changes must preserve both-rank numerical agreement, graph replay and no-P2P mapped-pinned-host protocol; fail closed on missing kernel support or disagreement.

## Acceptance Criteria

- For every reachable model site and phase there is one and only one reduction route, with correct post-op.
- Both ranks log the same policy ID and op/mode selection; mismatched hashes are rejected before serving.
- Model identity not changed, no runtime defaults re-stated in profiles, and exact is default for Flash-Next.
- Per-op/per-layer quality contribution measured and RWQ05 gates available.

## Notes

Depends RWQ01 and preferably RWQ02; can start exact-vs-WHT6 family isolation before WHT8 is ready. Explicitly do not conflate the global --tp-wire setting with the internally dispatched fused op variant.

## Change Log

- 2026-10-15T00:00:00+00:00 (created-by): Initial plan; unimplemented GPU work remains pending.
