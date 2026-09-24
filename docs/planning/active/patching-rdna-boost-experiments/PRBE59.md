---
id: PRBE59
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:57:36.613381+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# FORK-HIP-001: Restore/benchmark rocWMMA FlashAttention on RDNA4

## Description

TODO. Same GPT-timing situation as PRBE57/58. Relevance: no existing patch for RD76 (grep = no hits). Confirmed NOT upstream-absorbed and NOT currently used: grep for "rocwmma"/"ROCWMMA" across ggml/src/ggml-cuda at b11126 returns zero hits -- the current WMMA FlashAttention path (fattn.cu, ggml_cuda_get_best_fattn_kernel) uses native `amd_wmma_available(cc)` (common.cuh:356-357, gates on RDNA3/RDNA4 cc) via inline WMMA intrinsics, NOT the rocWMMA library. Source commit 5aa2f049 (per item notes) is from an external fork, not locally available. Disposition: TODO -- a genuine research-compare task (per the item's own framing: 'research-compare', 'keep conditional and experimental unless stably superior'), not a guaranteed-win port.

## Steps

1. Fetch/inspect the source commit 5aa2f049 and its build-integration approach (how it links rocWMMA -- likely via ROCm's rocwmma header-only library, requiring a CMake find_package/include-path addition this project's build does not currently have, since grep confirms zero current rocWMMA usage). 2. Design a build-time or runtime SELECTION gate letting both the current native-WMMA path and a new rocWMMA-based kernel coexist: an env var (e.g. GGML_CUDA_FA_ROCWMMA=1) selected at dispatch time in ggml_cuda_get_best_fattn_kernel, alongside the existing amd_wmma_available(cc)/GGML_CUDA_FA_WMMA_256 gating already there (fattn.cu, confirmed present). 3. Restrict the rocWMMA candidate path to RDNA4 (R9700/gfx1201) deep-context shapes specifically (32K/64K/128K) per the item -- do not attempt to replace the native path for shorter contexts or other architectures where no evidence of benefit exists. 4. Add the CMake/build integration for linking rocwmma (conditional on a build flag, not unconditional, to avoid a new hard dependency for users not opting into this experimental path). 5. Add FA backend-op correctness tests for the new kernel path (existing test-backend-ops FLASH_ATTN_EXT cases, run against both kernel selections). 6. Add long-context PPL and long-run stability tests (both paths, compare). 7. Measure PP/TG, VRAM/scratch usage, compile time, runtime stability, and graph-capture interaction (CUDA/HIP graph capture + a new kernel candidate is a real risk area -- confirm the rocWMMA kernel is graph-capture-compatible before enabling it under GGML_CUDA_GRAPH_OPT).

## Detailed Solution & Technical Design

This is explicitly framed by the item as a COMPARISON, not a replacement: 'keep conditional and experimental unless stably superior' and 'gate by head size, q_rows, depth, KV type and graph state; retain current kernel fallback.' The current native WMMA path (amd_wmma_available + the WMMA config table this session already inspected in patches/1203_.../patch.py, e.g. GGML_CUDA_FATTN_MMA_CONFIG_CASE tuning for heads 256/320/512/576) is itself already tuned RDNA4-specific work; rocWMMA is a DIFFERENT lowering (library-based, via ROCm's warp-matrix-multiply-accumulate abstraction rather than raw inline WMMA intrinsics) that may or may not generate better code on a given ROCm/driver version -- this is inherently an empirical question requiring the build-integration and hardware run this planning pass cannot perform.

## Code Samples & Guidance

No real anchors for the rocWMMA integration itself (external fork not locally mirrored, and zero existing rocWMMA references in-tree to anchor against). Real b11126 anchor for the EXISTING selection point this new path must hook into (fattn.cu, confirmed this session, same function 1203's rd06-wmma-gating edits target):\n```cpp\nconst char * wmma_256_env = getenv("GGML_CUDA_FA_WMMA_256");\nconst bool wmma_256 = wmma_256_env == nullptr || std::atoi(wmma_256_env) != 0;\nconst int wmma_max_head = (wmma_256 && GGML_CUDA_CC_IS_RDNA4(cc)) ? 576 : 128;\nif ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= wmma_max_head) ...) {\n    return BEST_FATTN_KERNEL_MMA_F16;\n}\n```\n(Note: this exact shape only exists if patch 1203's rd06-wmma-gating edit is applied -- 1203 is REJECTED, so the actual unpatched b11126 base is simpler: `if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 128) ...)`, confirmed via 1203's own `_WMMA_OLD` anchor text read this session.) A new BEST_FATTN_KERNEL_MMA_F16_ROCWMMA enum value + dispatch case, gated by depth/KV-type/env, would sit alongside this existing selection logic.

## Files

ggml/src/ggml-cuda/fattn.cu (kernel selection), new ggml/src/ggml-cuda/fattn-rocwmma*.cuh (new kernel, contingent on fetched source), CMakeLists.txt (conditional rocwmma link), tests/test-backend-ops.cpp (FA correctness cases for the new path), new package patches/<order>_rd76_rocwmma_fa_compare/ (experimental/conditional state, not a default-on change).

## Validation

Correctness: FA backend-op tests, PPL, long-run stability for both paths. Performance: PP/TG at 32K/64K/128K, VRAM/scratch delta, compile/runtime stability, graph-capture interaction (must not break GGML_CUDA_GRAPH_OPT). Acceptance per the item's own bar: keep ONLY as a conditional/experimental path if correctness+stability+clear deep-context PP benefit are all repeatable; otherwise retain the current native-WMMA-only path and document the negative result.

## Effort & Risk

Unscored by item; set to M-L -- build integration for a new external library dependency plus a genuinely uncertain performance outcome (this is real comparative research, per the item's own framing, not a guaranteed win).

## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance; experimental/conditional gating only, current kernel fallback always retained.

## Acceptance Criteria

Keep only as a conditional experimental path if correctness, stability, and clear deep-context PP benefit are repeatable; otherwise retain current FA path.

## Notes

Supersedes: RD76
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd76

2026-09-24 relevance at b11126: no existing patch for RD76 (grep = no hits); confirmed rocWMMA is not used anywhere in-tree today (zero grep hits for rocwmma/ROCWMMA across ggml-cuda), and the current native-WMMA FA selection point (fattn.cu, amd_wmma_available/GGML_CUDA_FA_WMMA_256) was directly re-verified this session (also cross-referenced against patch 1203's rejected rd06-wmma-gating edit, which targets the exact same selection function). External fork source (commit 5aa2f049) not locally available. GPT design request req_83fbfa0995034d2d (covering this + PRBE57/58/101) was in progress when this plan was authored; check for its response and merge if useful.

2026-09-24 GPT req_83fbfa0995034d2d COMPLETED. Its design is more concrete than this plan's sketch: isolates rocWMMA in a NEW translation unit (fattn-rocwmma.cu/.cuh) behind a CMake option GGML_HIP_ROCWMMA (native path always still built), a runtime selector env GGML_HIP_FA_IMPL=auto|native|rocwmma defaulting effectively to native until hardware evidence exists, and a concrete RDNA4 candidate shape matrix (KV depth 32K/64K/128K, head 64/128 first then 256, GQA 1/2/4/8, Q rows 1-16+prefill, F16 K/V first) with an explicit caution: 'most useful search space... aligned F16 K/V, head 64/128, GQA 4/8' and 'do not assume long context automatically favors rocWMMA; bandwidth can dominate' -- a sharper prior than this plan had. Promotion bar: correctness green + no PPL regression + stable long runs + repeatable E2E gain on a NAMED shape predicate (not a blanket claim). Prefer this design when implementing; build-integration details (CMake find_package) still need real verification against this project's build system.

## Change Log

- 2026-09-09T10:57:36.613381+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:14:47.897740+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.393793+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.204866+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:13:33.193064+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031346_repaired-four-more-migrated-pa_4345
- 2026-09-10T03:13:46.522718+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T04:52:43.672823+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
- 2026-09-24T04:53:46.389230+00:00 (updated-by): Updated: section:notes
