---
id: PRBE64
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:57:58.168957+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# VK-ACO-001: Audit redundant RADV/ACO scalar waits

## Description

TODO. Research-oracle work: no BigCherry patch or upstream absorption found for RADV/ACO scalar-wait auditing (grep of patches/*/patch.toml for RD81 finds nothing; this is shader ISA work in ggml/src/ggml-vulkan/vulkan-shaders/*.comp, not source we vendor as C++/CUDA). Still relevant to gfx1100/gfx1030 RADV-driven decode. Disposition: TODO, scoped as an offline tooling+decision-gate procedure, not a code patch until/unless a minimal shader rewrite is proven to help.

## Steps

1. Under tools/lab/vk-aco-wait-audit/, write capture.sh: for each of the Q4_K and Q8_0 GEMV compute shaders (ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec.comp and mul_mat_vecq.comp / quantized variants -- confirm exact filenames with `ls ggml/src/ggml-vulkan/vulkan-shaders/*mul_mat_vec*`), dump ACO ISA via `RADV_DEBUG=nofastio,nooptimizer... ACO_DEBUG=validateir,validatera AMD_DEBUG=nir ` combined with `ACO_DUMP_SHADERS=1` (or the current Mesa build's equivalent dump env vars -- confirm exact names for the installed mesa version with `radv-dump --help` / mesa docs, mesa version drift is expected) run against a minimal llama-bench invocation that dispatches each shader once.
2. Capture equivalent proprietary-driver (AMDVLK) ISA for the same shaders if AMDVLK is installed, plus one unaffected kernel (e.g. an F16 GEMV shader) as a within-driver control.
3. Count `s_waitcnt` occurrences and classify (vmcnt/lgkmcnt/expcnt) per shader per driver; record effective bandwidth and decode t/s via llama-bench at the same shapes.
4. Try minimal source transformations in a scratch copy of the .comp shader (e.g. reordering scalar loads, hoisting a shared read, adding an explicit barrier) one change at a time; re-dump ISA after each to see if the wait pattern actually changes (ACO codegen is not guaranteed to respond to source reordering).
5. Decision gate: port a change into ggml/src/ggml-vulkan/vulkan-shaders/ as a BigCherry patch ONLY if a transformation (a) reliably drops wait count across repeated dumps/compiler versions and (b) measurably improves bandwidth/t-s on real hardware with no correctness regression (test-backend-ops parity). Otherwise write up a Mesa/RADV issue with the minimal repro (shader source snippet + ISA dump + mesa version) and file it under tools/lab/vk-aco-wait-audit/mesa-issue.md; do not touch source.

## Detailed Solution & Technical Design

This is oracle/audit work, not a design-ahead code change: the outcome is data-driven (either a proven minimal shader edit, or a documented Mesa issue). No anchors are given here because no edit is authorized until step 5's evidence exists. If a rewrite is later approved, it becomes a new BigCherry patch touching the specific .comp file(s) under ggml/src/ggml-vulkan/vulkan-shaders/, built through the project's existing GLSL->SPIR-V generation pipeline (see ggml/src/ggml-vulkan/CMakeLists.txt for the shader compile step) -- that follow-on patch is out of scope for this item.

## Code Samples & Guidance

capture.sh skeleton:
```bash
#!/usr/bin/env bash
set -euo pipefail
SHADER_DIR=ggml/src/ggml-vulkan/vulkan-shaders
OUT=tools/lab/vk-aco-wait-audit/isa
mkdir -p "$OUT"
for drv in radv amdvlk; do
  for shape in q4_k q8_0 f16_control; do
    echo "== $drv $shape =="
    RADV_DEBUG=${drv:+} ACO_DUMP_SHADERS=1 AMD_DEBUG=nir \
      ./build/bin/llama-bench -m <tiny-model> -p 0 -n 1 -ngl 99 2> "$OUT/${drv}_${shape}.isa.txt"
  done
done
```
(exact env-var names must be confirmed against the installed mesa/RADV version before running -- do not assume the names above are final.)

capture.sh skeleton, corrected per GPT review: the real Q4_K shader is `mul_mat_vec_q4_k.comp` (confirmed present at ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q4_k.comp). Prior draft's `RADV_DEBUG=${drv:+}` did NOT select a driver (it's a no-op env-var trick) and AMDVLK is not an ACO-based compiler (ACO is RADV-specific; AMDVLK uses LLPC) -- drop the AMDVLK/ACO comparison, compare RADV(ACO) against llvmpipe/software or a differently-optimized RADV build instead if a within-driver control is needed. Driver selection must use `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json` (RADV) vs the AMDVLK equivalent ICD json, not RADV_DEBUG. Each loop iteration must dispatch a DIFFERENT real op/shape (not the same fixed <tiny-model> command every time) to actually force each named shader -- use test-backend-ops with an explicit MUL_MAT_VEC(Q4_K,...) case, or a llama-bench invocation pinned to a model whose Q4_K decode path is guaranteed to dispatch mul_mat_vec_q4_k.comp specifically (confirm via ggml_vk_should_use_mmvq's routing, see PRBE55/61's verified anchors in this same batch), not a generic tiny-model smoke run:
```bash
#!/usr/bin/env bash
set -euo pipefail
OUT=tools/lab/vk-aco-wait-audit/isa
mkdir -p "$OUT"
# RADV only (AMDVLK dropped -- not ACO-based, not a valid ACO comparison point)
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
  ACO_DEBUG=validateir,validatera ACO_DUMP_SHADERS=1 \
  ./build/bin/test-backend-ops perf -o MUL_MAT_VEC -b q4_k 2> "$OUT/radv_q4_k.isa.txt"
```
(exact ACO dump env-var names must still be confirmed against the installed mesa version before running -- this remains a required pre-step, not assumed final.)

## Files

tools/lab/vk-aco-wait-audit/capture.sh; tools/lab/vk-aco-wait-audit/isa/*.txt (captured evidence, not committed if large); tools/lab/vk-aco-wait-audit/mesa-issue.md (if no rewrite qualifies); ggml/src/ggml-vulkan/vulkan-shaders/*.comp (only if a rewrite is later authorized -- separate follow-up patch).

## Validation

Output parity via existing test-backend-ops MUL_MAT_VEC cases for Q4_K/Q8_0 before/after any shader edit; ISA wait-count and llama-bench decode t/s recorded per driver/shape in tools/lab/vk-aco-wait-audit/results.md. Hardware runs happen on Brutus (RADV is the default driver there); do not run hardware from this planning pass.

## Effort & Risk

S/basic-tooling to set up; the shader ISA experimentation itself is exploratory and may yield no portable fix (explicitly an acceptable outcome -- Mesa issue path).

## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Port only a minimal rewrite with reliable ACO codegen and runtime improvement; otherwise preserve code and record a reproducible Mesa issue.

## Notes

Supersedes: RD81
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd81

2026-09-24 relevance at b11126: TODO, no existing patch (grep patches/*/patch.toml for RD81/ACO finds nothing) and no upstream absorption possible to check generically (shader-ISA-level, driver-dependent). GPT design request: gateway rejected all submissions this session (VAL-AGW-025, EXT-GPTAUTO-003 in agent_task_gateway_overview); plan authored directly from repo layout inspection (ggml/src/ggml-vulkan/vulkan-shaders/ confirmed present) -- no GPT request id.

2026-09-24 GPT review req_d55aed71224e43a8 applied: NOT-READY -- corrected real shader name (mul_mat_vec_q4_k.comp, confirmed present in vendor tree); removed invalid RADV_DEBUG driver-selection trick and the AMDVLK-as-ACO-comparison error (AMDVLK is LLPC-based, not ACO); fixed capture loop to dispatch distinct real ops per shader instead of the same tiny-model command.

## Change Log

- 2026-09-09T10:57:58.168957+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:15:07.938367+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.415531+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.240506+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:17:31.880934+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031805_repaired-four-more-graph-and-v_2834
- 2026-09-10T03:18:05.321494+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:31:24.845486+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
- 2026-09-24T04:48:07.869768+00:00 (updated-by): Updated: section:code_samples
- 2026-09-24T04:48:13.655935+00:00 (updated-by): Updated: section:notes
