---
id: BHA02
order: 1
plan: build-hip-autotune
state: pending
created-at: '2026-09-11T22:57:11.143052+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# Selective HIP Flash-Attention quant build configuration (#28079)

## Description

Assess and, if applicable, adopt upstream PR #28079's selective FA quant compilation for BigCherry HIP builds to reduce build cost while preserving explicit fallback semantics. Freeze the supported quant-pair allowlist from actual model/config inventory before implementation.

## Steps

1. Resolve merged PR #28079 and current pinned release ancestry; record pin/source identity.
2. Inventory actual KV quant combinations used by supported models and distinguish HIP/CUDA-supported combinations from safe F16 fallback.
3. Add GGML_CUDA_FA_QUANTS through config/recipes.toml and canonical BuildPlan/cmake_configure_args(), not ad hoc flags.
4. Extend effective configure identity/capture so this setting is verified against CMakeCache.txt and changes reuse identity.
5. Verify supported pairs, unsupported-pair fallback and compile/runtime correctness before measuring impact.

## Detailed Solution & Technical Design

This is a build configuration/identity task, not a performance claim. Reuse recipes, BuildPlan, effective build identity, compile checks and validation evidence. Do not assume IQ4_NL is supported merely because a model requests it. Preserve explicit fallback and no identity collisions.

## Code Samples & Guidance



## Files

config/recipes.toml; tools/bigcherry/build/builds.py; tools/bigcherry/campaign/build.py; tools/tests/build/test_build_identity.py; tools/tests/build/test_build_descriptor.py; tools/tests/build/test_autotune_catalog_compile_input_stability.py; tools/tests/patch/test_recipes.py; docs/reference/build/**

## Validation

PR ancestry/current-pin audit; recipe-to-CMake propagation; build identity change on config change; CMakeCache verification; compile matrix for declared FA pairs; unsupported/fallback behavior; representative HIP correctness; compile-time comparison.

## Effort & Risk



## Standards

Canonical recipe/build identity; no hand-added campaign flags; explicit fallback; no duplicate implementation if pin already contains equivalent behavior.

## Acceptance Criteria

Selective FA compilation is represented by canonical recipe/build identity, exact quant allowlist is recorded, supported pairs compile and execute, unsupported pairs fall back safely, and no evidence identity collision exists.

## Notes

Provenance: shared ChatGPT conversation, 11 Sep 2026, '#28079 — use selective FA quant compilation for HIP builds'; source https://github.com/ggml-org/llama.cpp/pull/28079. Report suggested f16-f16;q8_0-q8_0;q8_0-q5_1;q5_1-q5_1 but this is a hypothesis until actual model/config inventory confirms it.

GPT order: independent build slice after PA21/BRC01/PA22 plan corrections. Provenance: llama.cpp PR #28079; report hypothesis f16-f16;q8_0-q8_0;q8_0-q5_1;q5_1-q5_1 remains unaccepted until inventory confirms it.

GPT roadmap provenance: request req_f7a013040828433c, same session ses_76206cac3e6b4be0, exact pushed bb20f104. This lands before PA20 recipe migration only if the current-pin/PR #28079 audit proves an actual recipe or build-identity change. If the pinned code already has equivalent behavior or the supported quant inventory does not justify a change, close BHA02 as no-op and leave BRBC01 dormant. recipes.toml is single-writer here; do not overlap PA20 recipe edits. When activated, use BRBC01's independent BuildPlan/ArtifactStore/check-parity oracle rather than building another planner.

## Change Log

- 2026-09-11T22:57:11.143052+00:00 (created-by): Created by agent

## Ledger-events


- chg_20260911_225756_added-six-provenance-rich-buil_2622
- 2026-09-11T22:57:56.715001+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T10:31:36.533030+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes
- chg_20260912_103200_updated-the-active-buildrunp_8222
- 2026-09-12T10:32:01.029391+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T18:54:11.881010+00:00 (updated-by): Updated: order=1, section:notes
- chg_20260912_185547_recorded-the-gpt-guided-non-vu_6004
- 2026-09-12T18:55:48.025832+00:00 (updated-by): Updated: section:ledger-events
