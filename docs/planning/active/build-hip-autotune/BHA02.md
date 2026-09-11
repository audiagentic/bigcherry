---
id: BHA02
order: 2
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

Assess and, if applicable, adopt upstream PR #28079's selective FA quant compilation for BigCherry HIP builds to reduce build cost while preserving explicit fallback semantics.

## Steps

1. Resolve merged PR #28079 and current pinned release ancestry. 2. Inventory actual KV quant combinations used by supported models and distinguish HIP/CUDA-supported combinations from silent F16 fallback. 3. Add the selective GGML_CUDA_FA_QUANTS configuration through the canonical BuildPlan/recipe path, not ad hoc shell flags. 4. Verify generated build identity includes the configuration and that unsupported combinations retain safe fallback. 5. Measure compile-time and run-time impact only after build/correctness checks.

## Detailed Solution & Technical Design

The shared report describes #28079 as merged and clarifies GGML_CUDA_FA_ALL_QUANTS is superseded by GGML_CUDA_FA_QUANTS. This is a build configuration/identity task, not a performance claim. Reuse existing recipes, BuildPlan, effective build identity, compile checks, and validation evidence. Do not assume IQ4_NL is a supported HIP/CUDA KV quant merely because a model requests it.

## Code Samples & Guidance



## Files

config/recipes.toml; tools/bigcherry/build/builds.py; tools/bigcherry/build/toolchain.py; tools/bigcherry/campaign/build.py; tools/tests/build/**; tools/tests/campaign/**; docs/reference/build/**

## Validation

PR ancestry/current-pin audit; recipe-to-CMake propagation; build identity changes on config change; compile matrix for declared FA quant pairs; explicit unsupported/fallback behavior; no silent omission; representative HIP runtime correctness and compile-time comparison.

## Effort & Risk



## Standards



## Acceptance Criteria

Selective FA quant compilation is represented by canonical recipe/build identity, with tested fallback and no evidence identity collisions. If current pin already contains equivalent behavior, record baseline coverage and do not add a duplicate implementation.

## Notes

Provenance: shared ChatGPT conversation, 11 Sep 2026, '#28079 — use selective FA quant compilation for HIP builds'; source https://github.com/ggml-org/llama.cpp/pull/28079. Report suggested f16-f16;q8_0-q8_0;q8_0-q5_1;q5_1-q5_1 but this is a hypothesis until actual model/config inventory confirms it.

## Change Log

- 2026-09-11T22:57:11.143052+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260911_225756_added-six-provenance-rich-buil_2622
- 2026-09-11T22:57:56.715001+00:00 (updated-by): Updated: section:ledger-events
