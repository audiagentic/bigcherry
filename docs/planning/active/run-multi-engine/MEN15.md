---
id: MEN15
order: 1
plan: run-multi-engine
state: pending
created-at: '2026-10-17T02:58:13.549094+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# RAM-only engine smoke tests: tiny models on CPU for llama.cpp and radiance, in the bump and change gates

## Description

Owner, 2026-10-17: quick, RAM-only smoke tests for both engines so bumps and other changes catch basic engine breakage without needing the GPUs (which may be occupied). Each engine builds a CPU-capable variant (or runs its normal build with no device visible), loads a tiny model, serves a few requests, and checks output sanity and determinism - seconds to a couple of minutes, no GPU, small RAM.

## Steps

1. llama.cpp: a CPU build lane (GGML_HIP off) or the HIP build with HIP_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES empty and -ngl 0; a tiny GGUF (e.g. a ~15-150M-parameter model, pinned by sha256 in config) fetched once to the lab model store.
2. radiance: check its CPU path (radiance_home has libavx / libref kernel plugins) - run with --kernels libavx,libref (or what radiance requires for CPU) and a tiny .rad container: find a published tiny container, else convert a tiny model with radiance's quantizers once and pin it.
3. Engine adapter method `cpu_smoke(context, build) -> SmokeSpec` (binary, args, env with no GPU visible, model, expected checks); generic runner in bigcherry: start server, health, 2-3 short greedy completions, check non-empty, deterministic across two runs, no GPU device opened (assert HIP device count 0 / env empty), graceful shutdown, time + peak RSS recorded.
4. Wire it: `bigcherry --engine <name> smoke --cpu`; pin-bump runs it as a phase before any GPU smoke (MEN10 step f); patch/component changes can run it locally and in CI where the build is cheap enough.
5. Docs in RELEASES.md / PIN_BUMP.md / engine READMEs.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Both engines pass the CPU smoke on dev with all GPUs busy (or devices hidden) in under ~2 minutes each; a deliberately broken patch (e.g. a crash on load) fails it; offline unit tests for the runner and adapters.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Sequence: after MEN11 leftovers, before the bumps (MEN10 step f uses it as the first bump smoke).

## Change Log

- 2026-10-17T02:58:13.549094+00:00 (created-by): Created by agent
