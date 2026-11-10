---
id: ENG02
order: 0
plan: tuning-radiance-engine
state: pending
created-at: '2026-11-10T13:11:37.675845+00:00'
breadth: ''
skill: basic
created-by: agent
priority: P1
work: S
---

# Current-build MTP depth and acceptance sweep on flashnext-4gpu

## Description

Depth sweeps exist for the dual-XTX 27B (no speculation 51.9 tok/s, MTP3 114.3 tok/s at acceptance 0.74, MTP7 108.2 at 0.45; runs/qwen27b-spec-band, qwen27b-spec-depth) and for llama.cpp Flash-Next (MTP3 best, MTP3+ngram-mod slower; QFN01), but none on the current Radiance flashnext-4gpu engine with the policy on. Acceptance varies -20%..+24% per prompt with expert placement (runs/policy-accept), so the sweep must use several prompts and judge ms/step and tokens/step, not one prompt's tok/s.

## Steps

1. Depths 0,1,2,3,4,5 (engine.num_speculative_tokens) on the production layout with the policy on and warmed; same prompt set (>=6 seeds one-stream, 3 six-stream), engine decode tok/s, ms/step, tokens/step, acceptance.
2. Per-arm short lease windows; a fresh server per depth (a different depth is a different configuration and plan key).
3. Record the winning depth per stream count and whether the default should move.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Per-depth tables over >=6 prompts with the policy state verified from each manifest; the MTP replay check (launcher) passes for every depth above 1.

## Effort & Risk



## Standards



## Acceptance Criteria

The default depth is confirmed or changed on measured one-stream and six-stream decode over several prompts, with the evidence recorded here.

## Notes

Memory: MTP acceptance over several prompts. Source: gateway prior-work audit 2026-11-10.

## Change Log

- 2026-11-10T13:11:37.675845+00:00 (created-by): Created by agent
