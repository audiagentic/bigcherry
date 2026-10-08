---
id: MEN01
order: 1
plan: run-multi-engine
state: pending
created-at: '2026-10-08T20:36:00.311928+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: S
---

# vLLM stack inventory: what is source, what is binary, where the speed comes from

## Description

Gate for the whole group. The reference container (image stilldeadcode/vllm-radiance:0.9.3, radiance layer, libr4d b9e42ab-rx9) is about 3x faster than BigCherry llama.cpp on prefill on one R9700 with a 4-bit Qwen3.8-27B (2026-10-09: ~3,000 vs 1,003-1,089 t/s to 35K, ~2,470 vs 764-773 at 111K). Before building any multi-engine tooling, establish which parts of that stack are editable source (vLLM Python, radiance, Triton attention) and which ship as binaries (libr4d kernels), and which of the 79 RADIANCE_* settings carry the gain.

## Steps

1. From the stopped container: list installed packages with versions and whether each has source in the image (vllm, radiance, libr4d, torch, triton); record image digest.
2. Record the launch command and environment verbatim as a settings snapshot.
3. Kernel-level view: rocprofv3 kernel trace of one 24K prefill inside the container; group by family the same way prefill-kernel-table.py does for llama.cpp, so the two censuses can be read side by side.
4. Write the finding: what a patch system over this stack could reach, and what it could not.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

A table of components with source/binary status and version; one kernel census at 24K; a one-paragraph conclusion that the owner can act on.

## Effort & Risk



## Standards



## Acceptance Criteria

The inventory names every component of the reference stack with source or binary status, and states whether carrying our own changes on top of it is possible.

## Notes

Read-only on the container: start, measure, stop. No configuration change. Production llama-swap docker stays off.

## Change Log

- 2026-10-08T20:36:00.311928+00:00 (created-by): Created by agent
