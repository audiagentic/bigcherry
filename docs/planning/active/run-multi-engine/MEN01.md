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

# Radiance inventory: build it on Brutus, what runs on which card, where the speed comes from

## Description

Gate for the group. Owner decision 2026-10-09: the second engine is standalone radiance (codeberg.org/StillDeadcode/radiance, Apache-2.0), not vLLM plus the radiance layer. Radiance is a C++/HIP inference server with a CMake build, an HTTP API that speaks OpenAI, Anthropic and llama-server's native protocol, and kernel libraries loaded as .so plugins through a C ABI: libr4d (gfx1200 / gfx1201 only), libref, libavx, libquant. The README says the engine starts on any AMD GPU family but another card needs a kernel library for it. Reference numbers to explain (one R9700, 4-bit Qwen3.8-27B, 2026-10-09): the vLLM + radiance container ~3,000 t/s prefill to 35K and ~2,470 at 111K; BigCherry llama.cpp on the same card 1,003-1,089 with MTP, 1,141-1,291 with no drafter, 764-857 at 111K.

## Steps

1. Clone at a recorded commit; read LICENSE, NOTICE, spec.md and the abi/ and arch/ directories; record what each kernel library covers and for which architecture.
2. Build on Brutus with the lab ROCm (/mnt/vault/tmp/bc-rocm) for gfx1201; record the exact commands and any change needed.
3. State per card what can run today: R9700 (libr4d), RX 7900 XTX and RX 6900 XT (libref only, or nothing).
4. Checkpoint formats it loads (MXFP4 from AMD Quark, FP8, bf16, int8) and which of our models exist in one of them; the container already has /models/Qwen3.8-27B-MXFP4-mtpfp8 and a DFlash2-FP8 drafter.
5. rocprofv3 kernel census of one 24K prefill, grouped like prefill-kernel-table.py, beside the llama.cpp census.
6. Write the finding: which kernel family carries the lead, and what is card-specific.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

A build that starts and answers /health on Brutus; a component table with architecture coverage; one kernel census at 24K beside ours.

## Effort & Risk



## Standards



## Acceptance Criteria

Radiance builds from source on Brutus at a recorded commit, and the inventory states what runs on each of the four cards and which kernel family explains the prefill lead.

## Notes

Read-only on the container: start, measure, stop. No configuration change. Production llama-swap docker stays off.

## Change Log

- 2026-10-08T20:36:00.311928+00:00 (created-by): Created by agent
- 2026-10-08T20:47:42.172651+00:00 (updated-by): Updated: section:title, section:description, section:steps, section:validation, section:acceptance_criteria

## Ledger-events

- chg_20261008_212949_added-a-like-for-like-single-c_9270
- 2026-10-08T21:30:02.720551+00:00 (updated-by): Updated: section:ledger-events
