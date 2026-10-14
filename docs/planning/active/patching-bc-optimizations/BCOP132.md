---
id: BCOP132
order: 132
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-11T06:08:37+00:00'
created-by: agent
priority: P2
work: S
---

# PRBE01: gate BF16 TILE evidence on actual selector and independent logits

## Discovery / disposition

Pinned llama.cpp b11474 chooses BF16 VEC, TILE or MMA_F16 based on shape/GQA/mask/device. For masked, padded GQA=4/8 head 64/128/256, pp512 on gfx1100/gfx1201 selects MMA_F16; the 1202 native BF16 TILE experiment is not activated by merely setting BF16 KV flags. The 1202 producer currently maps one PPL comparison to both `backend_reference` and `ppl_equality`, and its activation probe is deliberately skipped. Nine tracked records remain ineligible; b11126 PPL parse failures are not numerical proof. No current-pin gain is established. This is source/record evidence plus a 108-case illustrative host selector fixture, not GPU execution.

## Ownership / novelty / protected work

PRBE01/1202 is the authoritative technical owner. Native BF16 VEC is upstream llama.cpp #20525; MMA_F16 and selector are upstream b11474. PRBE54/1271 owns Q5 KV staging. No new kernel family, selector, allocator, or telemetry framework. The last independent PRBE01 plan update was 2026-10-08 06:47 UTC; patch/evidence path last moved mechanically 2026-10-09 04:20 UTC. No independent PRBE01/1202 implementation, PR, benchmark or queued work was found in the inspected preceding 12 hours. Protected QFP41/1356, QFP49/1360/1361, QFP48 and recent mainline prefill work are read-only dependencies. Recent BCOP124-131 address different slices.

## Bounded action / terminal gate

First extend existing 1202 mechanics tests with actual selector shape admission, then add a BF16 TILE executed-path trace-gated marker and independently bound logits/KLD comparison (retain PPL separately). Verify no F16 staging for the native TILE branch, graph/repeated-request and multi-ubatch correctness. Reuse existing rocprof/patch validation. Reject if the target model never selects TILE or removable E2E share is below 2.9126%; otherwise qualify paired gfx1100/gfx1201 BF16 KV decode and proven TILE prefill with >=4 sessions/arch, >=10 ABBA rounds, CI95-low >=3% E2E and <=1% controls. Keep patch untested until these receipts pass; no new hardware queue while active lanes run.

## Evidence / validation

Inspected `ggml/src/ggml-cuda/{fattn.cu,common.cuh}` at b11474, `1202/{patch.py,validation/producer.py,validation/producer.toml,evidence/validation.json}`, PRBE01, and upstream [#20525](https://github.com/ggml-org/llama.cpp/pull/20525)/[#26289](https://github.com/ggml-org/llama.cpp/pull/26289). Source-static checks 13/13 passed; illustrative selector host matrix 108 cases, 5/5 route assertions passed. No repository pytest, patch apply, HIP build, GPU execution or new benchmark. Historical external gfx1100 BF16-KV decode +3.39% is not current-pin evidence. Technical design is only in PRBE01.
