---
id: RNX07
order: 7
plan: patching-r9700-cross-rdna
state: superseded
created-at: '2026-10-03T01:33:19.833407+00:00'
breadth: ''
skill: advanced
created-by: codex
work: L
priority: P1
---

# R9X07 — WHT/compressed-wire extraction into the existing AllReduce provider stack

## Description

Evaluate WHT plus low-bit compression as a wire representation layered over existing AllReduce providers, especially CPU-root/SHM on no-P2P systems. Do not port r9700 P2P/IPC transport directly.

## Steps

- Reproduce WHT4/WHT6 pack/unpack error and cost at 40 KB, 120 KB, 640 KB, 3.3 MB, 6.5 MB, and 10 MB.
- Compare RCCL, cpu-root/adaptive, 1250/1272 wire modes, and WHT modes under 2/3/4 ranks.
- Keep measured small decode winners unchanged; test pack-before-D2H versus D2H-before-pack and require in-model evidence.
- Only implement 1308 if encode+transport+reduce+decode beats current providers; keep lossy mode opt-in until quality gates pass.
- Select by measured rank/message/topology region, not a copied constant.

## Detailed Solution & Technical Design

Analysis-only until the real communication path is proven. Existing 1250/1272 own compressed-wire behavior; a host-roundtrip WHT result would add another codec/provider rather than reduce bytes on the actual collective path. Audit device-derived sequence, rank participation, graph replay, partial-buffer exact-zero semantics, teardown, and wraparound in the existing provider. For RNX07, require an in-model prefill with the 96-AR pattern and distinguish that from standalone latency diagnostics.

## Code Samples & Guidance



## Files

- kernels/r9k_ar_wht.hip
- kernels/r9k_ar4.hip
- r9700_vllm/comm/r4d_ar.py
- tests/test_ar_wht.py
- ggml/src/ggml-cuda/ggml-cuda.cu
- common/arg.cpp
- tools/llama-bench/llama-bench.cpp

## Validation

Tensor-level max/mean/RMS error, logits/KL/PPL/greedy tokens, MTP acceptance, long-prompt prefill, uncomputed-meta-rank zero semantics, deterministic rank order where promised, graph replay, and dynamic message sizes. Performance must include transport bytes, pack/unpack time, host/GPU utilization and E2E.

Acceptance: 1308 must own a real size/topology region where it beats all existing choices. If quality or transform cost removes the wire win, reject it cleanly.

## Effort & Risk



## Standards

Reuse existing provider ownership 1001/1244/1250/1252/1272/1275/1276/1277/1290/1291; mixed-generation ranks must agree on wire format.

## Acceptance Criteria

- 1308 must own a real size/topology region where it beats every existing choice.
- If quality loss or transform cost removes the wire win, reject cleanly.
- R9X11/RNX11 must be resolved before promoting any custom sequence-counter protocol.

## Notes

Original source alias is R9X07. Proposed slot 1308. Existing Flash-Next evidence says host round-trip lost to RCCL in the measured f32 sweep; this remains analysis-gated.

Verbatim legacy source retained during R9X→RNX migration:

# R9X07 — WHT/compressed-wire extraction into the existing AllReduce provider stack

Status: planned
Proposed patch: `1308_r9x_allreduce_wht`
Depends on: R9X01, current collective benchmarks, and R9X11 before promotion of any custom sequence-counter protocol
External source: `kernels/r9k_ar_wht.hip`, `kernels/r9k_ar4.hip`, `r9700_vllm/comm/r4d_ar.py`, `tests/test_ar_wht.py`
Existing BigCherry owners: 1001, 1244, 1250, 1252, 1272, 1275, 1276, 1277, 1290, 1291



GPT design pass (req_83e7cdc000be4b9e): do not create a new WHT or third codec/provider. The existing 1291 package already contains the large CPU-root pipeline boundary; follow-up should harden or qualify that path rather than create a parallel implementation. No 1308 package now.

2026-10-04 superseded by PGC11 (single owner item for AllReduce wire formats). Carried over: WHT + low-bit compression as a wire representation layered over existing providers (CPU-root/SHM on no-P2P); do not port r9700 P2P/IPC transport.

## Goal

Extract WHT+low-bit compression ideas into BigCherry's existing HIP AllReduce provider framework. Do **not** port r9700-stack's P2P/IPC transport directly onto Brutus-style systems where useful GPU P2P is absent. The first candidate is a compressed **wire representation** layered over an existing transport, especially CPU-root/SHM for large prefill payloads.

## Exact source-to-target mapping

Source: `kernels/r9k_ar_wht.hip` implements 64-point Walsh-Hadamard rotation plus grouped low-bit quantization; `kernels/r9k_ar4.hip` shows hierarchical multi-rank use; `tests/test_ar_wht.py` supplies transform/packing reference coverage.

BigCherry: `patches/1291_ar_cpu_root/patch.py` edits `ggml/src/ggml-cuda/ggml-cuda.cu`, `common/arg.cpp`, and `tools/llama-bench/llama-bench.cpp`. Existing wire experiments are `1250_nro01_allreduce_q8_wire` and `1272_ar_host_compressed_wire`; selector/trace work is 1275/1276/1277. Reuse these ownership boundaries rather than creating a parallel provider.

Candidate data path for no-P2P topology: `f32 activation -> GPU or CPU WHT+4/6-bit pack -> existing CPU-root/SHM transfer -> deterministic rank-order reduction/dequant -> result`. Also benchmark pack-before-D2H versus D2H-before-pack; do not assume GPU packing wins when host round-trip is dominant.

## Implementation

1. Reproduce pack/unpack error and cost for 4-bit (4.25 bits/element class) and 6-bit (6.25 bits/element class) at 40 KB, 120 KB, 640 KB, 3.3 MB, 6.5 MB and 10 MB.
2. Compare RCCL, current cpu-root/adaptive, 1250/1272 wire modes, WHT6 and WHT4 under 2/3/4 ranks.
3. Keep small 10 KB decode reductions on their measured exact winner; WHT is initially a prefill/large-message experiment.
4. Only create 1308 if encode+transport+reduce+decode beats the current provider. Lossy mode remains opt-in until model quality passes.
5. Adaptive selection is by measured rank/message/topology region, never an r9700 constant copied across machines.

## RDNA adaptation

The transform is architecture-neutral; pack/unpack geometry and atomics are not. Tune separately on gfx12/gfx11/gfx103x. P2P capability is a topology property, not an RDNA-generation property. On mixed-generation tensor split, all ranks must agree on the same wire format/protocol even if local pack kernels differ.

## Change Log

- 2026-10-03T01:33:19.833407+00:00 (created-by): Created by codex
- 2026-10-03T01:39:02.345892+00:00 (updated-by): Updated: section:notes
- 2026-10-03T02:18:29.079208+00:00 (updated-by): Updated: section:detailed_solution, section:notes

## Reviews

- RV4209
- 2026-10-03T15:22:10.593899+00:00 (updated-by): Updated: section:notes
- 2026-10-03T15:22:26.218372+00:00 (state-transition): State: pending → superseded
