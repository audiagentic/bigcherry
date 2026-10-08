---
id: PRBE09
order: 0
plan: patching-rdna-boost-experiments
state: completed
created-at: '2026-09-09T10:54:07.484798+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P2
---

# UP-VK-005: Vulkan tensor-parallel AllReduce tracking — absorbed by PRVP03

## Description

Terminal disposition (2026-10-08 audit): this upstream-tracking item is superseded by the concrete Vulkan communication-provider work already implemented and hardware-screened under PRVP03 / patch 1290. Do not open a second Vulkan AllReduce implementation item from PRBE09.

The original premise ("no Vulkan tensor-parallel AllReduce implementation exists or is pinned yet") is no longer true for BigCherry. Patch 1290 exposes the meta-backend communication SPI from ggml-vulkan through `get_proc_address` and provides a default-off contiguous-F32 host reference provider. Its first-party dual-XTX RADV screen proved the SPI and fallback boundary while also proving the synchronous host implementation is not a performance candidate.

Upstream llama.cpp still has no Vulkan-local AllReduce implementation on current master. The backend-local optimized AllReduce in `ggml/src/ggml-cuda/allreduce.cu` is CUDA/HIP-specific; Vulkan tensor parallel continues through the generic meta-backend communication/fallback path. Therefore there is nothing new to port from upstream Vulkan today.

## Ownership / consolidation

- **PRVP03 is the authoritative Vulkan AllReduce/provider owner.**
- **1290 is the reference SPI/fallback implementation.**
- PRBE09 owns no code, selector, transport, staging allocator, synchronization primitive, or benchmark lane.
- Do not duplicate PRVP03 with a second Vulkan provider or a parallel "upstream adoption" patch.
- If upstream later lands a Vulkan provider, PRVP03 compares/replaces 1290 against that implementation; PRBE09 remains closed.

## Repository evidence

1. Commit `2740ff76a1dd9343e9d742b43dc93015bcbdaf71` created 1290 and PRVP03's provider lane.
2. Commit `fecfbad996a1309864ab9e61ff4e86d535a1f3a1` recorded the first hardware result: 27B Q8_0, dual 7900 XTX, RADV, same binary, three requests.
3. Provider OFF / stock meta fallback: tensor split 496 pp / 24.6 tg; MTP5 50.4.
4. Provider ON / synchronous host-F32 reference: 259 pp / 20.1 tg; MTP5 40.7.
5. Layer-split control remained 947-951 pp / 20.8 tg.
6. Provider activation marker fired 59,136 times and greedy output matched the layer-split reference in both arms.

These are first-party BigCherry measurements. They prove wiring/correctness for the tested lane and reject synchronous CPU reduction as a speed path; they do not prove the proposed mapped-host phase 1 will be faster.

## Current upstream / external mechanisms

Current llama.cpp master has a CUDA/HIP two-GPU AllReduce in `ggml/src/ggml-cuda/allreduce.cu`, but no corresponding Vulkan implementation. Upstream Vulkan tensor-parallel reports continue to exercise the meta backend, including multi-buffer/tensor-split failures. This is a correctness baseline and an integration warning, not a mechanism to port.

vLLM ROCm now layers multiple collective choices (custom AllReduce, AITER collectives, and quick-reduce-style paths). Those mechanisms reinforce the value of size/topology-specific collective selection, but they depend on ROCm/HIP process/device semantics and are not evidence that a Vulkan external-memory design will win on RDNA.

## Terminal disposition

**COMPLETE / ABSORBED.** PRBE09 is closed in favour of PRVP03. Re-open is prohibited merely because upstream gains another CUDA/HIP collective. Only a real upstream Vulkan communication provider changes the adoption question, and that comparison belongs in PRVP03.

## Validation

No new build or hardware run is required to close this duplicate tracking item. The existing 1290 hardware evidence is the measured basis for consolidation. PRVP03 owns all future correctness, fallback, multi-request/multi-ubatch, long-context and performance qualification.

## Acceptance Criteria

- PRBE09 no longer advertises a future implementation item.
- PRVP03/1290 are recorded as the sole current BigCherry Vulkan AllReduce owner/reference implementation.
- Future upstream Vulkan provider work is reconciled through PRVP03, not a resurrected PRBE09.

## Notes

Supersedes: RD104
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd104

## Change Log

- 2026-10-08 (triage): completed; Already completed on main; PRVP03/patch 1290 hardware evidence and terminal disposition documented above. File relocated from active to completed without changing its resolution.

- 2026-09-09T10:54:07.484798+00:00: Created by capability-rebaseline-v3.
- 2026-10-08: Audit closed PRBE09 as absorbed by PRVP03/1290 after tracing implementation and first-party hardware evidence.
