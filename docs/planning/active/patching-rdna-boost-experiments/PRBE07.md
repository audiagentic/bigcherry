---
id: PRBE07
order: 0
plan: patching-rdna-boost-experiments
state: completed
created-at: '2026-09-09T10:54:00.396867+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: M
priority: null
---

# BridgeSpec evaluation — inspiration-only; no sidecar adoption

## Description

Terminal research disposition for RD101. BridgeSpec was audited at commit `2b846f2ff1eb95ac84e4b0488882b7e4066bff14` (BridgeSpec 0.1.0 research preview, 2026-08-26), MIT licensed. Recommendation: **inspiration-only**. Do not import its sidecar runtime, model-specific drafter, or gfx1100 MMVQ patch into BigCherry.

BridgeSpec validates a useful mechanism: keep target verification authoritative while moving serial draft work into a HIP-specialized runtime, and tune target verification for widths 2-8. Its production integration is host-mediated; its Vulkan/HIP external-memory experiments proved memory sharing but did not establish reliable external-semaphore synchronization. The published runtime is single-GPU Windows/gfx1100, process-singleton/non-thread-safe, Qwen3.8-27B-specific, and requires one slot, no context shift, no checkpoint/cache restore.

BigCherry has since developed native MTP ownership and a separate-draft-GPU overlap pipeline (FMTP). Importing BridgeSpec would create a second speculative state/lifecycle/ABI path while weakening multi-request, long-context and multi-GPU guarantees. Its gfx1100 MMVQ tuning also overlaps current BigCherry/upstream small-N MMVQ qualification and must not bypass those owners.

## Evidence and comparison

Pinned source:
- BridgeSpec commit: `2b846f2ff1eb95ac84e4b0488882b7e4066bff14`.
- Integration base: upstream llama.cpp `f5a7ec15da6add890a5624c0990714498df837a4`.
- License: MIT for BridgeSpec/original integration; external model/vocabulary artifacts retain their own licenses.
- Architecture: HIP MTP sidecar keeps F16 KV, performs catch-up plus three serial draft steps, uses a 40,960-row sliced Q4_0 head and maps IDs to the target vocabulary. DFlash uses five selected target features, graph-captured HIP drafting and width-8 verification.
- Boundary: host-facing DLL calls. Zero-copy Vulkan/HIP is explicitly experimental, not the production path.
- Correctness/lifecycle limits: singleton state; no reset/save/restore/fork/shift/free lifecycle; multi-slot and context shift unsupported.

Published external measurements are directional only, not BigCherry evidence. On one RX 7900 XTX / Windows / ROCm 7.2, BridgeSpec reports controlled decode-only MTP A/B of 104.5→106.9 tok/s (code) and 117.7→120.7 tok/s (agentic edit). DFlash reports larger workload-dependent high-water results but also a much lower prose result. Raw per-request logs/public harness are not supplied, so these numbers do not establish portability to Linux, gfx1201, tensor split, the auxiliary gfx1030 GPU, or BigCherry's no-P2P topology.

## Ownership / consolidation

- Native speculative semantics, acceptance, rollback and target-authoritative commit remain with current llama.cpp/BigCherry MTP owners.
- Cross-device ahead/replay/overlap remains with `patching-flash-next-mtp-pipeline` (FMTP); do not add a sidecar scheduler or second draft-state machine here.
- Small-N MMVQ geometry remains with existing BigCherry/upstream MMVQ qualification owners; BridgeSpec's width-2..8 gfx1100 patch is an external comparison oracle only.
- THA05 is unrelated tune-campaign orchestration and is not a dependency; the original PRBE07 instruction to compare BridgeSpec against THA05 was stale.

## Transferable mechanisms

1. **Draft-head vocabulary slicing** is worth retaining as an offline design reference only. Any BigCherry use must be owned by the existing native draft-vocabulary work and prove exact ID remap, fallback and acceptance accounting.
2. **Catch-up that performs only state required for future drafting** is a useful reference for native MTP profiling, but only if profiling proves equivalent discarded work remains in the current path.
3. **Verification-width-specific MMVQ tuning** is a useful offline oracle for gfx1100. It does not justify importing hard-coded geometry without current-pin gfx1100/gfx1201 qualification.
4. **Cross-backend zero-copy** is rejected as a current dependency: BridgeSpec itself did not ship external-semaphore synchronization, and BigCherry lacks normal GPU P2P.

## Validation / next gate

No implementation or hardware run is required to close PRBE07. Reopen only through an existing authoritative owner when new evidence satisfies one of these bounded gates:
- native draft-head/logit work profiles at >=5% of speculative wall time and vocabulary slicing can remove a material fraction without changing full-vocabulary IDs;
- current production verification widths 2-8 spend >=5% wall time in an MMVQ signature not already covered by an active MMVQ item;
- a supported Vulkan/HIP interop path demonstrates explicit synchronization plus multi-request/context-shift lifecycle correctness on BigCherry hardware.

Any adopted mechanism must be tested on current pinned llama.cpp, Linux gfx1100 and gfx1201, with tensor-split/separate-draft topology as applicable; require target-authoritative greedy identity, multi-request same-process, multi-ubatch/long-context correctness, expected work/transfer accounting, and CI95-low-positive >=3% E2E improvement with <=1% unaffected-control regression.

## Notes

Supersedes RD101. Migration: capability-rebaseline-v3-2026-09. Terminal disposition recorded by BCOP51. No BridgeSpec benchmark is BigCherry-measured evidence.

## Change Log

- 2026-09-09T10:54:00.396867+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-10-08T04:07:42+11:00 (audit): pinned BridgeSpec source/license, reconciled current native MTP/FMTP ownership, and closed as inspiration-only.
