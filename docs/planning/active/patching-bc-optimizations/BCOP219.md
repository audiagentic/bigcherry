---
id: BCOP219
order: 219
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-15T00:00:00Z'
created-by: agent
priority: P1
work: S
---

# PGC14: block unsafe RCCL SHM CE coalescing; reuse existing per-slot ring

**Discovery:** PGC14's unimplemented `batchSteps[]` sketch can copy unready FIFO gaps when `sliceSteps > 1`, copy padding/omit a payload after a partial first slot, and omit destination per-slot size publication. Pinned RCCL `57e58688/src/transport/shm.cc` and current ROCm/rocm-systems already own the asynchronous `NCCL_STEPS` event ring. This is a proposed-design correctness issue, **not** a reproduced production defect. Host-only model: 9/9 cases passed, no HIP/rccl-tests or new GPU measurements.

**Owner/consolidation:** PGC14 is authoritative for RCCL SHM mode/proxy, coalescing admission and gfx1030 hostcall lane. `tools/lab/rccl/README.md` records lab controls. PGC15 owns producer/AR tiling; PGC09/11 own provider and wire; do not add another FIFO/allocator/scheduler. Existing 2026-10-06 protocol/algorithm sweep is terminal (no win; LL ~3.6x slower). The current upstream SHM proxy still uses per-slot events; ROCm/rocm-systems#12145 is external gfx1030 AllGather negative-test context only.

**Eligibility:** selected `feat/rad03-gpu-lanes` from newest independent MEN10 commit; PGC14 last independent update 2026-10-06 and no PGC14/rccl-lab change, associated PR or queued experiment identified within 12h. Excluded active Radiance RAD02/03/05/06, MEN10, RWQ and nearby GPU/helper work; prior BCOP215–218 audit slices not repeated.

**Unresolved action:** First inspect existing SHM/CE/direct channel logs and completed lab receipts; instrument source-not-ready/window/event stalls only if missing. Terminal `NO_CE_PATH`, `NO_SMALL_COPY_LIMITER` or `NO_E2E_HEADROOM` (<3% theoretical E2E) rejects coalescing. Only a proven candidate may use PGC14's fail-closed contiguous-byte/metadata/event gate and 2/3-rank AllReduce+AllGather rccl-tests. Promote only at ≥5% representative prefill or ≥20% collective critical-wall improvement without correctness or decode regression; otherwise `REJECT_COALESCE`. No new production implementation or hardware queue created.

**Source references:** https://github.com/ROCm/rccl/blob/57e58688/src/transport/shm.cc ; https://github.com/ROCm/rocm-systems/blob/develop/projects/rccl/src/transport/shm.cc ; https://github.com/ROCm/rocm-systems/issues/12145 .
