---
id: BCOP220
order: 220
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-15T00:07:39Z'
created-by: agent
priority: P1
work: S
---

# PGC15: freeze whole-boundary reduction codec before tiled prefill overlap

**Discovery:** Pinned llama.cpp b11552 selects RCCL F32/BF16 by **whole tensor** element count. A naive per-tile threshold changes the numerical contract: 3-rank `[768,512]` whole BF16 versus tile128 F32. Proposed post-submit false-to-Meta fallback can re-reduce mutated outputs. These are proposed-code hazards, not observed GPU failures.

**Owner:** PGC15 owns producer tiling, admission, per-rank event/stream lifetime and frozen codec. PGC12 owns actual provider-completion traces, PGC14 RCCL SHM, PGC16 native fold retirement, QFP17 acceptance. No duplicate scheduler/allocator/dispatcher. Active Radiance helper, RWQ, MEN10 and PNRO11 work untouched.

**Eligibility:** Branch `feat/rad03-gpu-lanes`, independent HEAD `6786b9d477e3d725a2ede2a7c5624786f06d8674` (2026-10-14 23:40 UTC). PGC15 last substantive independent update 2026-10-05; no independent PGC15 PR/patch/queue or related modification found in preceding 12 hours. Recent BCOP214-219 cover other capabilities.

**Action:** Preserve whole-boundary F32/BF16 policy in range-provider preflight; reject all unsupported ranges before submission; any later provider failure must be terminal, never whole-tensor fallback. First require a real attention-output producer boundary and codec/actual-provider trace; then an isolated identical-work two-stream RCCL overlap proof. Do not queue GPU work before source gate.

**Terminal:** `INELIGIBLE_BOUNDARY`, `CODEC_UNPRESERVED`, `NO_OVERLAP`, `POST_SUBMIT_ERROR`, `REJECT_TILE_OVERLAP`, or qualified default-off prototype. Promotion: >=3% E2E prefill or >=30% eligible attention-AR wall hidden, <=1% decode regression, full correctness and graph lifetime checks.

**Validation:** Host-only Python compile and 9/9 synthetic tests passed. No repository pytest, HIP build, GPU run or new benchmark. Technical design belongs in PGC15; this is disposition only.
