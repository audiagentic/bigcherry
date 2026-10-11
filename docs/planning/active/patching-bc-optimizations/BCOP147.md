---
id: BCOP147
order: 147
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-11T21:04:42Z'
created-by: agent
priority: P2
work: S
---

# PRBE12: fail-close fused ADD destination geometry before qualification

## Discovery / change

At pinned b11552, 1206's `MUL_MAT[_ID] -> RESHAPE/VIEW -> ADD[_ID]` matcher validates subgraph uses, ADD operands, aliasing and (for VIEW) equal bytes, but does not prove matmul-versus-ADD destination shape/stride compatibility. The fused MMVF/MMVQ kernels receive the ADD as `dst`, derive indexing from its `ne[]/nb[]` and assert `ids || dst->ne[1]==1`. An equal-byte reshape can pass static admission while violating that assertion. Source/host discriminator only: 12/12 static assertions; 8 deterministic shape cases, 2 with a no-IDs assertion mismatch and 4 with a changed row extent. **No observed GPU failure or new speedup.**

## Authoritative owner / independent work

PRBE12/1206 owns safety, shape census and its existing contract producer. PRBE39 VIEW/alias protection is already folded in; PVPS15 owns activation evidence. BCOP80 reconciled producer/status on 2026-10-09; this new shape/indexing finding is not a repeat of that audit. No independent 1206/PRBE12 implementation, PR, hardware lane or queue within the preceding 12 hours. Protected PA49/release, QFP41/1356, QFP48, QFP49/1360/1361 and 1281 refactor were not edited.

## Bounded next action / terminal disposition

Before hardware, add table-driven same-byte shape-change negative fixtures to existing 1206 tests and collect current-model `mm/mid/add` shape/stride/IDs/route census. Fail closed to unfused when kernel destination geometry is unproven. If safe positive-model activation disappears, close or retain default-off; do not queue benchmarks. If safe activation survives, run the existing full-vocab/replay/ABBA contract (4 sessions/architecture, >=10 paired rounds, CI95-low >0%, <=1% control regression). No second matcher, producer, scheduler, allocator or dispatch table. Details, exact functions and references: PRBE12.

## Evidence

Pinned llama.cpp b11552 `ggml-cuda.cu::ggml_cuda_try_fuse`, `ggml.c::ggml_can_fuse_subgraph_ext`, `mmvf.cu::ggml_cuda_mul_mat_vec_f`, `mmvq.cu::ggml_cuda_mul_mat_vec_q`; BigCherry `1206.../patch.py::_NEW`. Upstream #29633 changes MMVF eligibility, not this invariant. vLLM #52776 and SGLang TensorMatcher are contract references, not AMD performance evidence. No repository pytest, C++/HIP build or GPU benchmark was run.
