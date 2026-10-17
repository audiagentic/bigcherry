---
id: BCOP288
order: 288
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-17T19:13:00Z'
created-by: agent
priority: P2
work: S
---

# QFP04: gate F16 QSA decode GET_ROWS-to-CAST removal

**Discovery:** Validated 1295 writes F32 GET_ROWS then F16 CAST for both F16 K and V. Pinned ggml.c fixes GET_ROWS output to F32, although HIP getrows.cu supports F16 destinations. A direct F16 result could avoid two casts and 8 logical bytes per selected scalar, but this is not measured wall time and CPU/Meta graph semantics must be proved.

**Authoritative owner:** QFP04/1295; QFP03/1294 owns deterministic indices, QFP25/1334 sparse prefill and QFP17/1335 indexer. No new scheduler/cache/dispatch table or Q8_0 admission. Existing 1295 remains validated/default-on.

**Independent-work exclusion:** selected feat/rad06-helper-replay from independent 0295cde962a8 (2026-10-17 17:53 UTC); checked 12h branch commits and PRs, with no independent QFP04/1295 implementation, plan, benchmark, PR or queued lane found. RAD06 helper/replay/GEMM, MEN12/16-19 and Radiance release/qualification are protected. Recent BCOP287 was attention tiling, not selected KV gather.

**External mechanism:** merged llama.cpp #29639 Vulkan direct quantized sparse FA (16x AMD threshold); vLLM AMD QSA direct-index Triton BF16. Neither establishes an incremental BigCherry HIP gain.

**Unresolved action:** measure exposed F32 GET_ROWS+CAST time using existing profiling. If >=2.9126% baseline latency is optimistically removable, qualify one F16-only typed GET_ROWS reuse with CPU/Meta fallback and no new owner; else close.

**Terminal:** NO_ROUTE, BELOW_CEILING, UNSUPPORTED_TYPED_GRAPH, CORRECTNESS_FAIL, NO_GAIN, QUALIFIED_F16_ONLY. No experiment queued.

**Validation:** 19/19 pinned-source assertions; 4/4 host tests (65,536 F16 encodings, 64 index cases). No HIP compile, repository pytest, patch application or new GPU benchmark.
