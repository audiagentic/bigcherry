---
id: BCOP59
order: 59
plan: patching-bc-optimizations
state: superseded
created-at: '2026-10-08T15:05:57+11:00'
created-by: agent
priority: P2
---

# PRBE59 / RD76: superseded rocWMMA FlashAttention restoration

## Discovery / disposition

Upstream llama.cpp PR #26046 removed legacy rocWMMA FA in July 2026; pinned b11474 already contains native AMD MMA FA, tuned by PR #28102. Fork 5aa2f049 only supplies build plumbing. **Close without patch, selector, dependency or queue.** PRBE59 is authoritative.

## Owners / already-acted work

Upstream `fattn.cu` / `fattn-mma-f16.cuh` own FA. RD06/1203 is a separate rejected native config experiment (BigCherry kernel -0.0154%, CI95-low -0.0745%, not rocWMMA evidence); PRBE110 owns Q6_K, PRBE65 Vulkan FA. MTP/patch tooling is protected. Last independent PRBE59 update 2026-09-24, outside 12 hours.

## Acceptance / terminal gate

No RD76 action. Reopen only for maintained distinct code after >=5% first-party FA wall-share, full graph/long-context correctness, >=4 sessions and CI95-low >=3% E2E gain with <=1% controls. See PRBE59. Eleven b11474 static assertions passed; no hardware/build. Sources: https://github.com/ggml-org/llama.cpp/pull/26046 and https://github.com/ggml-org/llama.cpp/pull/28102 .

## Change Log

- 2026-10-10T10:25:36.280376+00:00 (state-transition): State: completed → superseded
