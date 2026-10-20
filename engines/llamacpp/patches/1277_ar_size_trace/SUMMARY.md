# 1277_ar_size_trace

**Status:** untested
**Plan item:** PGC09

Lab-only diagnostic for 0840's adaptive AllReduce dispatch. BIGCHERRY_AR_SIZE_TRACE=<n> prints the first n calls' bytes, ne0/ne1, type, **preferred** internal/RCCL route and switch threshold. It is inserted immediately after computing prefer_internal, **before either provider runs**. The static trace_count is non-atomic; this output is not concurrency-safe, not a completed-provider witness, not a phase/graph/ubatch identifier and not a measured transition latency. A later internal preflight decline can select RCCL despite an internal preference log. Do not use it for production performance attribution or an activation pass.

**2026-10-15 BCOP238:** PGC12 phase-policy proposal is retired; PGC09 owns any conditional outcome/phase receipt, preferably by reusing 0830's actual provider-return seam with per-request graph/phase identity and lab-only 1277. Keep trace disabled during timed ABBA runs. The validated exact-F32 96 KiB dual-gfx1100 policy is the control. No new provider, scheduler, dispatch table or GPU qualification is authorised without a proven >=3% exposed E2E bottleneck.
