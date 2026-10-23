---
id: BCOP186
order: 186
plan: patching-bc-optimizations
state: done
created-at: '2026-10-13T13:06:00Z'
created-by: agent
priority: P2
work: S
---

# PRBE52: close stale adaptive-MTP plan after measured rejection

**Finding/change:** PRBE52 remained pending with implementation/queue instructions after BigCherry PR #51 (2026-10-09) rejected both 1255 and 1268 on dev b11474 and removed their experiment sets. The pinned b11552 baseline already contains merged llama.cpp PR #29924's n-gram candidate-vector fix. The active plan is now a terminal negative decision with a bounded reopen gate; historical instructions remain in Git history.

**Authoritative owners:** PRBE52 owns the adaptive-depth disposition; 1255/1268 retain rejected implementation/evidence; 1321/1322 own MTP look-ahead; PRBE20/1210 owns verify-batch numerical identity; FMTP remains independent. No new controller, graph scheduler, experiment queue, allocation or telemetry framework.

**Already acted on:** First-party dev ABBA comparisons at 8K/24K/98K Flash-Next and 10K/32K dual-XTX 27B rejected adaptive settings against fixed depth; upstream PR #27210 remains open and is reference-only. Do not take over active Radiance helper/mailbox or patch-prefix work from the preceding 12 hours.

**Terminal action:** Mark PRBE52 done, do not requeue 1255/1268. Reopen only for a new eligible workload with measured >=3% E2E ceiling, offline held-out >=5% gain over best fixed k, graph/numerical correctness and >=4 independent-session paired AMD hardware proof; otherwise NO_REOPEN. Resolve a reproducible prior graph-destruction failure before any new performance promotion.

**Validation:** 15/15 source assertions; no repository pytest, build, patch apply, GPU run or new measured speedup. Upstream: https://github.com/ggml-org/llama.cpp/pull/27210 and https://github.com/ggml-org/llama.cpp/pull/29924.
