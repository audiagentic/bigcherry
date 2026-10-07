---
id: BCOP50
order: 50
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-08T03:13:00+11:00'
created-by: agent
priority: P2
---

# PRBE09 Vulkan AllReduce ownership disposition

## Audit result

PRBE09's premise is obsolete: BigCherry already implemented and hardware-screened the Vulkan meta communication-provider boundary under PRVP03/1290. Keeping PRBE09 pending would duplicate ownership and could spawn a second provider/transport path.

## Authoritative owner

PRVP03 owns Vulkan AllReduce/provider design and qualification. Patch 1290 is the evaluated reference SPI/fallback implementation. PRBE09/RD104 is terminally absorbed.

## Subsequent work already acting on it

Commit `2740ff76a1dd9343e9d742b43dc93015bcbdaf71` created 1290. Commit `fecfbad996a1309864ab9e61ff4e86d535a1f3a1` hardware-screened it on dual XTX RADV: stock fallback 496 pp / 24.6 tg and MTP5 50.4 versus host-F32 provider 259 pp / 20.1 tg and MTP5 40.7; 59,136 provider hits with matching greedy output.

## Unresolved action

Only PRVP03 phase 1 remains: first prove mapped-host import/coherency/synchronization for exact-F32 32-256 KiB on dual XTX, then compare against stock meta fallback. Do not add chunking/double buffering until profiling proves serialization worth attacking.

## Terminal disposition

- PRBE09: **complete/absorbed**; never reopened solely for CUDA/HIP collective changes.
- PRVP03 phase 1: promote only with correct work/output and CI95-low >=3% E2E production tensor-split gain with <=1% unaffected-control regression.
- Reject/defer phase 1 on unsafe Vulkan memory semantics, failed fallback, correctness/work mismatch, or no material E2E win.
- A future upstream Vulkan provider is evaluated/replaces 1290 under PRVP03 rather than creating another owner.

## Dependencies / blockers

Current BigCherry topology lacks normal GPU P2P; Vulkan phase 1 is therefore host-visible PCIe transport unless hardware/driver evidence proves otherwise. No hardware run, build, prototype or test was performed by this audit. Active pin/rebase/offline-validation and current MTP/MoE work were not modified.
