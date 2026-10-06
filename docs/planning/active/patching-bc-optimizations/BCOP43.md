---
id: BCOP43
order: 43
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-07T00:09:42+11:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: S
---

# Bound RDNA MMQ Stream-K follow-up

## Disposition

Authoritative owner: PKC05; HIP architecture/shape dispatch remains owned by existing HIP-autotune machinery.

This audit found upstream draft #30022 adding a GCN-specific Stream-K/tiled crossover and wave64 fixup sizing while retuning `mmq-config-gcn.cuh`. It contains no RDNA measurements and explicitly leaves other AMD architectures untested; BigCherry must not copy its GCN config table or thresholds to gfx1100/gfx1201.

No subsequent BigCherry work has implemented this mechanism. The unresolved action is only the cheap PKC05 discriminator: record tile efficiency, selected Stream-K/fixup and fixup share on representative quantized RDNA shapes. Run a hardware A/B only if regular tiling is already >=90% efficient while Stream-K is selected, or fixup is >=3% of MMQ wall time. Promote through existing HIP-autotune ownership only at >=3% repeated kernel-time and >=1% end-to-end gain with no >1% primary-control regression; otherwise close this sub-slice without new dispatch/scheduler state.
