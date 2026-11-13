# Run Radiance Host Link

## Next actions and ownership (reconciled 2026-11-13)

1. **HL01:** authoritative root-port/topology, slot-swap and contended-engine evidence. The PCIe configuration and IOMMU checks did not find a cause; the exact shared CPU-I/O mechanism remains inferred, not proven.
2. **HL02:** *single owner* of matched cross-GPU contention and GPU-pull/true-CPU-BAR-push A/B, including read-vs-write trigger, duty cycle, memory/coherence controls, and TP3 shapes. Reuse RAD40's baselines and RAD50's DMA-BUF/GPU-live test machinery.
3. **VTP38:** sole owner of the production CPU-reduced TP3 all-reduce, contingent on HL02 and real CPU-to-VRAM delivery on all ranks. VTP36 owns overall TP3 programme results, not a second implementation.
4. **HL03 / HL05:** distinct production consumers: MoE helper-activation relay and TP2 two-XTX collective relay. RAD42/RAD51 own MoE result-plane integration. None should duplicate HL02's generic host-link tests.
5. **HL04:** closed; its different-layout CPU-combine no-go is not a TP3 no-go.

RAD40 stays open for its historical link-physics harness and distinct outstanding all-pair coverage. RAD50 stays open for generic mapping, cross-process lifetime, GPU-live visibility, fault and MoE FP16 correctness gates. Do not retire either merely because HL02 reuses their evidence.

See [Flash-Next execution status](../../FLASHNEXT_EXECUTION_STATUS.md) for plan-item states and evidence paths.
