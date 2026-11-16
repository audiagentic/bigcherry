# Run Radiance Host Link

## Ownership and next tests (reconciled 2026-11-13)

1. **[HL01](../../completed/run-radiance-host-link/HL01.md) — completed diagnosis:** verbose PCIe/IOMMU checks and physical XTX slot swap proved the halving follows the first PEG x8 root port, not a card/VBIOS. CPU-internal arbitration is supported; the particular coherent-snoop stage remains a hypothesis. No PCIe configuration fix is identified.
2. **[HL02](HL02.md) — single laboratory experiment owner:** contended coherent/non-coherent mailbox test first, then true concurrent CPU push to one versus all three GPUs (320 KiB–10 MiB while GPUs publish partials), and R9700/full-size DMA-BUF live qualification. GPU-pull recovery, victim CPU BAR bypass and primary GPU write-only trigger are distinct controls.
3. **[VTP38](../run-dense-27b-virtual-tp/VTP38.md) — production TP3 transport:** generic per-rank receive methods and optional ordered CPU reduction. Its reference is the improved VTP36 rad0032/rad0033 GPU candidate, not the earlier ~4 ms rad0031 baseline. No host-reduced TP3 inference gain has been measured.
4. **[RAD40](../../completed/patching-radiance-rdna3/RAD40.md) — superseded:** retained historical transfer-physics/activation/chunk data; every remaining transfer-method/all-pair experiment moves to HL02. **[RAD50](../patching-radiance-rdna3/RAD50.md) — pending:** reusable DMA-BUF CPU-write, pre-armed GPU visibility, generation and crash/lifetime safety proof. Later R9700 DMA-BUF successes and the gfx11 `wall_clock64()` watchdog correction are preserved.
5. **[HL03](HL03.md) / [HL05](HL05.md) / [RAD42](../patching-radiance-rdna3/RAD42.md) / [RAD51](../patching-radiance-rdna3/RAD51.md)** retain separate MoE activation, TP2 relay and MoE result production scope. They reuse HL02/RAD50 lab gates rather than duplicating them.
6. **[HL04](../../completed/run-radiance-host-link/HL04.md) — completed:** ~0.1–0.2 ms exposed CPU combine, no MoE result-push integration. This is *not* a TP3 host-reduce NO-GO.

Historical results and exact source references: [2026-11-13 slot swap, RAD50 and VTP36 candidate receipts](../../../evidence/2026-11-13-hl02-vtp38-consolidation/README.md). See also [Flash-Next execution status](../../FLASHNEXT_EXECUTION_STATUS.md); preserve original raw lab logs and pending experiments under their separate owners.
