---
id: BCOP32
order: 32
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-05T21:46:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: S
---

# Track heterogeneous multi-GPU placement qualification

## Description

Action/disposition ledger for asymmetric multi-GPU placement. RPL01 is the cross-capability cost/recommendation owner; PHA03 owns topology/RCCL admission; llama.cpp's backend scheduler remains execution/allocation/copy owner. BCOP32 must not become another scheduler design.

## Actions

1. Feed measured per-device service rates and PHA03 topology/transport evidence into RPL01.
2. Require RPL01 to score placements already expressible through current layer/tensor split and scheduler controls before proposing new machinery.
3. Validate predictions against single-device, 2xXTX and 2xXTX+R9700 evidence; treat the chipset-routed 6900XT as a separate optional lane.
4. If an observed >=5% winning placement cannot be expressed by current controls, open the smallest technical change under the existing scheduler/provider owner. Otherwise keep the result advisory/configuration-only.
5. Record terminal disposition: `existing-controls`, `scheduler-seam-needed`, `not-predictive`, or `rejected-topology`.

## Gate

No runtime actuation or second placement engine is authorized here. No-P2P, PCIe asymmetry, VRAM limits and RCCL admission must be explicit inputs rather than inferred from device count.

## Related

RPL01; BCOP30; PHA03; MET01/MET05 where expert/auxiliary placement is involved.
