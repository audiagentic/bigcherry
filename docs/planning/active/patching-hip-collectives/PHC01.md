---
id: PHC01
order: 0
plan: patching-hip-collectives
state: pending
created-at: '2026-09-09T10:51:11.474188+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Qwen3.8-Flash 3-GPU/4-GPU tensor-split benchmark matrix

## Description

Resolve the blocked Qwen3.8-Flash-Next 3-GPU/4-GPU tensor-split benchmark only after no-GPU preflight proves model/source fit and HI138 releases hardware.

## Steps

- Identify exact GGUF, quant, total/PLE/MTP bytes and source revision with qwen4exp support; verify CPU/mmap PLE placement and derive per-GPU static/KV/workspace/ubatch budgets.
- Reframe the 6900XT effect as x4 endpoint penalty under host-staged traffic; measure negotiated link and do not assume PCIe P2P.
- Start with the minimal matrix: 3G 40/30/30 128K F16 ub1024 MTP off; 4G tiny ~38/29/29/4; 4G medium ~34/27/27/12; expand only if competitive.
- Measure at resident depths (~32K, ~128K, then 192K/262K where fit), pp512/pp4096/tg128/tg512, VRAM/RAM and correctness; use 3 screening reps then strengthen winner/control.
- Run MTP only on winning 3G/4G configurations and retain blocked-preflight or negative disposition if fit/source gates do not clear.

## Detailed Solution & Technical Design

The original full 20-run matrix is a hypothesis, not an execution plan. Qwen3.8-Flash-Next requires quantized GGUF and a revision with qwen4exp support; PLE is CPU/mmap placement. Keep HI138's fail-closed topology constraints and do not spend hardware time before source/VRAM preflight.

## Code Samples & Guidance



## Files

Exact Qwen3.8-Flash-Next GGUF/source revision record; VRAM-fit worksheet; 3G/4G benchmark recipes; resident-depth performance/correctness artifacts; MTP winner evidence; HI138 dependency record.

## Validation

No-GPU model/source/PLE preflight; per-split VRAM headroom; 3G/4G controls; x4 endpoint/link measurement; resident-depth pp/tg; correctness; VRAM/RAM; MTP acceptance; negative/blocker evidence.

## Effort & Risk



## Standards

Capacity before benchmark; no fabricated source support; topology claims measured; preserve HI138 fail-closed guards.

## Acceptance Criteria

Hardware execution begins only after model/source/fit gates pass. A reduced matrix yields an evidence-backed 3G vs 4G decision; MTP is tested only on winners; otherwise the item remains explicitly blocked-preflight with the missing gate named.

## Notes

Supersedes: HI140
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-hip-collectives-hi140

## Change Log

- 2026-09-09T10:51:11.474188+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:07:33.182278+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.003307+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.606099+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:39:24.373660+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_023948_hip-collective-and-kernel-cove_9788
- 2026-09-10T02:39:48.390428+00:00 (updated-by): Updated: section:ledger-events
