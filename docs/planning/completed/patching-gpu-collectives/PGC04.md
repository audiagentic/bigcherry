---
id: PGC04
order: 0
plan: patching-gpu-collectives
state: superseded
created-at: '2026-09-29T09:33:59.429712+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# 0860 allreduce provider CLI (--allreduce / --allreduce-wire)

## Description

Base-framework patch 0860_allreduce_provider_cli. Replace env GGML_CUDA_ALLREDUCE with --allreduce {auto,ccl,host,adaptive,p2p,root3,butterfly} and orthogonal --allreduce-wire {native,q8}; unknown name or unsupported provider+wire combo is a startup error. Old names (nccl/internal/hybrid) and GGML_CUDA_AR_WIRE removed; all callers/docs/tests migrated in one change (no shims). Emits BIGCHERRY_PATCH_HIT naming the selected provider. Updates requires/conflicts for 0840, 1244, 1250, 1252. 2026-09-30 GPT review: common/arg option callbacks now only record provider/wire and a single common_apply_allreduce_config() runs after parse_cli_args(), so option order no longer matters (previously '--allreduce-wire q8 --allreduce p2p' failed). Focal-overlay rebase check CLEAN; ARG_CPP mechanics tests added. Remaining: dual-XTX hardware matrix, both option orders.

## Steps

1. GPT authors package (req_2db947dd40ee4492 + amendment req_1a26da4733a148d2). 2. Apply, patch-lint, hardware-free tests under tools/tests/patch. 3. Migrate docs/reference/architecture/MULTI_GPU_DISPATCH.md and scripts. 4. Dual-XTX matrix: auto/ccl/host/adaptive (p2p/root/q8 only as fail-closed unless supported).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Hardware-free anchor tests; then order-balanced ab-benchmark per provider on GPUs 0,1 (Qwen3.8-27B Q8_0, -sm tensor, MTP n_max=4), activation marker per arm.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Framework change: no promotion gate. Q8 wire is lossy activation compression, orthogonal to model weight quant.

2026-09-30 AllReduce hardware matrices queued on Brutus (commits 65091e94, aa7a5dcb) ahead of 0860 using the stock GGML_CUDA_ALLREDUCE selector per A/B arm: dual-XTX 27B ccl/host(internal)/none (ab-27b-allreduce), 0840 adaptive vs control (ab-27b-adaptive), and 3-GPU XTX x2 + R9700 (gfx1100,gfx1201 multi-arch build) ccl / 1244 root3 internal / none (ab-3g-allreduce, experiment allreduce-root3). q8 wire (1250) not queued: it requires 1252 p2p provider, which needs working PCIe P2P (off on the stock kernel).

2026-09-30 superseded by PGC09: 0860 is part of the AllReduce promotion closure (0860 + 1225 + 0840); its evidence and promotion are tracked there. 0860 marker fixed to WARN (86c96eed) so it shows at default server verbosity.

## Change Log

- 2026-09-29T09:33:59.429712+00:00 (created-by): Created by agent

## Ledger-events


- chg_20260929_135722_allreduce-methods-are-now-sele_7144
- 2026-09-29T13:57:26.139352+00:00 (updated-by): Updated: section:ledger-events
- chg_20260929_215656_dual-xtx-27b-q8_0-plain-decode_1707
- 2026-09-29T21:57:02.675576+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-29T23:13:55.054240+00:00 (updated-by): Updated: section:description
- chg_20260929_231423_the---allreduce-and---allreduc_8647
- 2026-09-29T23:14:26.365015+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-30T01:22:47.196698+00:00 (updated-by): Updated: section:notes
- chg_20260930_021419_builds-firing-checks-and-ab_8646
- 2026-09-30T02:14:23.019228+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-30T12:49:28.815429+00:00 (updated-by): Updated: section:notes
- 2026-09-30T12:49:32.000222+00:00 (state-transition): State: pending → superseded
- chg_20260930_125153_reviewed-and-consolidated-toda_3870
- 2026-09-30T12:52:03.237715+00:00 (updated-by): Updated: section:ledger-events
