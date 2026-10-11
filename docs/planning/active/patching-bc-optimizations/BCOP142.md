---
id: BCOP142
order: 142
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-12T03:12:23+11:00'
created-by: agent
priority: P2
work: S
---

# RNX04: retire duplicate HC POST fusion

## Discovery / disposition

RNX04's historical 1304 raw-inject HC POST fusion is already implemented by validated default-on 1355_hc_post_gate_fuse with 1344_dsv4_hc_grid_index and 1313_scale_act_fuse. QFP35 is completed. Retire 1304, do not add a second HC matcher/kernel/allocator. RNX04 retains only conditional router-GEMM attribution; MET09/MET11 own routing ties, QFP48 active decode fusion, RNX09 integration.

## Evidence / eligibility

RNX04's substantive work predates 12 hours; QFP35 last independent closure was 2026-10-10 13:55 UTC, 1355 earlier. No independent HC gate work, queued HC benchmark or RNX04 PR was identified in the 12 hours ending 2026-10-11 16:12 UTC. Distinct from BCOP141 MoE tie and BCOP140 Vulkan queue. Protected PA49/b11552, QFP41/1356, QFP48, QFP49/1360/1361, PGC17 and MTP work were not edited. Historical b11474 1355 decode +0.5–0.7% with greedy/acceptance parity, no new b11552 result. External BF16 combine+RMSNorm is not llama.cpp F32 HC_POST. Source-static 14/14, flat-grid 61,102/61,102 and host semantic-distinction fixtures passed; no HIP build, pytest or GPU run.

## Bounded terminal gate

After active PA49/QFP48 work stops, reuse its existing trace for router dtype/M/N/K, actual BLAS/MMQ route and critical-rank wall share. If theoretical E2E ceiling <3% or no legal typed layout, close RNX04 without a patch. Otherwise one default-off typed prototype; full logits/KLD/greedy/MTP/graph/repeated-request controls; four sessions/architecture and ten paired ABBA rounds; promote only CI95-low >=3% E2E with <=1% control regression. No experiment queued. Technical design stays in RNX04.
