---
id: BCOP144
order: 144
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-11T18:03:58Z'
created-by: agent
priority: P3
work: S
---

# QFP12: close nonwinning shared-expert tensor split 1306

## Discovery and disposition

The b11552 1306 opt-in shards Qwen4Exp shared expert but historical 24K/80K screens were neutral-to-slower, with changed MTP acceptance and only ~185 MiB/XTX, ~135 MiB/R9700 saved. Its “no extra AllReduce” claim requires runtime Meta coalescing evidence: exact use-count/shape/ADD adjacency/compute flags. Close QFP12 as deprecated; keep patch 1306 evaluated/default-off, unpromoted. No new implementation.

## Ownership and independent work

QFP12 completed plan is authoritative. QFP09 skew, QFP13 launches, RNX10 pointwise tail and QFP07 attention placement are separate owners. PA49, MET07, QFP41, QFP48, QFP49/1360/1361 and PGC17 had independent work in the preceding 12 hours and were not edited. 1306/QFP12 last substantive independent work Oct 3. Recent BCOP139-143 concern distinct capabilities. vLLM's shared-expert quant-group admission is reference only.

## Acceptance / rejection / terminal

No action unless new independently measured per-device VRAM admission or >=3% theoretical E2E critical-path opportunity emerges after the active collective work settles. Then require unchanged collective count, full logits/KLD/greedy/MTP correctness, multi-request/graph safety, four sessions and >=10 paired ABBA rounds with CI95-low >=3% E2E, <=1% control regression. Otherwise terminal default-off. Detailed design in completed QFP12 and patch SUMMARY.

## Validation

15/15 source-static checks passed; 10,000-case seeded F32 host fixture found 4,958 bit differences (max abs 0.00048828125), numerical-risk illustration only. No repository pytest, C++/HIP build, GPU benchmark or new BigCherry speed claim.
