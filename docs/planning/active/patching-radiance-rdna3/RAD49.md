---
id: RAD49
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-11-08T11:39:12.700554+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P1
---

# flashnext-tp2xtx: dense/attention TP2 on the XTX pair, R9700 + RX 6900 XT hold the experts (Strata-informed)

## Description

Owner direction 2026-11-08: put the dense layers (attention, GDN, QSA, dense/shared projections) tensor-split across the two RX 7900 XTX cards, optimised as in the original multi-card setup, and use the R9700 (gfx1201) and RX 6900 XT (gfx1030) as expert holders (remote-MoE helpers). Use the Strata-rdna3 findings (the strata-rdna3 analysis, kept in the task scratch folder outside the repository) to boost it. Radiance already runs Flash-Next TP2 (--tp 2, host-memory all-reduce, wht6 wire); bigcherry runs qwen27b-2xtx TP2. Blocker: moe-remote refuses more than one helper behind a multi-rank engine (moe_remote_hook.cpp r3_moe_mailbox_device; step_slot uses the slot axis for either rank or helper).

## Steps

1. Card-free: layout flashnext-tp2xtx (engine gfx1100 count 2, tp-wire, helper gfx1201 placement head) and launcher/manifest support for a TP engine with helpers; inventory every TP1 assumption in the helper path (rank-sliced vs full-width expert results under TP, combine, owner bitmaps, placement policy).
2. Hardware (gpu lease): TP2 XTX + R9700 helper (the one-helper case the protocol allows today) vs flashnext-4gpu with the parity harness; exact BF16 wire first, then wht6. Decide go/no-go from decode ms/step + acceptance, doc and long prefill.
3. N helpers behind M engine ranks (slot = (rank, helper)), so the RX 6900 XT tail joins; GPT design review; KL gate.
4. Strata ideas inside our design: heat-derived expert deal across helpers and engine-resident VRAM (disjoint, no fixed budgets), exchange overlap, length-dependent prefill path, gfx11 GDN/QSA kernels behind strict numerics.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Parity harness (tools/lab/radiance/flashnext-parity.sh) against flashnext-4gpu baseline; same-session A/B for any claim; 4-GPU KL gate vs the R9700-alone control; decode and prefill reported separately.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

No P2P (host-memory transport only). No hand-picked budgets: derive from runtime VRAM, expert sizes, heat. Kernel changes only in libr3/moe-remote or managed patches. MIT attribution for any Strata code.



2026-11-09/10 outcome (findings: runs/rad49-findings-20261109.md): TP2 on the two XTX loses to the R9700-engine layouts on every metric. flashnext-tp2xtx-r9700 vs flashnext-2xtx (A B A B, runs/rad49-3card-split): prefill ~1,945 vs ~3,100 tok/s, decode 6 streams ~72 vs ~46 ms/step, context 6x65k vs 8x262k. TP2 prefill cost: host-memory all-reduce ~22% + helper waits ~15% (runs/rad49-tp-prefill-prof). Best layouts: flashnext-4gpu overall, flashnext-2xtx on three cards. Current main equals pre-RAD49 main on flashnext-4gpu within ~1% (runs/rad49-4gpu-regress). Follow-ons split out: tail kernel (tuning-radiance-tail-kernel TK01-05), host link (run-radiance-host-link HL01-05), speed-aware placement (RHP17).

## Change Log

- 2026-11-08T11:39:12.700554+00:00 (created-by): Created by agent

## Hardware findings (2026-11-08)

Same workload (65536 context), live policy off, flashnext-4gpu as control:

| layout / build | doc prefill tok/s | decode 1s ms/step | decode 6s ms/step | 6x8k sessions ms/step | cold 27k TTFT s |
|---|---|---|---|---|---|
| tp2xtx-r9700, #770 (helper computes each expert per rank) | 1,563 | 41.2 | 127 | 104 | 16.5 |
| tp2xtx-r9700, v25 full-width-once | 1,865 | 40.8 | 80.8 | 70.1 | 15.8 |
| tp2xtx-r9700, v25 + engine native MoE default on | 1,954 | 32.3 | 75.3 | 64.0 | 12.7 |
| tp2xtx (R9700 head + 6900 XT tail), same build | 1,994 | 32.9 | 75.6 | 74.0 | 17.9 |
| flashnext-4gpu | ~3,590 | ~24.9 | ~47.4 | ~43-46 | ~7.9 |

Per-op profile of the TP2 XTX engine (engine only, --profile-ops): routed-expert GEMM (moe_gemm_q) 51% of decode / 44% of
prefill, cross-card exchanges 11% / 23%, int8 trunk GEMM 14% / 6%, hc_read (fp8 mixing matrices) 10% / 7%.
Exchange alone (r3-arbench, exact wire): ~28-31 us at 1 row, ~1.7 ms at 2048 rows (wht6 0.67-0.85 ms).

Next, by expected gain: (1) native RDNA3 integer expert kernels (int8 activations; plain-i4 container converted,
KL gate pending; w4nl needs a two-pass integer form), (2) re-enable the fused gather + all-reduce with helpers (rad0002),
(3) per-layer consumer / combine choice from measured link rates (calibration at the measuring start, then live from
wait counters carried in the existing heat snapshot - one telemetry path), (4) hc mixing matrices int8 in the recipe.
The RX 6900 XT tail does not pay behind TP2 yet (its expert compute ~3.9 ms a request).
- 2026-11-10T00:11:22.301887+00:00 (updated-by): Updated: section:notes
