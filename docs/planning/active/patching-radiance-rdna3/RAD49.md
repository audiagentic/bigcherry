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

## Change Log

- 2026-11-08T11:39:12.700554+00:00 (created-by): Created by agent
