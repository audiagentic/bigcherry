# 1006 RDNA4 MMQ Q6_K codegen fix

Split from `1000_rdna4_mmq_q2k_q6k_fix` (rejected 2026-09-16, PA35) after
real hardware evidence separated the two independent edits it contained:
Q6_K genuinely improved (1.365x, CI95 1.362x-1.367x), Q2_K genuinely
regressed (0.959x, CI95 0.954x-0.963x). See
`patches/1000_rdna4_mmq_q2k_q6k_fix/SUMMARY.md` for the full disposition
record.

This patch carries only the Q6_K float-promotion edit.

## Promotion evidence (2026-09-29, gfx1201)

4 real hardware sessions on `tierA-qwen4b-q6k` (Qwen3.5-4B, Q6_K), paired
llama-bench, 10 rounds/session. 3 of the 4 (sessions 1, 3, 4) are clean and
closely agree; session 2 was run immediately back-to-back with no cooldown
gap and shows a GPU clock-instability confound (bimodal per-round values on
both arms alike) -- recorded but excluded from interpretation.

BigCherry-internal control/subject A/B (control = `bigcherry-tuning`
baseline, subject = baseline + this patch):

| session | prefill pp512 | decode tg128 |
| --- | --- | --- |
| 1 | +18.31% [CI95 18.04%, 18.75%] | flat -0.04% [-0.16%, +0.08%] |
| 3 | +18.12% [CI95 17.99%, 18.25%] | flat +0.01% [-0.11%, +0.11%] |
| 4 | +18.04% [CI95 17.96%, 18.12%] | flat +0.02% [-0.06%, +0.12%] |

3-arm comparison against **native llama.cpp** (each session's own
`reference-ladder.json`, all four arms built from the same revision:
`stock` = true upstream-native llama.cpp with zero BigCherry patches,
`validated` = current BigCherry `validated-enhancements` baseline,
`validated+patch` = baseline + this patch). Sessions 3 and 4 agree closely;
session 1's ladder shows the same instability as session 2 and is excluded:

| session | validated vs stock (pp512) | validated+patch vs stock (pp512) | validated+patch vs stock (tg128) |
| --- | --- | --- | --- |
| 3 | +10.68% | +31.20% | +0.11% |
| 4 | +10.50% | +30.39% | +0.05% |

This shows BigCherry's own baseline is already meaningfully ahead of
native/vanilla upstream llama.cpp on this shape (~+10.5-10.7%, an existing
gain unrelated to this patch), and this patch adds a further real,
consistent gain on top of that (~+18%), for a combined ~+30-31% over
native llama.cpp with no decode regression in either comparison.

Contract `RDNA4-MMQ-Q6K-CODEGEN`'s own aggregate verdict over all 4
sessions: `status=pass`. `state = "validated"`.
