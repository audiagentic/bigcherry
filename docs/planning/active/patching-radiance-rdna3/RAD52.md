---
id: RAD52
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-11-14T06:57:44.790336+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Prefill attention fills the card at every piece size (one-KV-head ranks, gfx1100 native split form)

## Description

In native TP3 an XTX rank holds one KV head (`kv=1,1,2`), so a prefill piece of R rows launches `ceil(R/64)` 768-thread workgroups on a 48-WGP card: 16 at 1024 rows. On `qwen27b-tp3-int8-556-long-bf16` at 100k tokens the XTX ranks' attention is the whole quadratic term of TTFT (18.7 s of 53.8 s from a two-point a·N + b·N² fit), runs at ~23-26 TFLOP/s against a ~142 TFLOP/s f16 roof (the R9700, 2 KV heads, ~70), and costs about the same per call at 1024 and 2048 rows: `--max-num-batched-tokens 4096` halves the term (100k TTFT 53.8 -> 44.7 s, 8k unchanged). The KV split meant to fill the card (`rad_prefill_splits_law`) is either not taken on this path or runs in the emulated-fragment form: libr3's native instantiation refuses it (`static_assert(!R3N || !(OPT & O_PSPLIT))`, `native/r4d_attn_prefill_h256_gqa6.hip.rw`). This item makes the cost per query row flat in piece size on every rank, from the rank's measured geometry, with no model or card constant.

## Steps

1. Card-level matrix on one XTX and the R9700 (selftest `--perfpaged` or rad-kbench): bf16 and fp8 KV, kv_heads 1 and 2, rows 128-4096, context 8k / 50k / 100k, forced splits 1-12 and the law's choice (print `last_splits`), native and compatibility forms; us per row and TFLOP/s. Decide which cause holds: split not taken (wgp_count, scratch, `kv_used_max_h` on the pipelined piece path), split taken in the compatibility form, or a one-workgroup-a-WGP law that is too conservative.
2. Trace one engine piece launch (`splits`, `wgp_count`, `scratch_bytes`, `q_len`, `total_q`) at 1024- and 2048-row pieces on both architectures.
3. libr3: native instantiation with the split store (convert the accumulator before the partial store, as the bf16 store already does); selftest rows against the compatibility form.
4. If two workgroups a WGP pay: the law's fill target as a tunable axis of the attention rows (`tuning.tune_pin`), calibrated per rank, default unchanged.
5. In-engine: `--profile-ops` at 2048 and 4096 steps (attn_paged us/call per rank), then ABBA on TTFT at 8k / 60k / 100k, the long prompt beside 4 streams, and `busy_prefill_chunk` 512 / 1024.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Selftest identity where the split count is unchanged; the VTP37 reordering gate (noise floor against serial, no loss against BF16) for any changed split count; int8 stress (12 runs); 2x exploratory and 3x final ABBA.

## Effort & Risk

M. A changed split count changes summation order (reordering gate, not bit identity). If the law changes it must stay a measured per-rank value. Expected TTFT at 100k -10% to -22% at 2048-token steps; a long prefill beside decoders with small chunks -20% to -35%; nothing at <= 8k or in decode.

## Standards



## Acceptance Criteria

- [ ] us per query row at 50k and 100k keys within 20% from 256- to 2048-row pieces on an XTX rank with one KV head, or a recorded NO-GO with the matrix that shows why.
- [ ] TTFT at 100k at least 10% better at 2048-token steps against the same build without the change; no regression at 512 / 2048 / 8192 tokens or in decode.
- [ ] No model, card or world constant: geometry from the rank's device properties and a per-layout axis.

## Notes

Created 2026-11-14 from the TP3 long-context profile and A/B (`docs/evidence/2026-11-14-tp3-long-context/README.md`, sections 2b-2d). Related: VTP36 (programme), VTP39 (small chunks become affordable), RAD53 (fp8 read on the same kernel), VTP38 (attacks the linear term; the two add).

## Change Log

- 2026-11-14T06:57:44.790336+00:00 (created-by): Created by agent
- 2026-11-14T06:58:35.979256+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria, section:notes
