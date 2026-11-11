# rad0019_vtp14_n2_tp_ownership_fp8

**Status:** untested
**Plan item:** VTP14 (N2)

Scope: `--tp-ownership qwen38-27b-6-6-12` for the fp8 Qwen3.8 plugin (`qwen35_fp8`). That plugin serves the quantised
27B containers, the ones that fit two RX 7900 XTX, so the matched TP2-vs-TP3 comparison needs it.

## Summary

Gives the fp8 Qwen3.8-27B plugin the same 6/6/12 asymmetric three-rank ownership as rad0018. The patch uses the fp8
blocks' existing per-rank spans: attention by head, delta-net uneven heads (`k_first`/`v_first`), and LinearFP8 row
and column spans. Every span is whole 128-row scale blocks.

## Outcome

Card-free only so far. The patch applies after rad0018 and is idempotent.

## Detail

- `qwen35_fp8` opts in. Its delta-net heads and first heads come from the profile, and it declares explicit
  vocabulary spans for `token_embd` and `output`.
- Under a profile, the plugin refuses two configurations:
  - a vocabulary-weighted head (`output.k<B>`);
  - a speculative drafter, because the drafter's geometry is not owned. Run with `--num-speculative-tokens 0`.
- In `rad_block_attn_gated_fp8.h` and `rad_block_mlp_fp8.h`, the spans apply only when `Geom::tp_owned` is set.
  Eight plugins share these blocks, but only one that opted in can set the flag.
