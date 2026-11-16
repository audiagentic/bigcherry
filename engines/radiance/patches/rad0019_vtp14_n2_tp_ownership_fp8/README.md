# rad0019_vtp14_n2_tp_ownership_fp8

**Status:** untested
**Plan item:** VTP14 (N2)

Scope: `--tp-ownership SPEC` (rad0018) for the fp8 Qwen plugin (`qwen35_fp8`, and `qwen35_w8a8` built from it). That plugin serves the quantised
27B containers, the ones that fit two RX 7900 XTX, so the matched TP2-vs-TP3 comparison needs it.

## Summary

Gives the fp8 Qwen plugin rad0018's uneven split, checked against the model the same way. The patch uses the fp8
blocks' existing per-rank spans: attention by head, delta-net uneven heads (`k_first`/`v_first`), and LinearFP8 row
and column spans. Every span is whole 128-row scale blocks.

## Outcome

On cards: the MXFP4 and w8a8 Qwen3.8-27B containers serve correctly at TP3 over (XTX, XTX, R9700); split
sweeps, A/Bs against TP2 and KLD against BF16 are in VTP14.md. The patch applies after rad0018 and is idempotent.

## Detail

- `qwen35_fp8` opts in. Its delta-net heads come from the spec (`bc_own_linear_heads`), and it declares explicit
  vocabulary spans for `token_embd` and `output`.
- Under a split, the plugin refuses a vocabulary-weighted head (`output.k<B>`).
- In `rad_block_attn_gated_fp8.h` and `rad_block_mlp_fp8.h`, the spans apply only when `Geom::tp_owned` is set.
  Eight plugins share these blocks, but only one that opted in can set the flag.
- **Drafter** (`drafter_geom`, `rad_block_dflash2.h`): under a split, the DFlash2 drafter scales the target's KV and feed-forward
  shares (`own_kv_split`, `own_ff_split`) to its own head and row counts, and refuses a split that does not scale to
  whole heads and rows. Its linears
  span through a new `DenseFP8::span`, which covers the block-fp8 scale and MXFP4 scale/ref planes. Its draft head
  stays whole on every rank, because a ragged vocabulary is not `vocab_head_sharded`.
- **MXFP4** (`LinearFP8::span_parts`): the reference plane is now spanned too.

### Promotion record

Model: Qwen3.8-27B MXFP4 and w8a8 (.rad), tensor parallel over two RX 7900 XTX and an R9700 (TP3)
Mechanics: the patch's tests in tools/tests/patch pass and patch-lint is clean on the promotion PR; the lane builds carrying it (VTP logs named below) build and serve.
Activation: the TP3 world over (XTX, XTX, R9700) starts and serves only with this chain (rad0016 per-rank code selection, rad0017 shared rendezvous, rad0018/rad0019 uneven ownership); without rad0017 any libr3+libr4d world hangs in all-reduce init (reproduced on cards, VTP14.md), the engine reports the rendezvous as shared by libr3.so and libr4d.so and the ownership split in its startup report.
Identity: TP3 6/6/12 against an exact TP2 reference over 42,761 positions: KLD mean 0.00307, p99.9 0.084, top-1 96.80%, PPL x0.9993; the production TP2 wht6 wire scores 0.00391 / 0.111 / 96.57% on the same corpus (VTP14.md, N7, build a0979ca2). Launchers qwen27b-tp3, -int8 and -capacity start from their layouts with greedy output identical to hand-flag runs.
A/B: control cannot run (mixed-library world hangs without the chain); TP3 6/6/12 serves 64-65 tok/s at about 0.05 s TTFT, correct (VTP14.md milestone table); w8a8 TP3 KLD against BF16 within TP2's median/p99/p99.9 (VTP14.md N6).
