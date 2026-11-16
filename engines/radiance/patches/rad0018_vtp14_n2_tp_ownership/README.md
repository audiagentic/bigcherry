# rad0018_vtp14_n2_tp_ownership

## Summary

Radiance divides a model evenly over `--tp` ranks (`geom_from` in `arch/common/rad_arch.h`). An even split fails or
wastes memory when the head or row counts do not divide by the world, or when the cards differ. The dense
Qwen3.8-27B on two RX 7900 XTX (24 GiB) and an R9700 (32 GiB) is the case that motivated this patch.

`--tp-ownership SPEC` sets an uneven split, one count a rank in `--rank-devices` order:

```
kv=<n,..>;ffn=<n,..>[;q=<n,..>][;lk=<n,..>;lv=<n,..>]
```

| key | what | default |
|---|---|---|
| `kv` | attention KV heads | required |
| `ffn` | feed-forward rows | required |
| `q` | attention query heads | `kv` × the model's query-per-KV group |
| `lk`, `lv` | delta-net key and value heads (together) | required by a plugin with a delta net |

The patch names no model and no split. The spec is configuration: the serving manifest's `scheduler.tp_ownership`,
set per layout, so a sweep is a manifest change and needs no rebuild. `geom_from` checks the spec against the loaded
model and refuses, with the reason:

- a count list whose length is not the world, an empty share, an unknown or repeated key;
- shares that do not sum to the model's query heads, KV heads or feed-forward width;
- a rank whose query heads are not whole groups of its KV heads;
- delta-net shares that do not sum to the model's, or a rank whose value:key ratio differs from the model's.

Each rank's first head and row are the prefix sums of the shares. The vocabulary stays evenly divided (with explicit
spans where the world does not divide it). The all-reduces are M × hidden on every rank, so they stay identical.

Example, the dense Qwen3.8-27B over (XTX, XTX, R9700), the config_test value:
`kv=1,1,2;ffn=5120,4864,7424;lk=4,4,8;lv=12,12,24` (query heads 6/6/12 by default).

## Outcome

Untested (not promoted). On cards, with rad0016 and rad0017, the dense Qwen3.8-27B BF16 split 6/6/12 over
(XTX, XTX, R9700) serves the 64-layer model with correct output: 23.4 tok/s single stream, 4 concurrent streams
correct (smoke numbers; see VTP14.md). Split sweeps on the MXFP4 and w8a8 containers are in VTP14.md.

## Detail

- **ABI.** `RadBuildCtx` gains a trailing `tp_ownership` field. Null means the even split, as upstream.
- **Core.** `--tp-ownership SPEC` is added to config, help and explain, with a `config_test` value. The parser checks
  the form only (keys, digits, separators); the engine passes it into the build context.
- **`rad_arch.h`.** `Geom` carries the per-rank first heads and rows and the KV and feed-forward shares
  (`own_kv_split`, `own_ff_split`, which a drafter scales from). `bc_own_spec_get` parses one key;
  `bc_own_linear_heads` checks and sets a plugin's delta-net heads. `geom_from` also refuses a spec for an
  architecture that has not opted in (`Geom::tp_owned_ok`), and calibration, whose taps assume the even split.
- **`qwen35_bf16`.** The plugin opts in and sets its delta-net heads through `bc_own_linear_heads`. It declares
  explicit vocabulary spans for `token_embd` and `output`, because the loader refuses an uneven vocabulary split
  without a span.
- **Blocks** (`rad_block_attn_gated.h`, `rad_block_gdn.h`, `rad_block_mlp.h`). The weight spans apply only under
  `Geom::tp_owned`.

The span arithmetic follows the VTP30 experiment (`rad0016_vtp30_native_tp3_ownership` in the xtx27-prefill
worktree). This patch differs: the split is generic typed configuration rather than an environment variable or a
named profile, the vocabulary spans and calibration refusal are new, and the plugin opts in through a `Geom` flag.

### Limits

- Plugins that opt in: `qwen35_bf16` here, `qwen35_fp8` and `qwen35_w8a8` with rad0019.
- Per-rank KV and delta-net state sizes differ between ranks. Admission and KV tiers have to be checked per rank,
  never with rank 0 as a proxy (N5).

### Promotion record

Model: Qwen3.8-27B MXFP4 and w8a8 (.rad), tensor parallel over two RX 7900 XTX and an R9700 (TP3)
Mechanics: the patch's tests in tools/tests/patch pass and patch-lint is clean on the promotion PR; the lane builds carrying it (VTP logs named below) build and serve.
Activation: the TP3 world over (XTX, XTX, R9700) starts and serves only with this chain (rad0016 per-rank code selection, rad0017 shared rendezvous, rad0018/rad0019 uneven ownership); without rad0017 any libr3+libr4d world hangs in all-reduce init (reproduced on cards, VTP14.md), the engine reports the rendezvous as shared by libr3.so and libr4d.so and the ownership split in its startup report.
Identity: TP3 6/6/12 against an exact TP2 reference over 42,761 positions: KLD mean 0.00307, p99.9 0.084, top-1 96.80%, PPL x0.9993; the production TP2 wht6 wire scores 0.00391 / 0.111 / 96.57% on the same corpus (VTP14.md, N7, build a0979ca2). Launchers qwen27b-tp3, -int8 and -capacity start from their layouts with greedy output identical to hand-flag runs.
A/B: control cannot run (mixed-library world hangs without the chain); TP3 6/6/12 serves 64-65 tok/s at about 0.05 s TTFT, correct (VTP14.md milestone table); w8a8 TP3 KLD against BF16 within TP2's median/p99/p99.9 (VTP14.md N6).
