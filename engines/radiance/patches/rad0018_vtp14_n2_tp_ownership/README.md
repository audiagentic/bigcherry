# rad0018_vtp14_n2_tp_ownership

## Summary

Radiance divides a model evenly over `--tp` ranks (`geom_from` in `arch/common/rad_arch.h`). The dense Qwen3.8-27B has
4 KV heads and a 17408-wide feed-forward, so three ranks cannot split it evenly. Its three cards are also unequal: two
RX 7900 XTX with 24 GiB each and an R9700 with 32 GiB.

`--tp-ownership qwen38-27b-6-6-12` assigns each rank, in `--rank-devices` order, the following (first head or row in
brackets):

| rank | Q heads | KV heads | GDN K heads | GDN V heads | FFN width | vocabulary rows |
|---|---|---|---|---|---|---|
| 0 | 6 (0) | 1 (0) | 4 (0) | 12 (0) | 5120 (0) | [0, 82816) |
| 1 | 6 (6) | 1 (1) | 4 (4) | 12 (12) | 4864 (5120) | [82816, 165632) |
| 2 | 12 (12) | 2 (2) | 8 (8) | 24 (24) | 7424 (9984) | [165632, 248320) |

`--tp-ownership qwen38-27b:K0,K1,K2:F0,F1,F2` sets the delta-net key heads (value heads are three times as many) and
the feed-forward rows per rank, with attention as 6-6-12, so that a sweep needs no rebuild. `geom_from` checks that the
key heads sum to 16, the rows sum to 17408 in 256-row tiles, and no share is empty. A sweep of five splits on the
MXFP4 container (4 and 8 streams, speculation off and at depth 3) found none faster than 6-6-12: the R9700's
remaining lag is its x4 link in the all-reduces, not its share of the compute (VTP14.md, N2 rebalance).

The named profile comes from one table (`PROFILES` in `patch.py`). Every rank keeps whole heads, with Q:KV = 6 and
GDN V:K = 3. The all-reduces are M × 5120 on every rank, so they stay identical across ranks.

## Outcome

Untested (not promoted). On cards, with rad0016 and rad0017, `--tp 3 --rank-devices <XTX>,<XTX>,<R9700> --tp-ownership qwen38-27b-6-6-12` serves the 64-layer BF16 model with correct output: 23.4 tok/s single stream, 4 concurrent streams correct (smoke numbers; see VTP14.md).

## Detail

- **ABI.** `RadBuildCtx` gains a trailing `tp_ownership` field. Null means the even split, as upstream.
- **Core.** `--tp-ownership NAME` is added to config, help and explain, with a `config_test` value, and the engine
  passes it into the build context.
- **`rad_arch.h`.** `Geom` carries the profile's per-rank first heads and rows. `geom_from` refuses:
  - a profile for an architecture that has not opted in (`Geom::tp_owned_ok`);
  - an unknown profile;
  - any model other than the original dense 27B;
  - a world other than 3;
  - calibration, because its taps assume the even split's global widths.
- **`qwen35_bf16`.** The plugin opts in and sets the GDN heads. It declares explicit vocabulary spans for
  `token_embd` and `output`, because 248320 is not a multiple of 3 and the loader refuses an uneven split without a
  span.
- **Blocks** (`rad_block_attn_gated.h`, `rad_block_gdn.h`, `rad_block_mlp.h`). The weight spans apply only under
  `Geom::tp_owned`.

The span arithmetic follows the VTP30 experiment (`rad0016_vtp30_native_tp3_ownership` in the xtx27-prefill
worktree). This patch differs in three ways:

- the profile is typed configuration rather than an environment variable;
- the vocabulary spans and the calibration refusal are new;
- the plugin opts in through a `Geom` flag instead of a new default parameter.

### Limits

- BF16 plugin only. The fp8 and w8a8 plugins come with N6.
- Per-rank KV and GDN state sizes now differ between ranks. Admission and KV tiers have to be checked per rank,
  never with rank 0 as a proxy (N5).
