# 1330_qsa_mask_inplace

QFP17 dense-QSA memory enabler against llama.cpp b11474.

## Switch

- Package/runtime default: `BIGCHERRY_QSA_MASK_INPLACE=0`.
- Not set by any profile (see Outcome).
- Mode `2`: diagnostic contiguous copy before flash attention.
- In-place mode is restricted to `n_tokens > 8`; decode/MTP-sized batches keep the exact production dense path.

## The ub1024 candidate

Brutus 2026-10-09, Flash-Next, 240K context, ~80K fill. 1330 on at ub1024 fits where production does not:

| arm | prefill t/s | decode t/s |
|---|---:|---:|
| production, ub512 | 1206.9 / 1228.9 | 64.4 / 64.5 |
| 1330 on, ub1024 | 1376.9 / 1371.6 | 66.1 / 66.3 |

Prefill is about +13%. Greedy text was identical. One ub512/24K A/B showed decode 75.7 / 76.3 off versus 73.4 on; the package therefore now leaves <=8-token decode/MTP batches unchanged.

At 240K/ub1024 the removed F16 `[n_kv,T]` ADD result is ~480 MiB per Meta rank. Prior peak accounting was ~1748 MiB/rank; removing that live tensor gives a ~1268 MiB theoretical peak before allocator effects. The 256-row padding costs only ~0.5 MiB at this shape.

## Composition at ub1024

- 1332: leave `BIGCHERRY_QSA_CHUNK` off for this candidate. If chunking is enabled, >8-token prefill takes 1332's I32/chunked path and bypasses 1330's dense add, so the mechanisms are alternatives for the large-batch QSA mask.
- 1345: its MoE routing multi-warp threshold is `n_tokens >= 128`; both full ub512 and ub1024 prefill chunks already select it. ub1024 changes work per launch, not the full-chunk mode.
- 1348: target prompt chunks grow from up to 512 to up to 1024 tokens. For a fixed prompt this roughly halves chunk count; deferred catch-up processes twice as many tokens per full chunk and has fewer submit/catch-up boundaries plus one final flush. Re-check 1346 prompt timing when qualifying the launch change.

## Outcome

Not adopted at ctx 245760: the configuration fits at load and fails with an illegal memory access once about
118K tokens are in the context. Clean at ctx 196608 and below. State `evaluated`; the runs are in SUMMARY.md.
