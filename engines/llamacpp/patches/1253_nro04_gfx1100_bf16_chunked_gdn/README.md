# 1253_nro04_gfx1100_bf16_chunked_gdn

**State: validated; production validated-enhancements (historical b11126 qualification).** Plan: PNRO04. External fork source: nasone `4169fbbf50d24beb6d269a2350e7f780b85369e6`.

## Mechanism and actual dependency

1253 creates separate gfx11 and gfx12 BF16/WMMA chunked GDN kernels plus the chunked launch header. For K=1, non-KDA, S_v=128, n_tokens>1, it runs a KKT/inverse scratch pass and a scan/output/state-publication pass. The original F32 sequential kernel is retained as the fallback. `GGML_CUDA_GDN_CHUNKED=0` disables the chunked route; `GGML_CUDA_GDN_CHUNKED_BF16=0` disables the BF16 route. Both are otherwise default-on for eligible RDNA3/RDNA4.

**1253 does not require 1221.** `patch.toml` declares `requires=[]` and `conflicts=["1221_rd50_gdn_chunked_recurrence"]`; 1221 is a mutually exclusive FP32 chunked implementation. `1254_nro05_gdn_mtp_prefix_tail` requires 1253, but its K>1 MTP correctness/qualification is separate and remains untested.

## Measured scope and limitations

`evidence/validation.json` has four qualifying b11126 sessions each on gfx1100/gfx1201. `config/recipes.toml` records gfx1100 pp512 +8.2/+8.8/+7.7/+9.3% and gfx1201 +10.4/+11.8/+10.6/+11.0%, with approximately flat tg128 controls and GATED_DELTA_NET backend-reference pass. No current-pin b11474, gfx1151, gfx1030, no-P2P multi-rank or MTP benefit is established by these receipts.

Source admission caveat: outer BF16 dispatch does not check the KKT GQA-ratio specialization set `{1,2,3,4,6,8}` before selecting the wrapper; unsupported ratios hit `GGML_ABORT` rather than fallback. The `gdn_f2bf`/ `gdn_f2bf2` helpers use `bits+0x8000` (halfway round-up, not strict ties-to-even). These are source/host findings, not reproduced GPU failures. The once-per-process `BIGCHERRY_PATCH_HIT patch=1253_nro04 path=gdn_chunked_bf16` marker is not per-request completion proof. See PNRO04 / BCOP113 for bounded current-pin and admission gates. No implementation changes are included in this audit.
