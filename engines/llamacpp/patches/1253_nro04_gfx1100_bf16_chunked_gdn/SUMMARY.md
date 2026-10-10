# 1253_nro04_gfx1100_bf16_chunked_gdn

**Status:** validated (production `validated-enhancements`; historical b11126 hardware contract)
**Plan item:** PNRO04

## Implemented

Separate gfx11/gfx12 BF16 WMMA K=1 GDN prefill kernels; two ordered launches (KKT scratch, then scan/output/state), stock sequential fallback, opt-out `GGML_CUDA_GDN_CHUNKED_BF16=0`. Source: nasone `4169fbbf`. **Requires no 1221; conflicts with 1221.** 1254 MTP prefix-tail is a separate untested dependent.

## Measured evidence

Four bound sessions per gfx1100/gfx1201 at b11126: pp512 gfx1100 +8.2/+8.8/+7.7/+9.3%, gfx1201 +10.4/+11.8/+10.6/+11.0%; tg128 approximately flat; backend-reference pass. Earlier incomplete receipts are not validation. No b11474/current-pin performance claim.

## Remaining boundary

The dispatcher lacks an explicit supported GQA ratio check before KKT wrappers' `GGML_ABORT` default (ratios other than 1/2/3/4/6/8). BF16 packing uses `bits+0x8000`, which differs from RNE on halfway ties (host fixture: 16,256/32,512 positive-normal exact ties). No GPU failure demonstrated. Upstream #29353's HIP path is gfx115x-only and does not replace gfx1100/gfx1201 1253. PNRO04 / BCOP113 own current-pin attribution and fail-closed admission qualification; no new kernel/queue is authorised by this note.
