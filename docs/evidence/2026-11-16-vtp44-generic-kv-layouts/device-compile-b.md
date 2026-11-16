# Patch B device compile: bf16_fp8 / bf16_i8 on the gqa8-body units (gfx1201)

`r4d_attn_paged_h256_gqa12.hip` (gqa 12, block 4) and `r4d_attn_paged_h256_gqa6_bs4.hip` (gqa 6, block 4) compiled for
gfx1201 with the production flags plus `-Rpass-analysis=kernel-resource-usage`, with `R4D_ATTN_B16 1` added to both units.
No errors, no warnings. Spilled VGPRs per kernel (a kernel is one wave-count variant; decode `nw` = waves = ceil(q_len * gqa / 16)):

| layout | kind | variants | max VGPR | spilled VGPRs per variant |
|---|---|---|---|---|
| existing (fp8, bf16, fp8_v4, k6_v4 ...) | prefill | 8 | 240 | 9, 8, 0, 0, 10, 10, 0, 0 |
| existing | decode | 16 | 256 | 6, 13, 6, 0, 20, 0, 25, 0, 16, 0, 0, 0, 14, 0, 0, 0 |
| bf16_fp8 | prefill | 2 | 240 | 6, 5 |
| bf16_fp8 | decode | 4 | 256 | 45 (nw 1), 22 (nw 2), 7 (nw 3), 0 (nw 4) |
| bf16_i8 | prefill | 2 | 238 | 0, 0 |
| bf16_i8 | decode | 4 | 256 | 22 (nw 1), 0, 0, 0 |

The two units give identical counts: the row tile is 16 query rows, so gqa enters the indexing only. At gqa 12 a decode of one
token is 12 rows, one wave (nw 1), the common decode shape. bf16_i8 is at or below the existing worst case (25). bf16_fp8's
single-wave decode spills 45 VGPRs against at most 20 for the existing single-wave variants: a performance risk to measure on
hardware (Flash-Next decode A/B), not a correctness one. Classification: bf16_fp8 = option bit 2^30 set; bf16_i8 = bit 128 set
and the rest equal to another instantiation's option word.
