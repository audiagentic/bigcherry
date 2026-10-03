# R9X — r9700-stack cross-RDNA extraction campaign

Status: active planning group
Prefix: `R9X`
Source baseline: `bkvargyas/r9700-stack@8dfda41f25f8ff6837cce48bd52b7fd249648f0e`
BigCherry branch at plan creation: `patch-refactor`
BigCherry patch high-water mark at plan creation: `1292_kpool_tail_truncate`
Proposed implementation slots: `1300`–`1309` (advisory only; re-scan `patches/` before creating any patch).

## Objective

Extract the useful kernel/dataflow ideas from r9700-stack into BigCherry's llama.cpp/HIP patch stack without treating gfx1201 code as universally portable. The campaign must produce independently reviewable BigCherry patches, preserve current fallbacks, and deliberately test what can be re-targeted to RDNA3 (`gfx11`) and RDNA2 (`gfx103x`) as well as RDNA4 (`gfx12`).

This group does **not** supersede `patching-nasone-rdna-optimizations` (PNRO), `rdna-boosts`, or the current 127x/128x/129x Qwen4Exp/collective work. Every R9X item must first reuse, extend, or explicitly decline existing mechanisms rather than create a parallel implementation.

## Architecture policy

| Tag | Hardware | Rule |
|---|---|---|
| `common` | all AMD HIP | Prefer shared graph/dispatch/layout/cache ideas here. No ISA-specific assumptions. |
| `gfx12` | RDNA4 / gfx120x | Native reference target for r9700-stack extraction; gfx12 WMMA/intrinsics allowed behind exact capability gates. |
| `gfx11` | RDNA3 / gfx110x, gfx115x | Reuse dataflow/fusion/layout where valid; retune wave/tile geometry and replace gfx12-only instructions. |
| `gfx103x` | RDNA2 | Reuse architecture-neutral scheduling/cache/launch reductions; use supported MFMA/vector paths and correct wave mode. Never copy gfx12 intrinsics blindly. |

Rules for every implementation:
1. Fail closed: unsupported arch/shape/type takes the pre-R9X path unchanged.
2. Prefer one capability/dispatch layer over duplicated per-generation patches.
3. Keep numerical-format changes distinct from launch/tile changes so regressions can be bisected.
4. Prove activation with `BIGCHERRY_PATCH_TRACE` or the current project-equivalent marker.
5. Validate correctness before performance; microbench wins alone do not justify promotion.
6. Benchmark decode (`M=1/2/4/8`), short prefill, long prefill, and tensor-split/multi-GPU where the code participates.
7. A common-path patch cannot promote with a regression on any supported RDNA generation used by BigCherry.

## Campaign map

| Plan | Area | Proposed patch slot(s) | Primary existing overlap |
|---|---|---|---|
| R9X01 | Intake, mapping, baseline, capability matrix | none | PNRO; 1273–1292 |
| R9X02 | Attention + QSA/indexer | 1300, 1301 | 1201/1202/1203/1266/1270/1271 |
| R9X03 | GDN short-prefill/MTP/conv | 1302 | 1221/1253/1254/1255 |
| R9X04 | MoE router + hyper-connection decode fusion | 1303, 1304 | 1207/1237/1256/1257/1279 |
| R9X05 | MXFP4/FP8 MoE + skinny GEMM | 1305, 1306 | 1203/1237/1241/1262/1265/1267/1273/1274 |
| R9X06 | Expert residency/LRU + pinned-host backing | 1307 | 1235/1280/1286/1292 |
| R9X07 | AllReduce/WHT/compressed collective | 1308 | 1001/1244/1250/1252/1272/1275/1276/1277/1290/1291 |
| R9X08 | PLE, norm+RoPE, transpose microfusions | 1309 | 1004 + qwen4exp model path |
| R9X09 | Cross-generation integration/promotion | none by default | all above |

## Source inventory to pin in evidence

r9700-stack reference kernels: `kernels/r9k_attn.hip`, `r9k_qsa.hip`, `r9k_qsa_score.hip`, `r9k_gdn.hip`, `r9k_router.hip`, `r9k_hc.hip`, `r9k_moe_mxfp4a8.hip`, `r9k_gemm_fp8.hip`, `r9k_ple.hip`, `r9k_norm_rope.hip`, `r9k_transpose.hip`, `r9k_ar.hip`, `r9k_ar_wht.hip`, `r9k_ar4.hip`, and `kernels/third_party/davetha/r4d_lru.hip`.

Host/reference adapters include `r9700_vllm/attn/qsa.py`, `r9700_vllm/kernels/attn.py`, `r9700_vllm/models/gdn.py`, `r9700_vllm/kernels/moe.py`, `r9700_vllm/router.py`, `r9700_vllm/hc.py`, `r9700_vllm/moe/cache.py`, `r9700_vllm/kernels/ple.py`, `r9700_vllm/ple/int6.py`, and `r9700_vllm/ple/short_conv.py`.

## Patch allocation discipline

`1300`–`1309` are reservations, not ownership locks. Before implementation, re-read the branch head and the maximum numeric directory under `patches/`. If another agent has consumed a slot, preserve the R9X plan ID but allocate the next free patch number and update this README plus the corresponding plan. Every created patch package must follow current BigCherry package conventions (`patch.py`, `patch.toml`, `SUMMARY.md`, validation contract/evidence where required).
