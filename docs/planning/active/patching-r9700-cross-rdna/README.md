# R9X — r9700-stack cross-RDNA extraction campaign

Status: active planning group
Prefix: `R9X`
Source baseline: `bkvargyas/r9700-stack@8dfda41f25f8ff6837cce48bd52b7fd249648f0e`
BigCherry branch at plan creation: `patch-refactor`
BigCherry patch high-water mark at plan creation: `1292_kpool_tail_truncate`
Proposed implementation slots: `1300`–`1312` (advisory only; re-scan `patches/` before creating any patch).

## Objective

Extract the useful kernel/dataflow ideas from r9700-stack into BigCherry's llama.cpp/HIP patch stack without treating gfx1201 code as universally portable. The campaign must produce independently reviewable BigCherry patches, preserve current fallbacks, and deliberately test what can be re-targeted to RDNA3 (`gfx11`) and RDNA2 (`gfx103x`) as well as RDNA4 (`gfx12`).

This group does **not** supersede `patching-nasone-rdna-optimizations` (PNRO), `rdna-boosts`, or current Qwen4Exp/collective work. Every R9X item must first reuse, extend, or explicitly decline existing mechanisms rather than create a parallel implementation.

## Architecture policy

| Tag | Hardware | Rule |
|---|---|---|
| `common` | all AMD HIP | Prefer shared graph/dispatch/layout/cache ideas here. No ISA-specific assumptions. |
| `gfx12` | RDNA4 / gfx120x | Native reference target for r9700-stack extraction; gfx12 WMMA/intrinsics allowed behind exact capability gates. |
| `gfx11` | RDNA3 / gfx110x, gfx115x | Reuse dataflow/fusion/layout where valid; retune wave/tile geometry and replace gfx12-only instructions. |
| `gfx103x` | RDNA2 | Reuse architecture-neutral scheduling/cache/launch reductions; use supported vector/MFMA paths and correct wave mode. Never copy gfx12 intrinsics blindly. |

Rules for every implementation:
1. Fail closed: unsupported arch/shape/type takes the pre-R9X path unchanged.
2. Prefer one capability/dispatch layer over duplicated per-generation patches; separate kernels are allowed where ISA/fragment layout materially differs.
3. Keep numerical-format changes distinct from launch/tile changes so regressions can be bisected.
4. Prove activation with `BIGCHERRY_PATCH_TRACE` or the current project-equivalent marker.
5. Validate correctness before performance; microbench wins alone do not justify promotion.
6. Benchmark decode (`M=1/2/4/8`), MTP verify, short prefill, long prefill, and tensor-split/multi-GPU where the code participates.
7. Promotion is architecture-specific: gfx12/gfx11 may promote while gfx103x stays on fallback. A shared path cannot promote with a material regression on any generation that selects it.
8. Port semantics/dataflow into llama.cpp; do not transplant vLLM/Python registration machinery.

## Priority and campaign map

| Plan | Priority | Area | Proposed patch slot(s) | Primary existing overlap |
|---|---:|---|---|---|
| R9X01 | P0 | Intake, mapping, baseline, capability matrix | none | PNRO; 1273–1292 |
| R9X03 | P0 | GDN MTP decode + short-prefill fusion | 1302, 1312 | 1221/1253/1254/1255 |
| R9X04 | P0 | MoE router + hyper-connection decode fusion | 1303, 1304 | 1207/1237/1256/1257/1279 |
| R9X10 | P0 | Shared-expert decode fusion | 1310 | 1207/1215/1237/1265 |
| R9X02 | P1 | Attention + QSA/indexer | 1300, 1301 | 1201/1202/1203/1266/1270/1271/1280/1292 |
| R9X06 | P1 | Expert residency/LRU + pinned-host backing | 1307 | 1279 + memory/lifetime work |
| R9X07 | P1 | WHT/compressed collective experiment | 1308 | 1001/1244/1250/1252/1272/1275/1276/1277/1291 |
| R9X11 | P1 correctness | Collective fixed-grid/replay hardening | 1311 | 1252/1275/1276/1291 |
| R9X08 | P2 | PLE, QSA-specific norm+RoPE, transpose microfusions | 1309 | 1004 + qwen4exp model path |
| R9X05 | P3 / format-gated | MXFP4/FP8 MoE + skinny GEMM | 1305, 1306 | 1203/1237/1241/1262/1265/1267/1273/1274 |
| R9X09 | final | Cross-generation integration/promotion | none by default | all above |

The priority order deliberately puts launch-fusion work ahead of generic FP8/MXFP4 format work: BigCherry profiling and r9700-stack independently indicate Flash-Next decode has substantial launch/dispatch overhead, while the format-specific kernels need a deployable GGUF/storage path before they are useful.

## Source inventory to pin in evidence

r9700-stack reference kernels: `kernels/r9k_attn.hip`, `r9k_qsa.hip`, `r9k_qsa_score.hip`, `r9k_gdn.hip`, `r9k_router.hip`, `r9k_hc.hip`, `r9k_moe_mxfp4a8.hip`, `r9k_gemm_fp8.hip`, `r9k_ple.hip`, `r9k_norm_rope.hip`, `r9k_transpose.hip`, `r9k_ar.hip`, `r9k_ar_wht.hip`, `r9k_ar4.hip`, and `kernels/third_party/davetha/r4d_lru.hip`.

Host/reference adapters/tests include `r9700_vllm/attn/qsa.py`, `r9700_vllm/kernels/attn.py`, `r9700_vllm/models/gdn.py`, `r9700_vllm/kernels/moe.py`, `r9700_vllm/moe/shared.py`, `r9700_vllm/router.py`, `r9700_vllm/hc.py`, `r9700_vllm/moe/cache.py`, `r9700_vllm/comm/r9k_ar.py`, `r9700_vllm/comm/r4d_ar.py`, `r9700_vllm/kernels/ple.py`, `r9700_vllm/ple/int6.py`, `r9700_vllm/ple/short_conv.py`, `tests/test_shared_expert_r9k.py`, `tests/test_ar_wht.py`, and `tests/test_ar_race.py`.

## Patch allocation discipline

`1300`–`1312` are reservations, not ownership locks. Before implementation, re-read branch head and the maximum numeric directory under `patches/`. If another agent has consumed a slot, preserve the R9X plan ID but allocate the next free patch number and update this README plus the corresponding plan. Every created package must follow current BigCherry package conventions (`patch.py`, `patch.toml`, `SUMMARY.md`, validation contract/evidence where required) and carry its R9X plan item ID.

## Licensing/provenance

`r9700-stack` is Apache-2.0; the vendored davetha LRU kernel is also Apache-2.0 with its own attribution. Any adapted source must preserve SPDX/copyright/NOTICE obligations and record source commit/path in BigCherry provenance. Do not copy unrelated carve-out material mentioned by the source repository's `NOTICE`. Concept-only reimplementations must say so explicitly rather than claiming a verbatim port.
