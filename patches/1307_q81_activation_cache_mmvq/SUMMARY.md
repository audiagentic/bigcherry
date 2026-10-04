# 1307_q81_activation_cache_mmvq

**Status:** validated
**Plan item:** PRBE05/QFP13

## What it does

Stage 2 of RD09: with `GGML_HIP_Q8_1_CACHE_MODE=on`, `ggml_cuda_mul_mat_vec_q` looks up the Q8_1 quantization of
src1 in 1235's generation-scoped cache (keyed by the consumed node, not its view root, so in-place rewrites never alias) and skips the `quantize_q8_1` launch on a hit; a miss quantizes into a
stable cache slab and publishes it; any reserve failure falls back to the original pool allocation and quantizer.
A cache generation begins at every `ggml_backend_cuda_graph_compute`, and slab growth is blocked while a HIP graph
is captured, so captured graphs reference only never-relocated memory. Motivation (QFP13): decode is launch-gap
bound and issues ~183 quantize launches per generated token per GPU, one per MMVQ consumer, many re-quantizing the
same activation. Cache statistics (1235) report hits and launches saved.

## Hardware result (2026-10-04 review)

Profile v3 adoption ABBA (flashnext-v2-fusion-ab-3, with 1308-1310): ~10K 74.3 -> 76.5 t/s (+3%), ~80K 53.3 -> 55.2 (+3.6%), complete separation, greedy identical; quantize_q8_1 183 -> 45/token with the producers.
