---
id: BCOP136
order: 136
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-11T10:12:25+00:00'
created-by: agent
priority: P3
work: S
---

# PRBE02: absorb upstream RD05 k00 barrier; gate residual kbc race

## Discovery / disposition

Merged llama.cpp #30103 (2026-10-10, b11552) changes `fattn-mma-f16.cuh` end-of-k00 barrier to `np > 1 || nbatch_combine != DKQ/2`. It supersedes the first of patch 1266's two synchronization edits and makes its old anchor inapplicable. The second inter-`process_tile` barrier remains absent upstream, but no post-#30103 stock race has been demonstrated. The once-per-process WMMA marker and archived full-vocabulary parity do not prove the residual barrier executed. Keep PRBE02 pending as a **bounded correctness discriminator**, not a new optimisation campaign.

## Ownership / protected work

PRBE02 owns only the residual race decision; PRBE110/1266 owns historical RD05 patch identity, while PRBE110/1267 and current b11552 MMQ/pin work are protected and unchanged. Do not re-port the merged k00 barrier or add another dispatcher/allocator. PRBE02 last independent edit 2026-10-08 06:47 UTC; 1266's last path change was a mechanical 2026-10-09 relocation. No independent RD05 patch, PR, benchmark or queued experiment was found in the preceding 12 hours. Active b11552 0300/1006/1237/1265/1350 MMQ and 1346/1359 MTP, QFP41/1356, QFP48/49, QFP50 and QFP17/1330 were excluded.

## Bounded action / terminal gate

Use b11552 source-static selector/shape checks first. Only if a post-#30103 stock gfx1201 failure is reproduced in actual WMMA-F16, multiple-kbc, graph-on/off and repeated-request cases should one isolated residual-only barrier be tested against identical full-vocab/logits/KLD, MTP and replay controls. If no reproduced race, close PRBE02 as upstream-absorbed; do not queue hardware or rebase 1266 speculatively. No performance claim without measured E2E improvement and no correctness regression.

## Evidence

Pinned b11474/b11552 `ggml/src/ggml-cuda/{fattn-mma-f16.cuh,fattn.cu}`; 1266 `patch.py`, producer and tests; PRBE110 archived four-session -0.0499% target/CI95-low -0.1609%. 38 RDNA config rows parsed, 10 multi-k00 configurations; 6 source-static b11552 checks passed. No HIP build, GPU test, sanitizer, repository pytest or new throughput measurement. Sources: [#30103](https://github.com/ggml-org/llama.cpp/pull/30103), [#29314](https://github.com/ggml-org/llama.cpp/issues/29314), [#28102](https://github.com/ggml-org/llama.cpp/pull/28102), [#28867](https://github.com/ggml-org/llama.cpp/issues/28867), [vLLM ROCm](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/rocm_attn.py).
