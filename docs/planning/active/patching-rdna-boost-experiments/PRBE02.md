---
id: PRBE02
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:53:35.078709+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: S
priority: P3
---

# RD05 WMMA FlashAttention: upstream k00 barrier; residual tile boundary gate

## Authoritative disposition — 2026-10-11, b11552

**Pending, correctness-only; no patch or hardware queue authorised.** Merged upstream llama.cpp [#30103](https://github.com/ggml-org/llama.cpp/pull/30103), commit `1bb2b9fcbe85c35c4e3b787e1c868ef05ca702f3` (2026-10-10), repairs the **first** of the two barriers extracted into BigCherry `1266_rd05_wmma_fa_tileq_sync`. The upstream issue [#29314](https://github.com/ggml-org/llama.cpp/issues/29314) was closed on that merge. The `1266` patch still anchors on the old `if (np > 1)` block and cannot be treated as an applicable b11552 patch without reconciliation. Its **second** inter-`process_tile` barrier is not present in b11552. There is no demonstrated residual stock race at that boundary. Do not port a redundant first barrier, create a new attention dispatcher, or reuse the rejected `1203` evidence.

This section **supersedes all older steps, package-blocked claims, and historical head-size campaigns** retained in Git history. PRBE110 owns the archived `1266` identity and RD07/1267; PRBE02 owns only the bounded *residual kbc-boundary correctness decision*. PRBE110 and current b11552 MMQ/pin work are read-only dependencies.

## Source and dataflow

- `ggml/src/ggml-cuda/fattn.cu::ggml_cuda_get_best_fattn_kernel`: on AMD WMMA, `gqa_opt_applies` and `Q->ne[0] <= 256` plus effective query/GQA width thresholds admit `BEST_FATTN_KERNEL_MMA_F16`. A model's existence or a generic FA flag does not prove this route. Head sizes 320/512/576 are **negative WMMA-dispatch controls** on gfx1100/gfx1201, not positive RD05 qualification lanes. The MFMA branch is distinct; do not conflate CDNA with RDNA.
- `ggml/src/ggml-cuda/fattn-mma-f16.cuh::flash_attn_ext_f16_process_tile`: `VKQ_C` is written to shared `tile_Q`, `__syncthreads()` precedes consumer reads, and the old end-of-`k00` guard was `if (np > 1)`. BigCherry `1266/patch.py::_K00_SYNC_NEW` adds `k00 + nbatch_combine < DV/2`. Upstream b11552 instead uses `if (np > 1 || nbatch_combine != DKQ/2)`; the upstream guard is **not textually equivalent** to 1266, especially when DKQ != DV. Do not stack them.
- The same file's outer `kbc` loop calls `flash_attn_ext_f16_process_tile` and advances `kbc += iter_k` without an explicit intervening barrier. BigCherry `1266/patch.py::_KBC_ANCHOR_NEW` inserts one. A residual hazard is *possible* only if a subsequent `process_tile` reuses shared storage before all consumers finish and the inner end barrier did not already synchronize; it is **not established** by source inspection.
- `1266/validation/producer.py::run` compares full-vocabulary outputs for a single `tierA-qwen4b-q6k` prompt and emits a once-per-process `BEST_FATTN_KERNEL_MMA_F16` dispatch marker. Neither the marker nor perplexity/full-vocab parity proves which `k00`/outer-`kbc` loop instance executed or that a race occurred. The `test_1266_rd05_wmma_fa_tileq_sync.py` fixtures check text application/idempotence/anchor failure, not GPU barrier semantics.

## Cheapest discriminator before hardware

Source-static fixture against the **exact b11552** `fattn-mma-f16.cuh` and `fattn.cu` must prove: upstream `np > 1 || nbatch_combine != DKQ/2` present; legacy anchor absent; `kbc` boundary lacks an independent barrier; and WMMA head gate `<=256`. Parsed RDNA config table has 38 entries; 10 have more than one `k00` combine chunk (256/ncols32,64; 512 and 576/ncols8,16,32,64). The latter 8 are **not** RDNA WMMA-positive dispatch lanes under current `fattn.cu`. This is a source-derived necessary-condition census, **not** an executed kernel count, race proof, or GPU measurement.

If a real stock failure is independently reproducible after b11552 pin transition, use existing `test-backend-ops -o FLASH_ATTN_EXT` / full-vocab infrastructure with `gfx1201` primary and `gfx1100` control. Force actual F16 WMMA dispatch for head 192 and 256 at q_rows 1/2/4/8/16/32, GQA 1/4/8 where admitted, KV lengths spanning 1 and multiple `kbc` tiles, mask on/off, graph on/off, and same-process repeated requests. Record per-case DKQ/DV, ncols1/2, `np`, `nbatch_combine`, `k00` iterations, `kbc` iterations, marker *per case*, and the first divergent output/logit/state. Head 320/512/576 and TILE/VEC are negative controls. Preserve the stock b11552 control and any independent upstream fix. Do not confuse a known #28867 WMMA **performance-routing** regression with a barrier correctness failure.

## Conditional implementation / ownership

Only after a reproducible, source-correlated **post-#30103** kbc-boundary race: isolate the smallest remaining inter-`process_tile` barrier using the existing 1266 source and tests as a reference; do **not** reapply the obsolete k00 replacement. Gate by actual loop shape and architecture only where proved safe; preserve shared `tile_Q` lifetime, warp completion, graph capture/replay, stream ordering, K/V staging and MTP acceptance. Use a fresh attested patch identity/contract if semantics change, rather than inheriting `1266` receipts. Do not modify PRBE110/1267 or current MMQ/pin files under active ownership.

Baseline/control: exact same b11552 binary/quant/model/ubatch/context, with and without the minimal fix; full-vocabulary logits/KLD, deterministic greedy, multi-ubatch, multi-request same-process, graph replay, sanitizer where supported, and no missing attention work. Record kernel launches, synchronization events and prefill/decode exposed wall. Only after correctness can a >=4-session, >=10 paired ABBA campaign be considered. A safety fix needs a reproduced race and no material regression; a performance claim separately requires CI95-low >=3% E2E and <=1% unaffected-control regression. No speculative speedup is expected from adding a barrier.

**Terminal gate:** if no residual post-#30103 race can be reproduced in a bounded targeted matrix, close PRBE02 as upstream-absorbed/no residual action and leave 1266 unpromoted; if reproduced, hand one exact failing shape and trace to the existing RD05 owner. No independent hardware campaign is queued by this audit.

## Existing measured evidence and limitations

Historical 1266 gfx1201: four sessions passed activation and full-vocabulary correctness, but target throughput point estimate **-0.0499%**, CI95-low **-0.1609%** (PRBE110); no stock race was reproduced. Rejected mixed 1203 PPL sigma=1.35 is provenance only. No b11552 HIP build, real graph trace, sanitizer run, GPU test or new benchmark was performed here.

## Upstream / alternative engine decision

- [llama.cpp #30103](https://github.com/ggml-org/llama.cpp/pull/30103): **adopt via b11552 pin**; source changed one k00 guard, not the outer kbc boundary. [#29314](https://github.com/ggml-org/llama.cpp/issues/29314): original gfx1201 backend-op failure, closed with #30103.
- [llama.cpp #28102](https://github.com/ggml-org/llama.cpp/pull/28102) and [#28867](https://github.com/ggml-org/llama.cpp/issues/28867): WMMA admission/performance controls, not proof of the residual barrier.
- [vLLM ROCm backend](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/rocm_attn.py): explicit architecture/head-size/shape capability gating is a useful *qualification pattern*, not a transplantable barrier or AMD speedup. No new engine mechanism is adopted.

## Provenance

PRBE02 supersedes RD05. Prior b11126/b11474 plan proposed a 192/256/320/512/576 hardware campaign and claimed PRBE110 extraction was not yet delivered; 1266 was subsequently extracted and tested, and #30103 changed the baseline. Historical steps are retained in Git history, not active instructions. BCOP136 records this disposition.
