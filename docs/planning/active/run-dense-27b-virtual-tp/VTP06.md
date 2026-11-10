---
id: VTP06
order: 5
plan: run-dense-27b-virtual-tp
state: active
created-at: '2026-11-10T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Q6/Q8 model and four-plus-stream quality/capacity qualification

Separate this work from VTP05 kernel-only screening and VTP04 isolated transport.

1. From the original, fingerprinted Qwen3.8-27B BF16 checkpoint, select a W8A8 baseline and calibrated IQ6R variants. Never equate GGUF Q8_0/Q6_K import support with a native Radiance Q8/IQ6R kernel. Do not re-quantise Q4 weights. Use converter plan-only and matching kernel ABI and producer activations first. IQ6R needs qualified calibration Hessians and the selected model recipe.
2. Build opt-in candidates in isolated scratch. Confirm tensor/split legality, scales, graph dispatch and gfx11/gfx12 kernels, numerical oracle and independent top1/KLD against BF16. Reject immediately on quality regressions or unsupported operator paths.
3. Audit per-physical-card weights, full-attention KV, recurrent/GDN state, embedding/head, activation/graph scratch, drafter and driver reserve at 1/4/6/8 streams. No silent host spill. Context tests 8K/32K/64K/128K/256K within the current checkpoint's native limit. One future 1M request requires a model that genuinely supports it.
4. Compare same-format controls: dual-XTX TP2, R9700 independent replica if it fits, TP2-XTX plus R9700 whole-layer stage when present, and only then virtual TP4 (VTP03). Match checkpoint, context, quant, KV precision, sampling, MTP, runtime pin and benchmark corpus.
5. Go/no-go based on four-plus-stream aggregate tokens/sec, p95 latency, resident context, safe memory admission, numerics and clean shutdown. No microkernel-only speedup authorizes production promotion.

Do not create another worktree. Keep vendor/radiance pinned and read-only; managed patches only. No model download unless the local BF16 checkpoint or required calibration evidence is truly absent. No hardware runs without the GPU lease.

VTP05 kernel screening is recorded; VTP01 production execution design remains pending.

## 2026-11-10: BF16 checkpoint provenance preflight (card-free)

Original source: `${BC_MODEL_ROOT}/qwen3.8-27b/Qwen3.8-27B`.
No download, tensor loading or GPU required.

Command: `python3 tools/lab/radiance/xtx27_checkpoint_audit.py --checkpoint "${BC_MODEL_ROOT}/qwen3.8-27b/Qwen3.8-27B" --receipt <new scratch path>`.

Saved receipt: `${BC_WORK_ROOT}/xtx27-prefill/quant-shape/bf16-audit-20261110-v3.json`
(SHA-256 `860f85044dc44127036997f41868c7688a76f15e14a42ce3b3bbc0019173ff34`).

**Verified:** 18 safetensors files, 1,199 BF16 tensors,
55,562,855,904 tensor payload bytes, header/index tensor-name equality,
contiguous offsets and correct shape × dtype lengths. All 192
gate/up/down projections match the 64-layer, 5120-hidden,
17408-intermediate source geometry. Native context **262144** tokens.
Hashes capture config, index, tokenizer, tokenizer config, chat template,
generation config, vocab and merges.

**This is metadata-only:** no full weight-payload bytes were hashed.
Before conversion and quality evaluation, use
`--fingerprint-payload` in a safe I/O window to stream complete
per-shard SHA-256 in bounded 4-MiB blocks into a **new** receipt.
That provides an identity fingerprint, not publisher verification.
Never represent the metadata receipt as full weight integrity.

The audit lives at `tools/lab/radiance/xtx27_checkpoint_audit.py`,
with eight card-free negative and positive tests at
`tools/tests/lab/test_xtx27_checkpoint_audit.py`.
119 total XTX27 card-free tests passed.

### Remaining gates

1. Canonical isolated **full gfx1100** rebuild of VTP07's IQR
   layout fix; scratch CMake configured all **107** units at
   `${BC_WORK_ROOT}/xtx27-prefill/iqr-layout-fix/gfx11-full`.
   Full build deliberately not started while another 4-GPU
   Flash-Next A/B run held the lease and host available RAM was low.
2. Full-shard SHA-256 identity and native quant producer/calibration
   provenance; original BF16 only, never a Q4 requantisation.
3. Convert native W8A8 and calibrated IQ6R candidates, then
   compare to the BF16 model with per-layer and full-run numerical
   checks, original-output KLD/top1/perplexity and attention/GDN parity.
4. Only after a real multi-card execution backend exists,
   qualify 8K..256K context and 1/4/6/8 streams, physical
   memory/KV budgets, collectives and clean shutdown.
   Synthetic IQ6R V4 kernels and GGUF Q6_K/Q8_0 are NOT a
   validated production quantized 27B checkpoint.

### Exact original BF16 tensor categories (source census)

| Source category | Tensors | BF16 bytes |
| --- | ---: | ---: |
| Text FFN gate/up/down | 192 | 34,225,520,640 |
| Other text (attention/GDN/norms, etc.) | 657 | 14,480,882,688 |
| Embedding + language head | 2 | 5,085,593,600 |
| Vision | 333 | 921,460,192 |
| MTP | 15 | 849,398,784 |

These sum exactly to the indexed 55,562,855,904 BF16 bytes.
Only the FFN category corresponds directly to the VTP05 measured FFN
shard geometry. The other categories do NOT imply whole-card replication
or straightforward proportional sharding; the actual Radiance
converter/operator ownership must determine physical allocation.
