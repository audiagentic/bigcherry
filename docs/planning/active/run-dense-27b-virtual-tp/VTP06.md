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

## 2026-11-10: real converter preflight and existing native W8A8 inventory

**No conversion required to get first Q8 test artifact.** Original
`rad-convert --plan-only -v` with BF16 source, pinned
`data/recipes/q38-27b-w8a8-df2.recipe`,
`--set quantization_config.quant_method=w8a8`,
`--set quantization_config.fmt=fwht128`, `--kernels libr3,libref`,
and pinned build's `--home` succeeded with no GPU or write:

| Source combination | Planned weights | Recipe quantized | Kept | Converter plan size |
| --- | ---: | ---: | ---: | ---: |
| BF16 27B, no external draft | 1038 | 295 | 743 | 27.94 GiB |
| BF16 27B + local DFlash2 FP8 | 1101 | 321 | 780 | 29.03 GiB |

Plan logs and commands:
`${BC_WORK_ROOT}/xtx27-prefill/quant-shape/w8-plan-probe/{receipt.json,stderr.log}`
and `w8-plan-probe-draft`.
The DFlash2 plan reports 15 source MTP tensors not declared or
written; verify drafter/MTP choice against quality before production.

**Existing containers**, not newly generated:
`${BC_MODEL_ROOT}/radiance/bigcherry/qwen3.8-27b-w8a8/`

| Existing RAD2 file | Exact bytes | Actual size | RAD2 directory tensors | First 8 MiB SHA-256 |
| --- | ---: | ---: | ---: | --- |
| `qwen3.8-27b-w8a8.rad` | 29524840448 | 27.50 GiB | 1024 | `9224b7e0bdb29085d33b548538c6f8f1a6d544c1b8b16d85aa13067df7fff590` |
| `qwen3.8-27b-w8a8-df2.rad` | 31180201984 | 29.04 GiB | 1101 | `e98ccbfbdd62342687d609bf07d0d875cc10ff89c4673e75744a0d12cdf81e91` |

Both open through pinned `rad-info` and report
`qwen35 / w8a8_fwht128`, geometry 64/5120/17408 and
262144 native context. Each embedded recipe has 17 normalized
rules and matches the currently pinned source recipe text.
Source recipe SHA-256:
`09725afb09d1816380c4128332ebf0eda8488e6018f9248843514a95d826d9ed`.
RAD2 headers identify source name and original November 4 conversion,
**not a full content hash binding to the BF16 source**. First-8-MiB
hashes fingerprint the headers only; full payload integrity is
not yet verified.

Archived historical baseline at
`docs/archive/radiance/qwen3.8-27b-pre-v117-reference.md`:
dual XTX, W8A8+DFlash2, FP8 KV, measured **205.6 aggregate tok/s
at 4 streams, 201.4 at 6** (2026-11-04 single run per row). Do not
present these as current three-physical-GPU performance or rigorous p95.

The same plan restricted to `--kernels libr4d,libref` initially
failed declaring the architecture because Radiance selected
**physical GPU0 gfx1100** and correctly excluded the gfx1201-only
libr4d device code. This is a *device selection mismatch*, not
evidence that R9700 W8A8 is broken. A gfx1201-only plan was subsequently **completed successfully** under
the exclusive GPU lease, with GPU2 isolated by
`ROCR_VISIBLE_DEVICES=2` and `--kernels libr4d,libref`.
The converter reported one visible `gfx1201` device,
**1038 planned weights (295 quantised, 743 retained)**,
and **27.94 GiB** estimated output, exit code zero.
Receipt/log: `${BC_WORK_ROOT}/xtx27-prefill/quant-shape/w8-plan-probe-r9700-v2/`.
A second exclusive-leased gfx1201-only plan **with the local DFlash2
drafter** also succeeded: 1101 weights (321 quantised, 780 kept),
29.03 GiB estimate, one visible gfx1201 device, exit code zero.
Receipt/log: `${BC_WORK_ROOT}/xtx27-prefill/quant-shape/w8-plan-probe-r9700-draft/`.
Both R9700 checks are plan-only and wrote no converted model.
This validates the R9700 *graph/encoding plan only*, not serving or
mathematical quality. The GPU lease was released and no KFD processes remained.

**IQ6R-V4** needs matching RADGRAM2 calibration Hessians, plus
the prepared vocabulary-head classes (the pinned
`data/vhead/q38-27b-vhead-5.0.cls` exists) and GPTQ producer.
No matching model calibration corpus was confirmed in the inspected
local locations. Do not run an uncalibrated, unqualified 6-bit
conversion or replace IQ6R with GGUF Q6_K.
