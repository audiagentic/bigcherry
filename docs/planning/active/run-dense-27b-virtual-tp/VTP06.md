---
id: VTP06
order: 5
plan: run-dense-27b-virtual-tp
state: in_progress
created-at: '2026-11-10T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Q6/Q8 model and four-plus-stream quality/capacity qualification

> Archived 2026-11-12 from branch `exp/xtx27-prefill-opt` (git tag `archive/exp-xtx27-prefill-opt`), which never merged. Its lab programs (tools/lab/radiance/xtx27_*), tests and early patch drafts are in that tag. Superseded by VTP14 (native Radiance TP3, completed 2026-11-12), which kept these programs as reference oracles only.

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

## Full original BF16 payload fingerprint: completed

Full, bounded-memory SHA-256 across all **18** original safetensors
shards completed twice, independently. Each of the 18 per-file
full-payload digests matched between receipts, covering exactly
55,562,855,904 tensor bytes plus safetensors headers. The corrected
immutable audit receipt is at
`${BC_WORK_ROOT}/xtx27-prefill/quant-shape/bf16-audit-fullsha-20261110-v2.json`
with receipt SHA-256
`28ae5dd7678a07d9558a7328d78c1cc1b19c7f45ffe6011fb6776ddc6aeeed13`.
Status: `STRUCTURALLY_VALID_FULL_SHARDS_SHA256_FINGERPRINTED`;
payload integrity: `FULL_SHARD_SHA256_FINGERPRINTED_NOT_EXTERNALLY_VERIFIED`.
Original earlier receipt had all hashes but an inconsistent header-only
status string, now corrected in code/tests; earlier receipt remains
untouched. Full hashes are identities, not publisher authenticity or
per-layer/model numerical quality proof. Bind the v2 receipt digest
and 18 shard hashes into future conversion/quality campaigns.

## IQ6R calibration input contract (not executed)

Pinned Radiance `docs/TOOLS.md` calibration mode accepts
`radiance --model BF16.rad --calib-corpus CORPUS.jsonl --calib-dir DIR --calib-layers LO:HI`.
It records whole-input `hessian_upper` (f32) safetensors, even when
TP shards input columns, with metadata k, rows and linears.
The same model, tokenizer, prompt corpus and precision must be used
for IQ6R GPTQ; a reused Q4/MXFP4 calibration is not qualified.
Upstream warns each full dense 27B layer costs ~786 MB of host Gram
accumulator, so collect limited adjacent layer ranges and ensure the
final layer range captures head input. Do not run calibration until a
BF16 model-serving configuration and GPU lease are available.

## Fully fingerprinted pre-existing W8A8 containers

Both existing RAD2 files were streamed end to end for SHA-256.
These are **whole-file** hashes (not earlier 8-MiB header hashes):

- `qwen3.8-27b-w8a8.rad` (29,524,840,448 bytes):
  `6bbf6e4a3dbf5634babdd5659b27336a11c372cdfa641802f2d451fb3233d619`.
- `qwen3.8-27b-w8a8-df2.rad` (31,180,201,984 bytes):
  `5a9af2dfaf3150e59cfc0001f8fe4f33549b201c6be19700c09c9e9209cc2701`.

Pinned `rad-info -v` reports the DFlash2 container encoding
breakdown (display-rounded, not VRAM allocations):

| Component | RAD2 entries | Approx GiB |
| --- | ---: | ---: |
| Fused FFN gate/up and down | 128 | 16.187 |
| Embedding and output head | 2 | 3.570 |
| Other text, attention and GDN | 561 | 6.873 |
| DFlash2 drafter | 77 | 1.542 |
| Vision tower | 333 | 0.858 |
| **Total** | **1101** | **29.029** |

Only the FFN class corresponds directly to VTP05 row-specific GEMM
split experiments; the other operators can have different rank
legality, replication and allocator costs. The Q8 file is exact-id
fingerprinted, but that alone does not prove it was converted from
the new fully hashed source (the RAD2 contains source name, not
source full-shard digests); compare inference logits to original BF16.

## VTP11 calibration plumbing milestone (2026-11-10)

Original-BF16 layer-0 Hessian generation now passes on R9700 TP1 and
dual-XTX TP2 (both 2 documents / 185 rows, four F32 safetensors
matrices with finite values and nonnegative diagonals). See VTP11.
**Pipeline unblocked; production IQ6R quality not qualified.**

Next work under existing VTP06 (no separate duplicate plan):
1. Choose and license-check a diverse, reproducible corpus: language,
   code, maths, variable lengths and realistic deployment prompts.
   Capture text provenance, deduplication, filtering, tokenizer
   fingerprint, seed, token histogram, held-out subset and SHA-256.
   This must not be the 185-row micro-corpus.
2. Disk/RAM preflight for all 64 layers of full upper-triangular F32
   Hessians. Run bounded adjacent 2-4 layer ranges with resumable
   per-layer outputs and reject gaps, duplicate outputs, drifted hashes
   or invalid shapes.
3. Independently compare gfx11/gfx12 GDN GPU kernels against a trusted
   BF16 reference before treating Hessians as numerically qualified.
4. Calibrate and convert IQ6R using the exact pinned producer, verify
   per-tensor encodings and actual gfx1100/gfx1201 graph dispatch.
5. Compare logits, top1, KLD and held-out perplexity to original BF16,
   then benchmark 4+ concurrent streams with resident KV (VTP03).

## 2026-11-10 GDN seeded per-op reference gate

VTP11's repaired kbench checked the actual BF16 h128/w4/chunk64
gdn_conv_prep GPU row against libref on **R9700, XTX0 and XTX1**.
With an independent seed, M=1..512 and 0.001 relative tolerance,
**80/80 cases per card** passed, 0 skips and no guarded-buffer
overwrites. See VTP11 and
BC_WORK_ROOT/xtx27-prefill/vtp11-gdn-kbench-20261110-v5-m512-strict/validation.json.
These are seeded GDN operands drawn using pre-existing W8A8 RAD2
geometry. Warm convolution history, original-BF16 full-model logits
and representative 64-layer Q6 quality are still unqualified.


## 2026-11-11 — card-free calibration and IQ6R quality workflow

**Implementation state: tools and a synthetic-only pilot are staged; no
representative all-layer run, IQ6R conversion or quality approval.**
This remains under VTP06 in the existing worktree. Production is unchanged.

The card-free tools added in tools/lab/radiance are:

- xtx27_calib_corpus.py: audit, prepare and verify. Requires source, licence,
  category and split for each JSONL document; deduplicates text across
  calibration and holdout, preserves native score_from on holdout only,
  records exact tokenizer/corpus/source SHA-256 and refuses drift. Candidate
  mode has structural minima of 200 calibration / 50 holdout documents,
  three source identities and five categories. These counts are neither
  a representative native-token dataset nor legal verification.
- xtx27_calib_ranges.py: immutable 64-layer plan, explicit 1–4 layer runs,
  unique attempt numbers, per-range verification, status --full-verify.
  Binds the exact previously fingerprinted original BF16 checkpoint,
  experimental source/build binary, tokenizer and corpus. A run requires
  its own exclusive GPU lease, no competing KFD PIDs and fresh memory/disk
  admission; its monitor stops only its own process if RAM falls or the
  lease disappears. It never loops over 64 layers automatically.
  The verifier streams entire F32 packed-upper Hessians and rejects
  nonfinite values, missing/overlapping layers, incorrect row counts,
  preemption and unexpected files. Exactly four declared inputs per
  layer are required, with distinct full-attention/GDN names, plus
  lm_head.input in the last range ending at 64.
- xtx27_iq6r_quality.py: offline validation of native Radiance's BF16
  KLD reference and IQ6R report with the .rows F32 position records,
  independent frozen holdout, native token offsets/sources and
  proposed numerical thresholds. Checks KL mean/p99/p99.9, top1,
  perplexity ratio and source-level coverage. Uses sparse top-128
  plus tail, which is a lower bound on true full-vocabulary KL.
  The companion policy JSON is explicitly PROPOSED_NOT_APPROVED.
  Even if metrics pass, the report never self-promotes to quality
  or production approval.

The tests are in tools/tests/lab/test_xtx27_calib_pipeline.py and
test_xtx27_iq6r_quality.py and are card-free; negative cases check
source/holdout leakage, missing model inputs and final head, memory
and lease fail-closed behavior, file checksum drift and KLD provenance.

**Real pilot:** the earlier successful two-document 185-token smoke
corpus was frozen under
${BC_WORK_ROOT}/xtx27-prefill/VTP06-pilot-two-doc-20261111/.
It has zero holdout documents, 32 planned two-layer ranges and
zero newly completed calibration layers. The isolated VTP11
successful engine build and BF16 full-shard fingerprint are pinned.
Observed layer-0 Hessians total about 0.75 GiB, extrapolated to
46.88 GiB across 64 layers (EXCLUDES final head and variation);
about 695 GiB filesystem headroom was observed. Host free memory
has varied widely and swap is almost exhausted during concurrent
measurements, so every run requires fresh admission. Another
BigCherry session held the exclusive GPU lease; no GPU run was
started or interfered with.

**Remaining gates:** source/licence-reviewed representative dataset,
native tokenizer token counts and token-length histogram, sufficient
and genuinely disjoint held-out eval, one fresh 0:2 or 2:4
candidate-range under an owned GPU lease, remaining 64-layer
resumable validated Hessians, native IQ6R conversion and encoding
checks, original BF16-vs-IQ6R KLD/top1/PPL and serving/KV/p95
quality. The two-document pilot must never be fed to a production
converter to claim calibrated quality. See tools/lab/radiance/README.md
for explicit commands and restart rules.

## 2026-11-11 — hardware qualification of VTP06 range runner (pipeline only)

Two bounded two-layer runs were executed on original fully fingerprinted
Qwen3.8-27B BF16, on the dual RX 7900 XTX (gfx1100) using TP2,
exact TP wire, P2P off, 16,384 MiB host pool, BF16 KV, max context 512,
batch 128 and layer_offload. Each run had its own exclusive GPU lease
and no other KFD processes. Leases were explicitly released; the
production serving configuration was never changed.

Exact immutable campaign manifest and raw command/exit/memory/hash
receipts are under:

  BC_WORK_ROOT/xtx27-prefill/VTP06-pilot-two-doc-20261111/campaign/

The corpus is the *same two-document smoke-only* 185-token input as VTP10:
the bundle's metadata says PILOT_ONLY_NOT_QUALITY_QUALIFIED, and it has
zero holdout examples. It cannot be used for representative GPTQ.

| Calibration range | Exit | Documents / tokens | Hessians | Integrity | Output |
| --- | ---: | --- | ---: | --- | ---: |
| 0:2 (GDN layers 0,1) | 0 | 2 / 185 | 8 | PASS | ~1.5 GiB |
| 2:4 (GDN 2, full-attention 3) | 0 | 2 / 185 | 8 | PASS | ~1.5 GiB |

Both jobs reported **zero scheduler preemptions**. All 16
full-width packed-F32 Hessians passed whole-file scans for expected
schema, exactly four taps/layer, positive diagonal, no negative
diagonal, no NaN or Inf, nonzero content, exact row counts and
SHA-256. In particular layer 3 emitted the full-attention
self_attn.o_proj.input tap, whereas the GDN layers emitted the
linear_attn.out_proj.input tap. The independent campaign
status --full-verify command rehashed every one of the 16 files and
reported output_bytes_reverified=true, validated_layers=4, total_layers=64.

The layer-0 four-output SHA-256 comparison against the earlier
independently executed VTP11 dual-XTX TP2 0:1 smoke test yielded
**4/4 byte-identical results**, even though this run recorded two
layers in one calibration range. This is a useful deterministic
replay check, not an independent BF16 mathematical reference.

**Verdict: VTP06's resumable range pipeline is real-hardware validated
for both GDN and full-attention layer layouts, and 4/64 layers are
pilot-verified.** It is NOT calibrated IQ6R. There is no approved
diverse corpus, native-token histogram, 64-layer Hessian set,
converter output or BF16-vs-IQ6R KLD result yet. The first range of
any *representative* campaign must be separately rerun from scratch
with its own exact provenance and lease; pilot Hessians do not carry
over to it.

Full receipts:
- 0:2: campaign/ranges/00-02/attempt-01/{receipt.json,validation.json,engine.log,hessians/}
- 2:4: campaign/ranges/02-04/attempt-01/{receipt.json,validation.json,engine.log,hessians/}
