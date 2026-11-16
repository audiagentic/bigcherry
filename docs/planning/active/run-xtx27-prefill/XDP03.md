---
id: XDP03
order: 2
plan: run-xtx27-prefill
state: in_progress
created-at: '2026-11-09T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Dense 27B heterogeneous quantization and GPU-aware layer placement

## Description

Owner direction (2026-11-09): re-quantize the **original BF16 tensors**
into formats useful on gfx1100 RX 7900 XTX and gfx1201 R9700, allow
different layers and projections to use different compatible quantization
methods, then select an appropriate stored variant according to the GPU
where the layer executes. This is a **joint prefill-speed, quality, and
resident-KV-capacity** objective under XDP01/XDP02, not a stand-alone
weight-compression exercise.

**Available source**: the actual Qwen3.8-27B BF16 checkpoint is
the local BF16 checkpoint (`${BF16_CHECKPOINT}`), with config and 18 safetensors
shards (~52 GB). No re-download is needed. Use BF16 directly rather than
dequantizing the present Quark MXFP4 container (rounding would compound).
The current 27B geometry is 64 dense layers, with 16 full-attention
layers; its declared context is 262,144, **not 1M**.

## Technical findings / non-goals

- Radiance `rad-convert` accepts **per-logical-weight** first-match recipe
  patterns and emits each weight's canonical encoding. It checks encoding
  against the resolved kernel's layout hook; `--plan-only` is the
  engine-level preflight. Canonical models are **not tied to a kernel
  library**, but each logical weight is stored **once** per container.
- `arch/common/rad_fp8.h` `LinearFP8` supports MXFP4, block FP8,
  plain/rotated INT8, and unsigned W4 with a per-group zero and scale.
  W4 uses `w4a16_asym` at skinny rows and `w4a8i8_asym` at prefill.
  This **does not prove native gfx1100 IU4 WMMA**; the latter may require
  a new kernel. `qwen35_w8a8` also has global activation-mode rules:
  not every arbitrary INT8/MXFP4 mixture is valid.
- `libr3/native/r4d_gemm_8a8_prefill.h.rw` accelerates INT8 at an
  isolated shape, but the measured W8A8 27B trunk was slower end-to-end
  than MXFP4 (~1,940 vs ~2,790 prefill tok/s at 4.5K on an older pin).
  **Do not presume an 8-bit conversion improves speed**; it costs KV
  headroom as well.
- Existing layer movement to an R9700 full-transformer stage is not
  implemented in Radiance: remote-MoE helpers are not dense stages.
  A different quant per GPU at **load/placement time** is feasible
  as a design; **on-the-fly same-layer re-quantization or migration
  while serving** is not supported and not proposed as a v1 goal.
- A pair of XTX tensor-parallel ranks must use the **same logical
  representation** for each shared layer; do not choose a different
  quant independently per rank without protocol/numerical proof.
  Do not duplicate unused variants in VRAM when maximizing resident KV.

## Bounded implementation (this worktree)

`tools/lab/radiance/xtx27_quant_plan.py` is a **card-free**, fail-closed
recipe generator using the local BF16 config. It emits two candidates:

1. `baseline`: all eight trunk projection families MXFP4
   (`fp4_e2m1 / e8m0 [1x32]`), block-FP8 output head, all unspecified
   BF16 planes intact.
2. `mixed`: the two FFN projections in XTX-prefix layers
   `[preserve_prefix, cut)` use `u4 zero=u8 group=128 scale=f16
   rule=search` (recognised by `LinearFP8::linear_is_w4`).
   All other trunk projections remain MXFP4; optional suffix FFNs use
   block FP8 on the proposed R9700 layers `[cut,64)`. First eight
   layers default to MXFP4 to reduce unproven quantization risk.
   This **is not** a claim that those layers execute on R9700.

`--write` generates a recipe and provenance manifest in isolated scratch,
including the planned exact `rad-convert --plan-only` argv. The default
is **read-only dry-run**. Neither mode launches a converter or touches
a GPU. Conversion, kernel plan validation, actual placement, KL, and
speedups are **not yet done**.

The companion card-free `xtx27_quant_select.py` accepts **complete,
provenance-bound per-layer metrics** for each GPU stage (one metric row
for every layer), physical VRAM and runtime reserve. It computes exact
logical FP8 attention KV by stage at the requested native context,
then runs bounded dynamic programming over conservative rounded-up
MiB costs to select the minimum measured prefill/decode time under the
per-stage weight budgets. Its results are ALWAYS marked
`PREVIEW_ONLY_NOT_END_TO_END_QUALIFIED`. Synthetic inputs are permitted
only for tests; selection-to-recipe promotion requires **measured**
evidence, a matching checkpoint config digest and complete layer
coverage. The selector is **not a serving admission plugin**. No vendor source changes.

## Steps

1. **Current checkpoint and encoding inventory**: confirm exact BF16
   source digest, model tensor names, candidate GPU capabilities, native
   W4/INT8/MXFP4 kernel row at the projection geometry, and current
   kernel library provenance. Parse the model's actual layer types.
2. **Mixed recipe and matched control**: generate reproducible
   baseline and mixed recipes from the **same** BF16 source; record
   experiment manifest, `cut`, quant ranges and all fallback formats.
   Start with `cut=40`, `preserve=8`; test 32/40/48 later. The
   optional R9700 FP8 suffix trades more weight memory for quality/speed.
3. **Engine preflight under exclusive lease**: run `rad-convert
   --plan-only` against the pinned, production `libr3` / `libr4d`
   library stack. It may initialize device plugins, so **do not run
   during someone else's GPU lease**. Fail on any missing producer's
   BF16 plane, mismatched activation dtype, GEMM/kernel row, TP
   alignment or layout, or architecture format rejection.
4. **Quality, kernel and memory tests**: once plan checks pass, create
   matched BF16->MXFP4 control and mixed `.rad` outputs, capture
   `rad-info -v`, tensor-wise quant error, kernel oracle, KLD/top-1
   vs BF16 and control, and head/token agreement. W4 is a *stored
   encoding*; profile the **actual** gfx1100 W4 GEMM instructions and
   host quantization overhead before calling it native-fast.
5. **Benchmark as a joint objective**: on the same TP2 dual-XTX
   pinned engine compare cold/warm TTFT, 2K-32K prefill, decode
   ms/token, concurrency, VRAM **weights vs KV vs activations**, and
   output quality; no unpaired one-shot winner. Repeat on the R9700
   with each R9700-relevant suffix encoding in its **existing supported
   single-card geometry**, before writing stage code.
6. **Static GPU-aware variant selection (design)**: model a mapping
   `layer range -> stage group -> device type -> preferred quant
   variant`. Persist a provenance/key for each immutable variant.
   Resolve profiles at **model admission / restart**, not every token.
   Validate stage memory (XDP02), performance and a quality ceiling
   with joint optimization and an explicit fallback to qualified MXFP4.
7. **Runtime integration (not yet implemented)**: decide whether to
   support multiple `.rad` files, per-layer sidecar variants or
   container extension. Each logical weight currently exists only once
   in a container; a true multi-variant loader needs lookup, GPU
   constraints, residency accounting, variant cache invalidation and
   fail-closed error paths. A full R9700 layer stage needs transfer,
   synchronization, per-stage KV/GDN ownership and continuation
   correctness **before** any dynamic placement. Live mid-request
   migration is a separate, likely costly experiment.

## Reproduction (no GPU)

```sh
python3 tools/lab/radiance/xtx27_quant_plan.py \
  --checkpoint ${BF16_CHECKPOINT} \
  --output-dir ${BC_WORK_ROOT}/xtx27-prefill/quant-lab \
  --profile mixed --cut 40 --preserve-prefix 8 --write

python3 tools/lab/radiance/xtx27_quant_plan.py \
  --checkpoint ${BF16_CHECKPOINT} \
  --output-dir ${BC_WORK_ROOT}/xtx27-prefill/quant-lab \
  --profile baseline --cut 40 --preserve-prefix 8 --write

python3 -m unittest discover -s tools/tests/lab -p 'test_xtx27