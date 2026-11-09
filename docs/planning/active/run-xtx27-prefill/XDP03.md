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

python3 -m unittest discover -s tools/tests/lab -p 'test_xtx27_*.py' -v
ruff check --no-cache tools/lab/radiance/xtx27_quant_plan.py tools/tests/lab/test_xtx27_quant_plan.py
```

Before running any command from the generated manifest, verify and acquire
an **exclusive GPU lease**, use the correct pinned `rad-convert` binary
and `RADIANCE_HOME`, and maintain all outputs outside the checkout.
Never invoke the converter just because a recipe/manifest exists.

## GPU-aware offline selector demonstration

`xtx27_quant_select.py --config CHECKPOINT/config.json --metrics MEASURED.json --cut 40 --tokens 131072 --objective prefill` produces an unqualified preview. The metrics JSON must supply `data_kind`, an evidence ID, canonical JSON `source_config_sha256`, `native_context`, budgets with `vram_mib` and `runtime_reserve_mib` for `xtx` and `r9700`, and 64 ordered layer rows. Each layer has candidate arrays for both stages; each candidate declares `format` (`w4`, `mxfp4`, `fp8`), `prefill_us`, `decode_us` and `weights_mib_per_card`. All metrics must be comparable, measured at the intended geometry. No unknown numbers are silently inferred.

Given reviewed **measured** selection output, the generator accepts `--profile selected --selection-file SELECTION.json --cut 40` and emits one format per logical FFN weight plus the MXFP4 fallback for other projections. It rejects synthetic selections, mismatched checkpoint configuration, incomplete coverage and inconsistent stage ordering. This creates a future *static model-load candidate*, not live runtime re-quantization.

## Actual pinned-engine feasibility (2026-11-09)

An **exclusive GPU lease** and the pinned Radiance build's `rad-convert
--plan-only` with `qwen35_fp8`, `libr3,libref`, BF16 source,
`--max-spec 0`, and `--set quantization_config.quant_method=fp8` /
`fmt=e4m3` accepted both experimental recipes. Each planned 1,024
logical weights (289 recipe-quantized, 735 left unchanged, including the
BF16 vision weights); **nothing was converted**:

| Variant | Estimated container | Engine plan result |
| --- | ---: | --- |
| BF16-derived MXFP4 control | 16.50 GiB | PASS |
| Middle-prefix W4 FFN, MXFP4 remainder | 16.43 GiB | PASS |

The ~0.07 GiB saving is **not** a compelling capacity win. The
reference Quark MXFP4 container is a different source/recipe and is not
the matched comparison for this BF16-derived experiment.

A single RX 7900 XTX running the pinned `libr3` GPU plugin was also
tested with `r4d_selftest` on `N=17408, K=5120` projection shapes.
These are **kernel-only** timings without the real activation,
quantization, TP collective, GDN or serving graph costs:

| Token rows M | Existing W4 kernel | MXFP4 comparison | Interpretation |
| ---: | ---: | ---: | --- |
| 1 | W4A16 BF16-input best tuning 76.9 µs | gated GEMV 103.4 µs, decode 146.9 µs | W4 decode may have a narrow advantage; different variants and tuning vs selected runtime |
| 64 | W4A16 BF16-input best tuning 368.3 µs | tiled 308.7 µs | W4 is ~19% slower even at 64 rows |
| 512 | W4A8I8 tiled minimum 3226.6 µs | MXFP4 tiled 931.6 µs | W4 is **3.46x slower**; does not include separate int8 activation quantization |

Raw stdout evidence is in the isolated scratch `${BC_WORK_ROOT}/xtx27-prefill/quant-lab/` under
`plan-baseline.log`, `plan-mixed.log`, `w4-gfx11-*-perf.log`,
`mxfp4-gfx11-*-perf.log`. All probes completed successfully and the
GPU lease was released.

**Decision:** do **not** spend resources converting the full 27B to
this existing broad W4 format for prefill, nor treat it as an improved
KV-capacity profile. The GPU-compatible encoding is real, but the
current gfx11 GEMM kernel is dramatically slower at prefill. Focus
next on either a genuinely native gfx11 IU4 MMA with carefully
quantized activation plus numerical/KL proof, or MXFP4 GEMM / TP
all-reduce improvements already proven relevant in XDP01. The W4
decode-only hint remains a separate, quality-gated investigation, not
a runtime recommendation.

## Follow-on native gfx11 W4A4 experiment

See [XDP04](XDP04.md): an **isolated** gfx1100 signed-A4 x
unsigned-W4 native IU4 WMMA probe, including exact group128
asymmetric-zero correction, has compiled and passed six GPU
oracle cases. The untiled kernel is a correctness reference, **not**
a serving/prefill speed optimization. Synthetic outlier activation
error is substantial (relative L2 0.374 W4A4 vs 0.193 W4A8),
so real Qwen3.8 activation/KL qualification is mandatory before
using A4 for any layer. Broad W4 conversion remains deferred.

## Risks and promotion gates

- The W4 `u4+zero+scale` path may widen activations and do several
  WMMA operations, so faster **native IU4** execution is unproven.
- W4 quantization error at the FFNs may be much larger than MXFP4;
  targeted RTN may need GPTQ/Hessian calibration and selectively
  higher precision at sensitive layers. Early/late-layer protection is
  a hypothesis, not a quality measurement.
- Mixed activation consumers, operator fusions and TP2 gates can fail
  even when a weight is representable; engine plan and end-to-end
  service/KL are mandatory.
- Different suffix FP8 weights use more VRAM; account for impact on
  **resident** FP8 KV before accepting a speed improvement.
- GPU-aware `cut` is declarative until XDP02 builds a real dense stage;
  the manifest must not claim the R9700 is executing layers.
- Never edit `vendor/radiance` directly; all code changes via managed
  components/patches, in this existing worktree; no push/PR.

## Acceptance Criteria

A measured, quality-gated mixed-quantization candidate from the same
BF16 source as its matched control, demonstrating a net speed/capacity
benefit on the actual two-XTX workload, **or** a documented negative
result. A separate staged design must prove safe GPU-specific variant
selection, memory admission, and correctly owned state before claiming
dynamic hardware-specific execution.

## Change Log

- 2026-11-09: Pinned-engine plan-only accepts both recipes; isolated W4
  prefill kernel is 3.46x slower than MXFP4 at M512. Broad W4
  conversion held pending a genuinely faster native gfx11 kernel.
- 2026-11-09: BF16 source confirmed local; generator, per-weight W4/MXFP4
  with optional FP8 suffix, dry-run provenance and card-free tests added. A measured-input-constrained, card-free
  quantization selector and selected-profile recipe integration are now present.
  Production runtime dynamic selection and converter GPU plan remain gated.
