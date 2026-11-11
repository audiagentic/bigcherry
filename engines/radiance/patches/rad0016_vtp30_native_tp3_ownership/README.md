# VTP30 — native Radiance Qwen3.8-27B asymmetric TP3 weight ownership

**Experimental; not production; not a heterogeneous-GPU launch enablement.**

This source-pinned v1.17.0 patch modifies Radiance's *existing* Qwen BF16
declaration and common graph blocks. It changes no scheduler, model
execution loop, KV allocator, recurrent/conv state, weight loader, collective
implementation, embedding or sampling code. It does not install or invoke
the standalone VTP HIP kernels.

Activate only an experimental `source.radiance-vtp30-native-tp3` recipe,
with `BC_VTP_NATIVE_TP3=1`, exact original 64-layer dense Qwen3.8-27B
model geometry and `--tp 3`. All other models and invocations follow the
unchanged Radiance geometry. Wrong dimensions return an error rather than
guessing the source weight layout.

## Physical rank source ownership

| Component | XTX0 | XTX1 | R9700 |
| --- | ---: | ---: | ---: |
| GDN key heads | 4 | 4 | 8 |
| GDN value heads | 12 | 12 | 24 |
| Full attention Q/output-gate heads | 6 | 6 | 12 |
| Full attention KV heads | 1 | 1 | 2 |
| FFN intermediate width | 5120 | 4864 | 7424 |
| GDN/attention output-projection input columns | 1536 | 1536 | 3072 |

Uses the existing `rad_weight_shard_span` for simple row/col cuts
and `rad_weight_shard_span_parts` for GDN stacked Q|K|V|a|b and
depthwise convolution. The builder, loader, KV manager and native
per-layer ops are the source of truth.

## Acceptance and deliberate blocker

Card-free pinned-tree patch-application and idempotence tests, original
source-geometry coverage tests and `g++ -std=c++23 -fsyntax-only` of
the patched Qwen BF16 architecture plug-in against Radiance's headers.
The isolated native C++23 plugin also builds with `-fPIC -shared` to ${BC_WORK_ROOT}/xtx27-prefill/vtp30-native-source-smoke/librad_qwen35_bf16_vtp30.so.
This library is not installed or linked into the serving engine.

**This does not run heterogeneous TP3.** Radiance currently filters to
devices of one architecture in `core/engine_bringup.cpp:usable_devices`,
and kernel plugin loading checks device code against **only the first
device**. Disabling the filter alone would enable invalid kernel
selection on the R9700. A later gated native integration must provide
per-rank device-code eligibility and correctly resolved per-rank kernel
dispatch before hardware activation. Until then, keep the filter intact.

The original BF16 TP3 VTP19/VTP20/VTP23/VTP25/VTP27 receipts remain
independent numerical and timing references; no file-mediated model
execution is added to Radiance. No claims of full-model text-generation,
quality equivalence, matched TP2/TP3 tokens/s or 262K admission.
