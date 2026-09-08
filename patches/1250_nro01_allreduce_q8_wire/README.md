# 1250_nro01_allreduce_q8_wire

Plan: `NRO01`. State: `untested`. Backend: HIP. Required patch: `1001_hip_internal_allreduce`.

## Scope

This package is the first atomic extraction from nasone commit `e06dcf6300718227cb8cfda9e61fb12ccb693418`. It owns only Q8_0 encoding/decoding primitives and the runtime threshold state. Residual fusion is NRO02; P2P transport is NRO03; existing BigCherry split-reduce telemetry remains the observability authority.

The initial implementation is deliberately **not validation-ready**. It adds compile-time kernel primitives and an opt-in threshold field but does not dispatch production reductions through Q8. This prevents a lossy path from becoming reachable before a reference fixture and numerical policy exist.

## Draft behavior

`GGML_CUDA_AR_Q8_THRESHOLD` is parsed into the internal AllReduce pipeline. Default `0` means disabled. The patch adds a block-Q8_0 quantizer and a Q8_0 two-rank dequantized add kernel with explicit padded-tail handling. A later implementation step will connect these to the copy-engine path and add activation/wire-byte counters.

## Control / subject

Future causal comparison:

- control: `1001_hip_internal_allreduce`, exact FP32 wire, same provider/config;
- subject: control + this patch with Q8 enabled at a pre-registered threshold.

BF16, RCCL, residual fusion and P2P are characterization/composition arms, not substitutes for the focal control.

## Limitations

No live dispatch is wired yet. No Experiment Contract or `validation.toml` is intentionally created until the source transformation applies/builds and numerical fixtures define a defensible tolerance. Do not add this patch to production recipes.

See `TESTING.md` and `docs/planning/active/nasone-rdna-optimizations/NRO01.md`.
