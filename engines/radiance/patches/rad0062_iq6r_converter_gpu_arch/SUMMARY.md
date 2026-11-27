# rad0062 — converter quantizer GPU targets (XDP10)

**Status:** untested
**Plan item:** XDP10

## Summary

The runtime build targets one server-side architecture, so the quantizer's fat binary carries only that architecture's code objects, yet the standalone converter can be asked to quantize on a different GPU. The crash trace is `lq::run_rows -> hipLaunchKernel` with the other generation's image absent from the binary inventory. This patch gives the `libquant` HIP target a separate, explicit build-time architecture setting: `RAD_QUANT_GPU_TARGETS` defaults to the server's `RAD_GPU_TARGETS` and a converter build can explicitly use, for example, `-DRAD_QUANT_GPU_TARGETS=gfx1100;gfx1201`. The converter operator must include the ISA of every conversion GPU and verify fat-binary coverage with `llvm-objdump --offloading` before running. It changes no kernel algorithm, tensor bytes or device arbitration, is not a fallback that skips GPTQ, and does not relax any quantization input requirement.

## Outcome

Untested on device. Verified so far: the anchor applies cleanly and idempotently to the pinned 2.2.0 source (b6b8751), and the hermetic patch test (idempotence, missing-anchor refusal) passes. `RAD_GPU_TARGETS` is defined by the root CMake in every HIP build (device enumeration or explicit `-D`), so the defaulted cache variable is non-empty whenever `LQ_HIP_SRC` exists. Remaining validation before this patch may be selected by a source: quantizer code-object enumeration, a synthetic device conversion, then a full-recipe conversion on the conversion GPU.

## Detail

One anchored edit in `libquant/CMakeLists.txt` at the existing `LQ_CARD` definition block: it declares `RAD_QUANT_GPU_TARGETS` as a cache string initialized from `RAD_GPU_TARGETS`, refuses an empty value with `FATAL_ERROR` instead of a silent default, and sets the target's `HIP_ARCHITECTURES` property to the value. The unit test applies the patch to a synthetic copy of the anchor block in a temporary directory and checks the idempotence and missing-anchor behaviours without any vendor checkout.
