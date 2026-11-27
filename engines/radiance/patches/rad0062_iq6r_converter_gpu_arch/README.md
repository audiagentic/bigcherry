# Converter GPU target coverage

The existing runtime build targets one server-side architecture. Its quantizer fat binary therefore carried only that architecture, yet the standalone converter can be asked to quantize on a different GPU. The old crash trace has `lq::run_rows -> hipLaunchKernel`; the binary inventory confirms the absent image. This patch gives the libquant HIP target a separate, explicit build-time architecture setting while preserving the old default. The converter operator must include the ISA of every conversion GPU in `RAD_QUANT_GPU_TARGETS` and verify fat-binary coverage with `llvm-objdump --offloading` before running. It does not relax a quantization input requirement or silently fall back to a different algorithm.

## Summary

Give the `libquant` HIP target an independent, explicit build-time architecture list (`RAD_QUANT_GPU_TARGETS`) so a converter can quantize on a GPU whose ISA is absent from the serving kernel images. The default stays the server's `RAD_GPU_TARGETS`; a converter build names its own, for example `-DRAD_QUANT_GPU_TARGETS=gfx1100;gfx1201`. No kernel algorithm, tensor bytes or device arbitration change, and this is not a fallback that skips GPTQ.

## Outcome

Untested on device. The anchor applies cleanly and idempotently to the pinned 2.2.0 source (b6b8751) and the hermetic patch test (idempotence, missing-anchor refusal) passes. `RAD_GPU_TARGETS` is always defined by the root CMake in a HIP build, so the defaulted value is non-empty whenever `LQ_HIP_SRC` is present. Remaining before a source may select it: quantizer code-object enumeration, a synthetic device conversion, then a full-recipe conversion on the conversion GPU.

## Detail

One anchored edit in `libquant/CMakeLists.txt` at the existing `LQ_CARD` definition block: it declares `RAD_QUANT_GPU_TARGETS` as a cache string initialized from `RAD_GPU_TARGETS`, refuses an empty value with `FATAL_ERROR` instead of a silent default, and sets the target's `HIP_ARCHITECTURES` property to the value. The unit test applies the patch to a synthetic copy of the anchor block in a temporary directory and checks idempotence and missing-anchor refusal without any vendor checkout. Validation: patch idempotence/anchor test, repository patch-lint and offline patch tests (host-only), then the device steps above. Keep this patch `untested` until full validation succeeds.
