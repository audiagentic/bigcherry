# Converter GPU target coverage

The existing runtime build targets one server-side architecture. Its quantizer fat binary therefore carried only that architecture, yet the standalone converter can be asked to quantize on a different GPU. The old crash trace has `lq::run_rows -> hipLaunchKernel`; the binary inventory confirms the absent image.

This anchored patch gives the libquant HIP target a separate, explicit build-time architecture setting, while preserving the old default. The converter operator must include the ISA of every conversion GPU in `RAD_QUANT_GPU_TARGETS` and verify fat-binary coverage with `llvm-objdump --offloading` before running. It does not relax a quantization input requirement or silently fall back to a different algorithm.

Validation: patch idempotence/anchor test; repository patch-lint and offline patch tests; quantizer code-object enumeration; synthetic and full-recipe device conversion. The first two are host-only. Keep this patch untested until full validation succeeds.
