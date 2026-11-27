# V3 draft input resolution

The lead supplied the upstream unquantized checkpoint with a verified published LFS digest. The local payload size is 3848817896 bytes and the header contains 81 BF16 tensors. No quantization configuration is present. Use it directly with the existing linked target-head view; do not dequantize FP8, reconstruct GGUF or change any model input. The earlier dual-target package remains untested and out of scope.
