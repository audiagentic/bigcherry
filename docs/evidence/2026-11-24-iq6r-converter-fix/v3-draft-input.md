# V3 draft input verification

The plain safetensors draft importer accepts BF16/F16; the only declared quantized import is ModelOpt NVFP4 group 16. GGUF is not a draft input.
The upstream plain draft header has 81 BF16 tensors and no quantization configuration. FC is 5120 x 25600; Q is 4096 x 5120; K/V are 1024 x 5120; O is 5120 x 4096; gate and up are each 17408 x 5120 (concatenated on import); down is 5120 x 17408.
The draft head copies the target head, not a draft checkpoint tensor. The existing linked head view supplies it. No reconstruction or dequantization is needed. The dual-target patch remains untested and is not required for this device.
