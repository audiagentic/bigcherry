# XDP10 — converter quantizer GPU targets

**Status:** untested
**Plan item:** XDP10

Provide an optional, independently selectable HIP architecture list for the converter's `libquant` plugin. The serving engine can continue to target only its own GPU generation. With a conversion worker on a different generation, the quantizer must carry that generation's code object before launching trellis kernels.

The default remains `RAD_GPU_TARGETS`; a converter build can explicitly use `-DRAD_QUANT_GPU_TARGETS=gfx1100;gfx1201`. This changes no kernel algorithm, tensor bytes or device arbitration and is not a fallback that skips GPTQ. Candidate state: untested until device and conversion validation.
