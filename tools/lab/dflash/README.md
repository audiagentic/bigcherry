# DFlash speculative-decoding probes

- `probe-27b.sh`: Qwen3.8-27B Q8_0 on dual XTX, MTP depth 5 vs DFlash2 (Q8_0/Q4_K_M), draft on the XTXs or the 6900 XT.
- `queue-27b-dflash.sh`: queues the probe on Brutus (R9700/vLLM untouched).

Flash-Next has no llama.cpp-convertible DFlash draft (the only one, PixelML's NVFP4 DeepSpec drafter, is `Qwen3DSparkModel`, which has no converter and its authors report about +3.9% vs MTP).
