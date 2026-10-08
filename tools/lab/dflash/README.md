# DFlash speculative-decoding probes

- `probe-27b.sh` (v3): draft x deployment matrix (dual XTX tensor, 3-card tensor, dual XTX layer); Qwen3.8-27B Q8_0 on dual XTX, MTP depth 5 (built-in and unsloth sidecar) vs DFlash2 (Q8_0/Q4_K_M) vs DSpark Q8_0, draft on the XTXs or the 6900 XT.
- `queue-27b-dflash.sh`: queues the probe on Brutus (R9700/vLLM untouched).

Flash-Next has no llama.cpp-convertible DFlash draft (the only one, PixelML's NVFP4 DeepSpec drafter, is `Qwen3DSparkModel`, which has no converter and its authors report about +3.9% vs MTP).
- `drafter-files.sh`: which drafter file is best for Qwen3.8-27B (current DFlash2 Q8_0 / Q4_K_M and DSpark Q8_0 vs the BF16 and the 2026-10-06 re-published GGUFs) against built-in MTP; reuses `probe()` from `probe-27b.sh`.
- `fetch-drafters.sh`: downloads those alternative drafter files and checks their published sha256.
- `dflash-accept-iso.sh`: isolates the DFlash2 / DSpark acceptance collapse at pin b11474 (1-2% against 44-59% at b11402) on a single-device Q4_K_M target, to compare native llama.cpp with BigCherry builds; reuses `probe()` from `probe-27b.sh`.
