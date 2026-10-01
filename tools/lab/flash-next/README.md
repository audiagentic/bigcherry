# Qwen3.8-Flash-Next layout lab (plan QFN01)

`layout-probe.sh` — first load test: per-layout llama-server start, per-device VRAM, and
prompt/decode tokens/s (+ draft acceptance) on a fixed ~1000-token prompt. Layer split only;
`per_layer_token_embd` pinned to CPU. Device order: 0,1 = 7900 XTX, 2 = R9700 (CPU PCIe),
3 = 6900 XT (chipset PCIe, always last). Run inside the 3-GPU window (vLLM stopped).
