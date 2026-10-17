#!/bin/bash
# MEN15: the CPU smoke models, under <model-root>/smoke/ (sha256 pinned in each engine.toml [smoke.cpu]).
#   tiny-llama.gguf  llama.cpp - llama.cpp's own CI model, stories15M unquantised (ggml-org/models)
#   tiny.rad         radiance  - Felladrin/Llama-68M-Chat-v1 converted by rad-convert: radiance's llama plugin
#                    takes fp8 linears in 128x128 blocks (stories15M's 288-wide dims do not divide) and bf16 elsewhere
# Usage: MODEL_ROOT=/mnt/data/llm-models RAD_BUILD=<radiance lane build> bash fetch-tiny.sh
set -euo pipefail
root=${MODEL_ROOT:?model root}
mkdir -p "$root/smoke"
python3 - "$root/smoke" <<'PY'
import shutil, sys
from huggingface_hub import hf_hub_download, snapshot_download
shutil.copyfile(hf_hub_download("ggml-org/models", "tinyllamas/stories15M.gguf"), f"{sys.argv[1]}/tiny-llama.gguf")
snapshot_download("Felladrin/Llama-68M-Chat-v1", local_dir=f"{sys.argv[1]}/src/llama-68m",
                  allow_patterns=["*.json", "*.safetensors", "*.model"])
PY
"${RAD_BUILD:?radiance build}/bin/rad-convert" "$root/smoke/src/llama-68m" -o "$root/smoke/tiny.rad" \
  --home "$RAD_BUILD/radiance_home" \
  --quant 'blk.*.weight=rtn:codes=fp8_e4m3,block=128x128,scale=bf16' --quant '*=cast:dtype=bf16'
sha256sum "$root/smoke/tiny-llama.gguf" "$root/smoke/tiny.rad"
