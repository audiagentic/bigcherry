#!/bin/bash
# Download the original Qwen/Qwen3.8-Flash-Next checkpoint (BF16 safetensors, 335 GiB) for re-converting a radiance
# container (e.g. with the MTP draft layer at higher precision). Resumable: re-run after an interruption.
# Usage: fetch-flashnext-source.sh   env: DEST (/mnt/vault/llm-models/hf/Qwen3.8-Flash-Next)
set -euo pipefail
dest=${DEST:-/mnt/vault/llm-models/hf/Qwen3.8-Flash-Next}
mkdir -p "$dest"
python3 - "$dest" <<'PY'
import sys
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen3.8-Flash-Next", local_dir=sys.argv[1], max_workers=8)
print("FETCH_DONE", p)
PY
