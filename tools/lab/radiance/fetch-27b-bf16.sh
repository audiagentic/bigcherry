#!/bin/bash
# RR01: download the bf16 safetensors release of Qwen3.8-27B. rad-convert's qwen35 plugins map the checkpoint's own
# tensor names (model.language_model.layers.N...), so the bf16 GGUF on Brutus cannot be the source of an 8-bit
# container, and a drafter is merged only into a safetensors target.
# Usage: fetch-27b-bf16.sh        (resumes; files already complete are kept)
# env: REPO (Qwen/Qwen3.8-27B), DEST
set -u
repo=${REPO:-Qwen/Qwen3.8-27B}
dest=${DEST:-/mnt/vault/llm-models/qwen3.8-27b/$(basename "$repo")}
mkdir -p "$dest"
echo "repo $repo -> $dest; free $(df -h "$dest" | tail -1 | awk '{print $4}')"
python3 - "$repo" "$dest" <<'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2], allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt"])
EOF
rc=$?
echo "download: exit $rc; $(du -sh "$dest" | cut -f1) in $(ls "$dest" | wc -l) files"
ls "$dest" | head -40
echo FETCH_27B_DONE
