#!/bin/bash
# Download the alternative Qwen3.8-27B drafter files compared by drafter-files.sh and verify their published sha256.
# Usage: fetch-drafters.sh        (idempotent: a file with the right hash is kept)
set -u
G=/mnt/data/llm-models/qwen3.8-27b/gguf
get() {  # <repo> <file> <dest dir> <sha256 prefix>
    local d="$G/$3" f="$G/$3/$2"
    mkdir -p "$d"
    if [ -f "$f" ] && [ "$(sha256sum < "$f" | cut -c1-20)" = "$4" ]; then echo "have $3/$2"; return; fi
    curl -fL --retry 5 -C - -o "$f.part" "https://huggingface.co/$1/resolve/main/$2" || { echo "DOWNLOAD_FAILED $1/$2"; return; }
    local h; h=$(sha256sum < "$f.part" | cut -c1-20)
    if [ "$h" = "$4" ]; then mv "$f.part" "$f"; echo "ok $3/$2 $(du -h "$f" | cut -f1)"; else echo "HASH_MISMATCH $3/$2 got $h want $4"; fi
}
get incoai/Qwen3.8-27B-DFlash2-GGUF Qwen3.8-27B-DFlash2-BF16.gguf dflash 26d47ca20ab07688327a
get magnitudedev/Qwen3.8-27B-DFlash2-GGUF Qwen3.8-27B-DFlash2-Q8_0.gguf dflash-magnitudedev e92a3bacead3b0ee40e2
get magnitudedev/Qwen3.8-27B-DSpark-GGUF Qwen3.8-27B-DSpark-Q8_0.gguf dspark-magnitudedev 466d05f4eab0f0e11d59
echo FETCH_DONE
