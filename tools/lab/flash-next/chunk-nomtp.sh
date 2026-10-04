#!/bin/bash
# 1332 single-token decode check (no MTP): the whole-mask kq_mask view crashed the meta backend before the
# n_tokens > chunk gate. Usage: chunk-nomtp.sh <llama-server> <out-root>
set -u
s=$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh
export NO_MTP=1 DECODE_N=64 DEPTH=24576 CTK=f16 CTV=f16
for c in 0 256; do
  echo "chunk $c: $(BIGCHERRY_QSA_CHUNK=$c bash "$s" "$1" "$2/c$c" timing 2>&1 | grep -E '^timing: prompt|SERVER_FAILED' | tr '\n' ' ')"
done
md5sum "$2"/c*/timing.24576.greedy.txt | awk '{print $1}' | sort | uniq -c
