#!/bin/bash
# Run the adaptive max-context search for the KV/ubatch combinations the owner allows (never q4).
# Usage: maxctx-search.sh <llama-server> <out-root>
set -u
bin=$1 root=$2
py=$(cd "$(dirname "$0")" && pwd)/maxctx-search.py
python3 "$py" "$bin" "$root/k16-v8-ub512" f16 q8_0 512 98304 262144
python3 "$py" "$bin" "$root/k16-v8-ub256" f16 q8_0 256 98304 262144
python3 "$py" "$bin" "$root/k16-v16-ub512" f16 f16 512 98304 262144
python3 "$py" "$bin" "$root/k8-v8-ub512" q8_0 q8_0 512 163840 262144
