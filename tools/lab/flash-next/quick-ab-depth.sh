#!/bin/bash
# quick-ab.sh at a chosen cached depth (queue.sh SCRIPT jobs run `bash <script>`, so no env prefix there).
# Usage: quick-ab-depth.sh <depth> <base llama-server> <new llama-server> <out-root> [new-arm env...]
export QUICK_DEPTH=$1
shift
exec bash "$(cd "$(dirname "$0")" && pwd)/quick-ab.sh" "$@"
