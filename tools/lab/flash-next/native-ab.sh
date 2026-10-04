#!/bin/bash
# QFP18: v6 (A arms, BIGCHERRY_FEATURES=flashnext, cpu-root AllReduce) vs native llama.cpp (middle arm, AR=none)
# at one cached depth via quick-ab. Usage: native-ab.sh <depth> <v6 llama-server> <native llama-server> <out-root>
export QUICK_DEPTH=$1 BIGCHERRY_FEATURES=flashnext BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0 BIGCHERRY_DRAFT_VOCAB_N=65536
exec bash "$(cd "$(dirname "$0")" && pwd)/quick-ab.sh" "$2" "$3" "$4" AR=none
