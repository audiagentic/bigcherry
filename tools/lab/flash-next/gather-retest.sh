#!/bin/bash
# QFP21 1295 re-test on v6 (240K f16, ub512, MTP3 on the 6900): same binary, gather off (A arms) vs on (middle arm),
# at one cached depth. DEPTH is the long-ctx-profile depth (prompt tokens ~1.57x). Peak VRAM per arm in timing.vram.txt.
# Usage: gather-retest.sh <depth> <llama-server> <out-root>
export QUICK_DEPTH=$1 BIGCHERRY_FEATURES=flashnext BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0 BIGCHERRY_DRAFT_VOCAB_N=65536
export CTX=245760 EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_QSA_GATHER=0
exec bash "$(cd "$(dirname "$0")" && pwd)/quick-ab.sh" "$2" "$2" "$3" BIGCHERRY_QSA_GATHER=1 BIGCHERRY_PATCH_TRACE=1
