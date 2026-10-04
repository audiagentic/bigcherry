#!/bin/bash
# FMTP03 screen: same binary, v6 (A arms) vs v6 + BIGCHERRY_MTP_AHEAD=1 (middle arm), 240K f16 deployment config.
# Usage: ahead-ab.sh <depth> <llama-server> <out-root> [tail p_min]
export QUICK_DEPTH=$1 BIGCHERRY_FEATURES=flashnext BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0 BIGCHERRY_DRAFT_VOCAB_N=65536
export CTX=245760 EXTRA_OT='^token_embd\.weight$=CPU'
newenv="BIGCHERRY_MTP_AHEAD=1"
[ -n "${4:-}" ] && newenv="$newenv BIGCHERRY_MTP_AHEAD_PMIN=$4"
exec bash "$(cd "$(dirname "$0")" && pwd)/quick-ab.sh" "$2" "$2" "$3" $newenv
