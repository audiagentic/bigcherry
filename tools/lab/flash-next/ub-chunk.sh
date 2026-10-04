#!/bin/bash
# QFP17 1332: ub 512 then 1024 load at 240K f16 with QSA token chunks of 256 and the live-at-peak trace.
# Usage: ub-chunk.sh <bin> <root>
set -u
export BIGCHERRY_QSA_CHUNK=256 BIGCHERRY_ALLOC_PEAK=12
exec bash "$(cd "$(dirname "$0")" && pwd)/ub-sweep.sh" "$1" "$2" 512 1024
