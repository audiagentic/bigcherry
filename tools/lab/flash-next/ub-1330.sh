#!/bin/bash
# ub sweep 512, 1024, 1024, 512 with 1330 on (in-place QSA mask) and alloc-top logging. Usage: ub-1330.sh <bin> <root>
set -u
export BIGCHERRY_QSA_MASK_INPLACE=1 BIGCHERRY_ALLOC_TOP=12
exec bash "$(cd "$(dirname "$0")" && pwd)/ub-sweep.sh" "$1" "$2" 512 1024
