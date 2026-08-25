#!/usr/bin/env bash
# Compatibility wrapper; source the canonical environment helper.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env/rocm-env.sh" "$@"
