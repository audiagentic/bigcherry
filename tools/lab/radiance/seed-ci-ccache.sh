#!/usr/bin/env bash
# shellcheck disable=SC1091
# BU01: import only a successful main CI compiler cache with an identical SDK,
# host compiler, HIP compiler and recipe fingerprint.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export PYTHONPATH="$repo_root/tools"
if [ -z "${BC_WORK_ROOT:-}" ]; then
  BC_WORK_ROOT=$(python3 -c 'from bigcherry.core.context import ProjectContext; print(ProjectContext.resolve(engine="radiance").work_root)')
fi
export BC_WORK_ROOT
export CCACHE_DIR="$BC_WORK_ROOT/.ccache"
export CCACHE_BASEDIR="$repo_root"
export CCACHE_NOHASHDIR=true CCACHE_COMPILERCHECK=content
export RADIANCE_HOST_CC="${RADIANCE_HOST_CC:-gcc-14}"
export RADIANCE_HOST_CXX="${RADIANCE_HOST_CXX:-g++-14}"
source "$repo_root/tools/lab/radiance/lab-env.sh"

exec python3 "$repo_root/tools/lab/radiance/ci_ccache.py" fetch --work-root "$BC_WORK_ROOT" "$@"
