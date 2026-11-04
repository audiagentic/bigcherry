#!/usr/bin/env bash
# shellcheck disable=SC1091
# BU01 stock release fetch: verified prebuilt bundle or normal lab compilation.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export PYTHONPATH="$repo_root/tools"
if [ -z "${BC_WORK_ROOT:-}" ]; then
  BC_WORK_ROOT=$(python3 -c 'from bigcherry.core.context import ProjectContext; print(ProjectContext.resolve(engine="radiance").work_root)')
fi
export BC_WORK_ROOT
source "$repo_root/tools/lab/radiance/lab-env.sh"
export BC_ROCM_ROOT="${BC_ROCM_ROOT:-$ROCM_PATH}"
if python3 "$repo_root/tools/lab/radiance/fetch_release_bundle.py" "$@"; then
    exit 0
fi

# Missing/corrupt/revoked assets, source edits and SDK drift all compile through
# the existing rad_build path. Never run a downloaded executable on a failed check.
echo "BU01: prebuilt asset unavailable; falling back to the local compiler" >&2
source "$repo_root/tools/lab/radiance/lab-build.sh"
mkdir -p "$repo_root/artifacts/logs/manual"
cd "$repo_root"
rad_build "$repo_root/artifacts/logs/manual/bu01-fallback-build.log"
