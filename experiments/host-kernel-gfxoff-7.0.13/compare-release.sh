#!/usr/bin/env bash
# Compare the upstream GFXOFF patch with a local Linux tree, tarball or version.
# All downloaded/unpacked content remains under the supplied scratch directory.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <source-tree|linux-VERSION.tar.xz|VERSION> <scratch-parent>" >&2
  exit 2
fi
input=$1
scratch=$2
[[ -d "$scratch" ]] || { echo "Create a scratch parent under the work root first" >&2; exit 2; }
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
work=$(mktemp -d -- "$scratch/gfxoff-compare.XXXXXXXX")
trap 'rm -rf -- "$work"' EXIT
source_dir=
if [[ -d "$input" ]]; then
  source_dir=$(cd -- "$input" && pwd -P)
else
  archive=$input
  if [[ ! -f "$archive" ]]; then
    version=${input#v}
    if [[ ! "$version" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]]; then
      echo "Expected a local tree, tarball, or Linux version/tag" >&2
      exit 2
    fi
    major=${version%%.*}
    archive="$work/linux-$version.tar.xz"
    curl -fL --retry 2 -o "$archive" "https://cdn.kernel.org/pub/linux/kernel/v${major}.x/linux-$version.tar.xz"
  fi
  tar -xf "$archive" -C "$work"
  source_dir=$(find "$work" -mindepth 1 -maxdepth 1 -type d -name 'linux-*' -print -quit)
  [[ -n "$source_dir" ]] || { echo "No Linux source directory in archive" >&2; exit 1; }
fi
target="$source_dir/drivers/gpu/drm/amd/amdgpu/amdgpu_amdkfd.c"
[[ -f "$target" ]] || { echo "Target file absent in this release" >&2; exit 1; }

echo "Input: $input"
clean=yes
while IFS= read -r patch_file || [[ -n "$patch_file" ]]; do
  [[ -z "$patch_file" || "$patch_file" == \#* ]] && continue
  if git -C "$source_dir" apply --check "$here/patches/$patch_file"; then
    echo "$patch_file: git apply --check CLEAN"
  else
    echo "$patch_file: git apply --check DOES NOT APPLY CLEANLY"
    clean=no
  fi
done < "$here/series"

# Limit clause inspection to the named compute-idle function, not other GFX12 tests.
function_body=$(sed -n '/^void amdgpu_amdkfd_set_compute_idle/,/^}/p' "$target")
if [[ -z "$function_body" ]]; then
  echo "Target function: not found; fix status unknown (function refactored or removed)"
elif grep -Fq 'IP_VERSION_MAJ(amdgpu_ip_version(adev, GC_HWIP, 0)) == 12' <<< "$function_body"; then
  echo "Targeted GFX12 exclusion: PRESENT"
  echo "Upstream fix effect: ABSENT in this release"
else
  echo "Targeted GFX12 exclusion: ABSENT"
  echo "Upstream fix effect: PRESENT or function changed; commit ancestry NOT proven by source inspection"
fi
echo "Series applies: $clean"
