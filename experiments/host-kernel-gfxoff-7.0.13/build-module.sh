#!/usr/bin/env bash
# Rebuild only the amdgpu module from the preserved Linux 7.0.13 source recipe.
# No installation, module loading, depmod, bootloader, or host kernel mutation.
set -euo pipefail

usage() {
  echo "Usage: BC_WORK_ROOT=<work-root> $0 <linux-7.0.13.tar.xz> <scratch-directory> [jobs: 1..12]" >&2
  exit 2
}
[[ $# -ge 2 && $# -le 3 ]] || usage
archive=$1
scratch=$2
jobs=${3:-8}
if ! [[ "$jobs" =~ ^[0-9]+$ ]]; then usage; fi
if (( jobs < 1 || jobs > 12 )); then usage; fi
[[ -n "${BC_WORK_ROOT:-}" ]] || { echo "BC_WORK_ROOT must identify the work root" >&2; exit 2; }
root=$(realpath -m -- "$BC_WORK_ROOT")
scratch=$(realpath -m -- "$scratch")
case "$scratch/" in "$root/"*) ;; *) echo "Scratch must be under BC_WORK_ROOT" >&2; exit 2 ;; esac
[[ ! -e "$scratch" ]] || { echo "Refusing existing scratch directory: $scratch" >&2; exit 2; }
[[ -f "$archive" ]] || usage
archive=$(realpath -- "$archive")
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)

# Archive digest is the observed kernel.org download; signature is separately
# published at https://cdn.kernel.org/pub/linux/kernel/v7.x/linux-7.0.13.tar.sign.
printf '%s  %s\n' '3c81edd0f716aca3dd48dff691681827580cc53d35a8eec3be47d346d1f89913' "$archive" | sha256sum --check --status || {
  echo "Linux archive SHA-256 mismatch" >&2; exit 1;
}
for tool in git make gcc nice tar xz sha256sum flex bison bc perl cmp; do
  command -v "$tool" >/dev/null || { echo "Missing build dependency: $tool; no build attempted" >&2; exit 1; }
done

mkdir -p -- "$scratch"
echo "Scratch output retained at $scratch (remove by exact name after receipt review)."
tar -xJf "$archive" -C "$scratch"
mv -- "$scratch/linux-7.0.13" "$scratch/stock"
cp -a --reflink=auto -- "$scratch/stock" "$scratch/patched"
while IFS= read -r patch_file || [[ -n "$patch_file" ]]; do
  [[ -z "$patch_file" || "$patch_file" == \#* ]] && continue
  git -C "$scratch/patched" apply --check "$here/patches/$patch_file"
  git -C "$scratch/patched" apply "$here/patches/$patch_file"
done < "$here/series"
cmp -- "$scratch/patched/drivers/gpu/drm/amd/amdgpu/amdgpu_amdkfd.c" "$here/amdgpu_amdkfd.c" || {
  echo "Patched reference source mismatch" >&2; exit 1;
}

for arm in stock patched; do
  src="$scratch/$arm"
  out="$scratch/$arm-out"
  mkdir -p -- "$out"
  cp -- "$here/config-7.0.13-gfxoff" "$out/.config"
  # olddefconfig can be sensitive to compiler versions; retain output .config.
  nice -n 10 make -j1 -C "$src" O="$out" olddefconfig
  nice -n 10 make -j"$jobs" -C "$src" O="$out" modules_prepare
  # In-tree amdgpu only. Do not build the full kernel or other modules.
  nice -n 10 make -j"$jobs" -C "$src" O="$out" M=drivers/gpu/drm/amd/amdgpu modules
  module="$out/drivers/gpu/drm/amd/amdgpu/amdgpu.ko"
  [[ -f "$module" ]] || { echo "No amdgpu.ko at $module" >&2; exit 1; }
  if command -v modinfo >/dev/null; then
    echo "$arm srcversion: $(modinfo -F srcversion "$module")"
  else
    echo "$arm built: $module; modinfo unavailable (inspect .modinfo via readelf)"
  fi
done
echo "Recorded control: 4F5C8C3B57F8DA15B66C039"
echo "Recorded patched: 2F0E6E3215789522FB92997"
echo "Compare both results before claiming byte-exact source reconstruction."
