#!/bin/bash
# check_build_sdk BUILD: a build runs against the ROCm SDK recorded in its metadata (rocm_root; RUNPATH points at it).
# Refuses (returns 1, says why) when the metadata records none, or when the environment points the process at another
# SDK: ROCM_PATH / HIP_PATH, or a LD_LIBRARY_PATH directory of a different ROCm install. Nothing is overridden silently.
check_build_sdk() {
  local build=$1 meta root var val dir
  meta="$build/bigcherry-build-metadata-radiance.json"
  root=$(python3 -I -c '
import json, sys
tool = dict(map(tuple, json.load(open(sys.argv[1])).get("toolchain", [])))
print(tool.get("rocm_root", ""))' "$meta" 2>/dev/null)
  [ -n "$root" ] || { echo "sdk-check: $meta records no rocm_root; rebuild with a named ROCm stack" >&2; return 1; }
  [ -d "$root" ] || { echo "sdk-check: the ROCm SDK this build records ($root) is not installed" >&2; return 1; }
  root=$(readlink -f "$root")
  for var in ROCM_PATH HIP_PATH; do
    val=${!var:-}
    [ -z "$val" ] || [ "$(readlink -f "$val")" = "$root" ] || {
      echo "sdk-check: $var=$val is not the SDK this build records ($root); unset it" >&2; return 1; }
  done
  local IFS=:
  for dir in ${LD_LIBRARY_PATH:-}; do
    [ -n "$dir" ] || continue
    [ -e "$dir/libamdhip64.so" ] || [ -e "$dir/libhsa-runtime64.so" ] || continue
    case "$(readlink -f "$dir")" in "$root"/*) ;; *)
      echo "sdk-check: LD_LIBRARY_PATH entry $dir carries ROCm libraries outside the recorded SDK ($root)" >&2; return 1 ;;
    esac
  done
}
