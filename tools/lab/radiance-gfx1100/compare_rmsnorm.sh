#!/usr/bin/env bash
# Independent gfx1100 wave32 vs block256 RMSNorm performance/correctness.
# Requires installed Radiance plugin under the FIRST RADIANCE_HOME entry.
set -Eeuo pipefail
if [ "$#" -ne 3 ]; then
  echo "usage: ROCR_VISIBLE_DEVICES=<single-gfx1100-id> RADIANCE_HOME=<candidate-home>:<engine-home> $0 <libr11-build> <rad-kbench> <new-output-dir>" >&2
  exit 2
fi
build=$1
bench=$2
out=$3
[[ "${ROCR_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]] || { echo "select exactly one XTX device" >&2;exit 2; }
[[ -x "$bench" ]] || { echo "rad-kbench not executable: $bench" >&2;exit 2; }
mapfile -t plugins < <(find "$build" -type f -name 'libr11.so')
[[ ${#plugins[@]} -eq 1 ]] || { echo "expected exactly one built libr11.so" >&2;exit 2; }
candidate_home=$(dirname "$(dirname "${plugins[0]}")")
[[ -n "${RADIANCE_HOME:-}" ]] || { echo "RADIANCE_HOME required" >&2; exit 2; }
[[ "$(realpath "${RADIANCE_HOME%%:*}")" = "$(realpath "$candidate_home")" ]] || {
  echo "RADIANCE_HOME first entry must be $candidate_home" >&2;exit 2;
}
[[ ! -e "$out" ]] || { echo "refusing existing output directory $out" >&2;exit 2; }
mkdir -p "$out"
{
 date -u +%Y-%m-%dT%H:%M:%SZ
 echo "ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES"
 echo "RADIANCE_HOME=$RADIANCE_HOME"
 echo "plugin=${plugins[0]}"
 echo "bench=$bench"
 sha256sum "${plugins[0]}"
 if command -v hipcc >/dev/null; then hipcc --version 2>&1 | head -4 || true; fi
} > "$out/env.txt"
timeout 180 ctest --test-dir "$build" -R '^r11_rmsnorm_smoke$' --output-on-failure \
  2>&1 | tee "$out/smoke.log"
timeout "${KBENCH_TIMEOUT:-1800}" "$bench" --kernels libr11,libref \
  --op rmsnorm --bench --report "$out/norm-kbench.md" 2>&1 | tee "$out/kbench.log"
[[ -s "$out/norm-kbench.md" ]] || { echo "no new kbench report" >&2;exit 1; }
# The existing BigCherry report parser rejects numerically failed cases.
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python3 "$here/../radiance/kdev_report.py" "$out/norm-kbench.md" --json "$out/norm-kbench.json" \
  > "$out/table.txt"
# Presence in the registry is insufficient: every candidate MUST have checked, passing
# and timed cases. Fail closed when a fixture contains no supported norm geometry.
python3 - "$out/norm-kbench.json" <<'PY'
import json, math, sys
cases = json.load(open(sys.argv[1], encoding="utf-8"))["cases"]
for name in ("r11_rmsnorm_wave32", "r11_rmsnorm_block256", "r11_rmsnorm_dpp32"):
    mine = [x for x in cases if x["op"] == "rmsnorm" and x["kernel"] == name]
    checked = [x for x in mine if x["verdict"] == "ok"]
    failures = [x for x in mine if x["verdict"] not in ("ok", "")]
    timed = [x for x in checked if x["us"] is not None and x["us"] > 0]
    if failures or not checked or not timed:
        sys.exit(f"FAIL {name}: reported={len(mine)} passed={len(checked)} timed={len(timed)} failed={len(failures)}")
    geo = math.exp(sum(math.log(x["us"]) for x in timed) / len(timed))
    print(f"{name}: PASS {len(checked)} checked, {len(timed)} timed, geometric mean {geo:.2f} us")
PY
echo "COMPLETE: $out/norm-kbench.md; parse $out/norm-kbench.json for per-geometry evidence."
