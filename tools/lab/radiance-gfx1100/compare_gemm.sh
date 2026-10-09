#!/usr/bin/env bash
# Hardware-only comparison of independent gfx1100 GEMM variants against Radiance libref.
# Does not build or modify Radiance, switch drivers, touch global GPU state or enable P2P.
set -Eeuo pipefail
if [[ $# != 3 ]]; then
  echo "usage: ROCR_VISIBLE_DEVICES=<verified-XTX-id> $0 <libr11-build> <rad-kbench> <new-output-dir>" >&2
  exit 2
fi
build=$1
bench=$2
out=$3
[[ -n "${ROCR_VISIBLE_DEVICES:-}" ]] || { echo "set ROCR_VISIBLE_DEVICES to exactly one verified gfx1100 GPU" >&2; exit 2; }
[[ "$ROCR_VISIBLE_DEVICES" =~ ^[0-9]+$ ]] || { echo "only one GPU may be visible" >&2; exit 2; }
[[ -x "$bench" ]] || { echo "rad-kbench is not executable: $bench" >&2; exit 2; }
[[ -f "$build/libr11.so" ]] || { echo "libr11.so is missing under $build" >&2; exit 2; }
[[ -n "${RADIANCE_HOME:-}" ]] || { echo "RADIANCE_HOME must point at the installed engine + libr11 plugin home" >&2; exit 2; }
if [[ -e "$out" ]]; then
  echo "Output directory already exists; refusing stale report: $out" >&2
  exit 2
fi
mkdir -p "$out"
{
  date -u +%Y-%m-%dT%H:%M:%SZ
  echo "ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES"
  echo "RADIANCE_HOME=$RADIANCE_HOME"
  echo "bench=$bench"
  echo "libr11_so=$build/libr11.so"
  hipcc --version 2>&1 | head -6 || true
} > "$out/environment.txt"
# Test the two paths against an asymmetric CPU oracle before interpreting any timings.
timeout 120 ctest --test-dir "$build" -R '^r11_gemm_smoke$' --output-on-failure \
  2>&1 | tee "$out/smoke.log"
# rad-kbench runs BOTH named candidates against libref, with identical buffers/red zones.
# IMPORTANT: the engine's RADIANCE_HOME must contain the same libr11.so as the smoke.
timeout "${KBENCH_TIMEOUT:-1800}" "$bench" --kernels libr11,libref \
  --op gemm_nt --bench --report "$out/gemm-kbench.md" \
  2>&1 | tee "$out/gemm-kbench.log"
test -s "$out/gemm-kbench.md" || { echo "no kbench report" >&2; exit 1; }
grep -Fq 'r11_gemm_bf16_scalar' "$out/gemm-kbench.md" || {
  echo "scalar candidate absent from kbench report" >&2; exit 1;
}
grep -Fq 'r11_gemm_bf16_wmma_gfx1100' "$out/gemm-kbench.md" || {
  echo "WMMA candidate absent from kbench report" >&2; exit 1;
}
# Presence does not prove the fixture covered a band: inspect checked/skipped/failing rows.
echo "COMPLETE: inspect $out/gemm-kbench.md for per-shape coverage and numerical results."
