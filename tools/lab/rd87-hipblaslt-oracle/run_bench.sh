#!/usr/bin/env bash
# RD87 oracle: run hipblaslt-bench heuristic vs all-solutions for each real
# captured (K,M,N) shape, dump raw stdout per shape. Run on Brutus only.
set -euo pipefail

BENCH=/home/audumla/bigcherry/vendor/rocm-libraries/projects/hipblaslt/hipblaslt-install/bin/hipblaslt-bench
export LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib/llvm/lib:/opt/rocm/lib
SHAPES_JSON="$1"
OUT_DIR="$2"
mkdir -p "$OUT_DIR"

python3 - "$SHAPES_JSON" <<'PYEOF' > "$OUT_DIR/shapes_unique_kmn.txt"
import json, sys
rows = json.load(open(sys.argv[1]))
seen = set()
for r in rows:
    key = (r["k"], r["m"], r["n"])
    if key not in seen:
        seen.add(key)
        print(*key)
PYEOF

while read -r k m n; do
  tag="k${k}_m${m}_n${n}"
  echo "=== $tag heuristic ===" | tee -a "$OUT_DIR/$tag.log"
  "$BENCH" -m "$m" -n "$n" -k "$k" --precision f16_r --algo_method heuristic --iters 50 --cold_iters 10 >> "$OUT_DIR/$tag.log" 2>&1 || echo "FAILED heuristic $tag" >> "$OUT_DIR/$tag.log"
  echo "=== $tag all ===" >> "$OUT_DIR/$tag.log"
  "$BENCH" -m "$m" -n "$n" -k "$k" --precision f16_r --algo_method all --iters 50 --cold_iters 10 >> "$OUT_DIR/$tag.log" 2>&1 || echo "FAILED all $tag" >> "$OUT_DIR/$tag.log"
done < "$OUT_DIR/shapes_unique_kmn.txt"

echo "done, logs in $OUT_DIR"
