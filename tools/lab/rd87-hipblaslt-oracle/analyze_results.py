"""Parse run_bench.sh's per-shape hipblaslt-bench logs and compare against
real native timing from the extracted shapes file, per RD87. Not a library --
ad hoc lab driver, run directly on Brutus. See README.md.

Bug note (2026-09-09): an earlier ad hoc version of this comparison matched
section headers with `header.endswith('all')`, but headers keep their
trailing ' ===' (e.g. "k5120_m5120_n1 all ==="), so the match always failed
and every all_ratio column silently came out empty. That earlier bug
produced a false blanket "no-go" -- it hid every shape where hipBLASLt's
best-of-all-solutions run actually beat native. Strip the trailing '=='
punctuation before matching (done below) to avoid repeating it.
"""
import csv
import glob
import json
import re
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} <shapes.json> <bench_out_dir> <out.csv>", file=sys.stderr)
        raise SystemExit(2)
    shapes_path, bench_dir, out_csv = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

    shapes = json.loads(shapes_path.read_text())
    native = {}
    for r in shapes:
        key = (r["k"], r["m"], r["n"])
        if key not in native or r["native_median_us"] < native[key][0]:
            native[key] = (r["native_median_us"], r["native_name"])

    rows = []
    for path in sorted(glob.glob(str(bench_dir / "k*_m*_n*.log"))):
        tag = Path(path).stem
        m3 = re.match(r"k(\d+)_m(\d+)_n(\d+)", tag)
        k, m, n = int(m3.group(1)), int(m3.group(2)), int(m3.group(3))
        text = Path(path).read_text()
        heur_us = all_us_min = None
        for section in text.split("=== "):
            lines = section.strip().splitlines()
            if not lines:
                continue
            header = lines[0].strip().rstrip(" =")
            us_vals = []
            for line in lines:
                parts = line.strip().split(",")
                if len(parts) > 3 and parts[0] in ("N", "T"):
                    try:
                        us_vals.append(float(parts[-1]))
                    except ValueError:
                        pass
            if header.endswith("heuristic") and us_vals:
                heur_us = min(us_vals)
            elif header.endswith("all") and us_vals:
                all_us_min = min(us_vals)
        nat_us, nat_name = native.get((k, m, n), (None, None))
        rows.append({
            "k": k, "m": m, "n": n,
            "native_us": nat_us, "native_path": nat_name,
            "hipblaslt_heuristic_us": heur_us,
            "hipblaslt_all_best_us": all_us_min,
            "heuristic_ratio": round(heur_us / nat_us, 3) if heur_us and nat_us else None,
            "all_ratio": round(all_us_min / nat_us, 3) if all_us_min and nat_us else None,
        })

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    wins = [r for r in rows if r["all_ratio"] and r["all_ratio"] < 1.0]
    print(f"{len(rows)} shapes; {len(wins)} where hipblaslt --algo_method all beats native")
    print(f"-> {out_csv}")


if __name__ == "__main__":
    main()
