"""Accuracy gates for lossy AllReduce wires (and MTP work-equivalence in A/B runs).

Usage:
  python3 gates.py kld <runs-dir> <run-name>...        # llama-perplexity --kl-divergence logs
  python3 gates.py acceptance <ab-result-dir>...       # per-arm MTP draft acceptance from server logs

KLD gates (vs an exact-f32 reference, same model/corpus): mean KLD <= 0.001, 99th-percentile
KLD <= 0.01, same top token >= 99.5%. Acceptance gate: every arm's mean draft acceptance within
0.5 percentage points of the first arm, otherwise the arms did different work.
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

GATES = {"mean_kld": 0.001, "p99_kld": 0.01, "same_top_pct": 99.5}


def kld(runs: Path, names: list[str]) -> int:
    failed = 0
    for name in names:
        log = runs / name / "perplexity.log"
        if not log.is_file():
            print(f"{name}: no perplexity.log (run missing or blocked) -> FAIL")
            failed += 1
            continue
        text = log.read_text(errors="replace")
        mean = re.search(r"Mean\s+KLD:\s+([0-9.eE+-]+)", text)
        p99 = re.search(r"99\.0%\s+KLD:\s+([0-9.eE+-]+)", text)
        top = re.search(r"Same top p:\s+([0-9.]+)", text)
        if not (mean and p99 and top):
            print(f"{name}: KLD statistics missing -> FAIL")
            failed += 1
            continue
        vals = {"mean_kld": float(mean[1]), "p99_kld": float(p99[1]), "same_top_pct": float(top[1])}
        ok = vals["mean_kld"] <= GATES["mean_kld"] and vals["p99_kld"] <= GATES["p99_kld"] \
            and vals["same_top_pct"] >= GATES["same_top_pct"]
        failed += not ok
        print(f"{name}: mean KLD {vals['mean_kld']:.6f}  p99 KLD {vals['p99_kld']:.5f}  "
              f"same top {vals['same_top_pct']:.2f}%  -> {'PASS' if ok else 'FAIL'}")
    return 1 if failed else 0


def acceptance(dirs: list[str]) -> int:
    failed = 0
    pat = re.compile(r"draft acceptance = [0-9.]+ \(\s*(\d+) accepted /\s*(\d+) generated\)")
    for d in dirs:
        acc = defaultdict(lambda: [0, 0])
        for log in Path(d).glob("pair-*-*/server.log"):
            arm = log.parent.name.split("-", 2)[2]
            for a, g in pat.findall(log.read_text(errors="replace")):
                acc[arm][0] += int(a)
                acc[arm][1] += int(g)
        arms = {log.parent.name.split("-", 2)[2] for log in Path(d).glob("pair-*-*/server.log")}
        rates = {arm: 100.0 * a / g for arm, (a, g) in sorted(acc.items()) if g}
        if len(arms) < 2 or set(rates) != arms:
            print(f"{d}: acceptance missing for arms {sorted(arms - set(rates))} (need >= 2 arms with data) -> FAIL")
            failed += 1
            continue
        ref = next(iter(rates.values()))
        ok = all(abs(r - ref) <= 0.5 for r in rates.values())
        failed += not ok
        print(f"{d}: " + "  ".join(f"{k} {v:.2f}%" for k, v in rates.items()) + f"  -> {'PASS' if ok else 'FAIL'}")
    return 1 if failed else 0


if __name__ == "__main__":
    if sys.argv[1] == "kld":
        sys.exit(kld(Path(sys.argv[2]), sys.argv[3:]))
    sys.exit(acceptance(sys.argv[2:]))
