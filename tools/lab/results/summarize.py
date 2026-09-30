"""Summarise finished Brutus lab runs into compact, committable JSON.

Run on the bench host:  python3 summarize.py <runs-dir> <out-dir>
Per run (one file <out-dir>/<run>.json, rewritten each harvest):
  - AB runs (<run>/result/run.json): per-arm metric means, the harness's paired
    comparisons (effect + CI95), MTP draft acceptance per arm, server args/env per arm.
  - KLD runs (<run>/perplexity.log): mean/p99 KLD, same-top-token rate, PPL ratio.
  - Patch campaigns (<run>/**/*-performance.json, *-correctness.json): lane effects
    and correctness verdicts.
Raw logs stay on the host; these summaries are what later analysis reads.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

ACCEPT = re.compile(r"draft acceptance = [0-9.]+ \(\s*(\d+) accepted /\s*(\d+) generated\)")


def ab_summary(run: Path) -> dict | None:
    p = run / "result" / "run.json"
    if not p.is_file():
        return None
    r = json.loads(p.read_text())
    cfg_path = run / "server-config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
    arms: dict[str, dict] = {}
    for row in r.get("runs", []):
        a = arms.setdefault(row["mode"], {"metrics": {}, "acc": [0, 0]})
        for k, v in (row.get("metrics") or {}).items():
            a["metrics"].setdefault(k, []).append(v)
        log = Path(row.get("server_log", ""))
        if log.is_file():
            for acc, gen in ACCEPT.findall(log.read_text(errors="replace")):
                a["acc"][0] += int(acc)
                a["acc"][1] += int(gen)
    out_arms = {}
    for name, a in arms.items():
        out_arms[name] = {
            "n": len(next(iter(a["metrics"].values()), [])),
            "mean": {k: round(statistics.mean(v), 2) for k, v in a["metrics"].items()},
            # pooled accepted/generated, same as tools/lab/ar-accuracy/gates.py
            "acceptance_pct": round(100 * a["acc"][0] / a["acc"][1], 2) if a["acc"][1] else None,
        }
    comps = {
        c: {m: {"effect_pct": round(v["geometric_effect_pct"], 2),
                "ci95": [round(v["ci95_low_pct"], 2), round(v["ci95_high_pct"], 2)]}
            for m, v in ms.items()}
        for c, ms in (r.get("exploratory_comparisons") or {}).items()
    }
    return {
        "kind": "ab",
        "model": cfg.get("model"),
        "server_args": cfg.get("server_args"),
        "arm_config": [{k: arm.get(k) for k in ("name", "binary", "server_args", "environment", "model")}
                       for arm in cfg.get("arms", [])],
        "schedule_seed": r.get("schedule_seed"),
        "arms": out_arms,
        "comparisons": comps,
    }


def kld_summary(run: Path) -> dict | None:
    p = run / "perplexity.log"
    if not p.is_file():
        return None
    t = p.read_text(errors="replace")
    def grab(pat):
        m = re.search(pat, t)
        return float(m.group(1)) if m else None
    s = {
        "kind": "kld",
        "mean_kld": grab(r"Mean\s+KLD:\s+(-?[0-9.]+)"),
        "p99_kld": grab(r"99\.0%\s+KLD:\s+(-?[0-9.]+)"),
        "same_top_pct": grab(r"Same top p:\s+([0-9.]+)"),
        "ppl_ratio": grab(r"Mean\s+PPL\(Q\)/PPL\(base\)\s*:\s*([0-9.]+)"),
    }
    env = run / "env.txt"
    if env.is_file():
        s["env"] = env.read_text(errors="replace").strip().splitlines()
    return s if s["mean_kld"] is not None else None


def campaign_summary(run: Path) -> dict | None:
    perf = sorted(run.glob("**/*-performance.json"))
    corr = sorted(run.glob("**/*-correctness.json"))
    if not perf and not corr:
        return None
    s: dict = {"kind": "campaign", "performance": [], "correctness": []}
    for p in perf:
        d = json.loads(p.read_text())
        s["performance"].append({
            "contract_id": d.get("contract_id"),
            **{lane: {"metric": d[lane].get("metric"), "effect": d[lane].get("effect")}
               for lane in ("positive", "control") if lane in d},
        })
    for p in corr:
        d = json.loads(p.read_text())
        s["correctness"].append({k: d.get(k) for k in ("contract_id", "check", "passed", "detail",
                                                        "subject_marker_ncols", "control_marker_ncols")})
    return s


def main() -> int:
    runs, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for run in sorted(p for p in runs.iterdir() if p.is_dir()):
        s = ab_summary(run) or kld_summary(run) or campaign_summary(run)
        if s is None:
            continue
        s["run"] = run.name
        (out / f"{run.name}.json").write_text(json.dumps(s, indent=1, sort_keys=True) + "\n")
        n += 1
    print(f"summarised {n} runs into {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
