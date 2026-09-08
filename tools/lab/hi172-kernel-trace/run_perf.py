"""HI172/HI133: CPU call-graph profiling of native vs replay dispatch, real perf.

Attaches `sudo perf record -g` to the real production native/replay servers
(same binaries+cache used throughout HI171/HI172), fires a batch of
completions to accumulate samples, then produces a per-arm call-graph
summary focused on BigCherry's own dispatch-resolution symbols
(ggml_hip_dispatch_resolve, ggml_hip_replay_*) versus everything else.

Requires: a cached sudo timestamp on the target host (run `sudo -v` once
interactively before this script; it will not prompt for a password).

Usage:
    python3 run_perf.py --config config.json --requests 40 --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_TOOLS_ROOT = Path(__file__).resolve().parents[3] / "tools"
if str(_TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOLS_ROOT))

from bigcherry.tuning.server_runner import ServerRunner

PERF_BIN = "/usr/lib/linux-tools-6.8.0-139/perf"

DISPATCH_SYMBOLS = ("ggml_hip_dispatch_resolve", "ggml_hip_replay_lookup",
                    "ggml_hip_replay_record_hit", "ggml_hip_registry_find")


def perf_command_prefix(*, out_dir: Path, label: str) -> tuple[str, ...]:
    """-A (askpass) rather than -n (non-interactive/cached-ticket-only):
    a cached sudo timestamp is per-tty/session and does not carry over to
    a detached subprocess launched from a different session, so -n fails
    here even right after an interactive `sudo -v`. SUDO_ASKPASS must be
    set in the launched process's own environment (see run_arm)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / f"{label}.perf.data"
    return (
        "sudo", "-A", PERF_BIN, "record",
        "-F", "999", "-g", "--call-graph", "dwarf,16384",
        "-o", str(data_path), "--",
    )


def run_arm(*, binary, model, common_args, env, prompt, n_predict, requests, label, out_dir, askpass_path):
    prefix = perf_command_prefix(out_dir=out_dir, label=label)
    runner = ServerRunner(
        binary=binary, model=model, extra_args=common_args,
        env_overrides={**env, "SUDO_ASKPASS": askpass_path}, command_prefix=prefix,
        log_path=out_dir / f"{label}.log",
    )
    with runner:
        for i in range(requests):
            runner.run_completion(prompt, n_predict=n_predict, timeout_s=180)
    return out_dir / f"{label}.perf.data"


def summarize_perf_data(data_path: Path, askpass_path: str) -> dict:
    """perf report --stdio --children -- children percentages give inclusive
    time under each symbol's own subtree, which is what we want for
    attributing time to the dispatch-resolution call path as a whole."""
    import os
    env = {**os.environ, "SUDO_ASKPASS": askpass_path}
    result = subprocess.run(
        ["sudo", "-A", PERF_BIN, "report", "-i", str(data_path),
         "--stdio", "--children", "--percent-limit", "0.01"],
        capture_output=True, text=True, timeout=120, env=env,
    )
    lines = result.stdout.splitlines()
    dispatch_lines = []
    total_samples_line = next((l for l in lines if l.startswith("# Samples:")), "")
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        if any(sym in line for sym in DISPATCH_SYMBOLS):
            dispatch_lines.append(line.strip())
    return {
        "samples_header": total_samples_line.strip(),
        "dispatch_symbol_lines": dispatch_lines,
        "raw_stdout_tail": "\n".join(lines[:40]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--askpass", required=True, help="path to a sudo SUDO_ASKPASS helper script")
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt = config["prompt_repeat_sentence"] * config["prompt_repeat_count"]
    n_predict = config.get("n_predict", 8)
    model = config["model"]
    common_args = config["common_args"]
    base_env = config["env"]

    arms = {
        "native": {
            "binary": config["native_binary"],
            "env": {**base_env, "GGML_HIP_DISPATCH_MODE": "native"},
        },
        "replay": {
            "binary": config["replay_binary"],
            "env": {**base_env, "GGML_HIP_DISPATCH_MODE": "replay",
                    "GGML_HIP_DISPATCH_CACHE": config["replay_cache"]},
        },
    }

    summary = {}
    for arm_name, arm in arms.items():
        print(f"[hi172-perf] {arm_name}: launching under perf, firing {args.requests} requests", flush=True)
        data_path = run_arm(
            binary=arm["binary"], model=model, common_args=common_args,
            env=arm["env"], prompt=prompt, n_predict=n_predict,
            requests=args.requests, label=arm_name, out_dir=out_dir,
            askpass_path=args.askpass,
        )
        print(f"[hi172-perf] {arm_name}: summarizing {data_path}", flush=True)
        summary[arm_name] = summarize_perf_data(data_path, args.askpass)

    (out_dir / "perf_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[hi172-perf] wrote {out_dir / 'perf_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
