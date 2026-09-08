"""HI172: repeatable, interleaved-control kernel-level trace, native vs replay.

See README.md for the question this answers. Reuses HI132's own primitives
(rocprofv3 command-prefix builder, kernel-trace CSV parser, ServerRunner's
command_prefix hook) -- no new profiler, per dev-gpt-agent's explicit
instruction (req_714d201cc4b64f87).

Usage (on the build server, from a checkout with tools/ on PYTHONPATH):
    python3 run.py --config config.json --passes 5 --out-dir /path/to/artifacts

config.json fields: model, common_args (list), env (dict, base visibility),
native_binary, replay_binary, replay_cache, prompt_repeat_sentence,
prompt_repeat_count, n_predict.

Design: PASS = one profiled native completion + one profiled replay
completion (rocprofv3-wrapped) with an UNPROFILED control completion on
EACH arm run before and after the profiled pair, matching HI132's own
interleaved-control-around-every-profiler-pass discipline -- an unprofiled
control's wall-clock time drifting >5% across the run means the whole
pass set is flagged environment_stable=false, exactly like HI132's own
gate, rather than silently trusting a single noisy sample.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

_TOOLS_ROOT = Path(__file__).resolve().parents[3] / "tools"
if str(_TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOLS_ROOT))

from bigcherry.profiling import rocprof
from bigcherry.tuning.server_runner import ServerRunner

ENVIRONMENT_DRIFT_THRESHOLD_PCT = 5.0  # same threshold HI132 uses


def build_prompt(sentence: str, repeat: int) -> str:
    return sentence * repeat


def run_unprofiled_control(*, binary, model, common_args, env, prompt, n_predict, label, log_dir) -> float:
    runner = ServerRunner(
        binary=binary, model=model, extra_args=common_args,
        env_overrides=env, log_path=log_dir / f"{label}-control.log",
    )
    with runner:
        result = runner.run_completion(prompt, n_predict=n_predict, timeout_s=180)
        timings = result.get("timings", {})
        return float(timings.get("prompt_ms", 0.0))


def run_profiled_pass(*, binary, model, common_args, env, prompt, n_predict, label, out_dir):
    prefix = rocprof.rocprofv3_command_prefix(output_dir=out_dir, label=label)
    runner = ServerRunner(
        binary=binary, model=model, extra_args=common_args,
        env_overrides=env, command_prefix=prefix,
        log_path=out_dir / f"{label}.log",
    )
    with runner:
        result = runner.run_completion(prompt, n_predict=n_predict, timeout_s=180)
    csv_path = out_dir / f"{label}_kernel_trace.csv"
    stats = rocprof.parse_kernel_trace(csv_path)
    return result.get("timings", {}), {s.name: s for s in stats}


def aggregate(passes: list[dict[str, rocprof.KernelStat]]) -> dict[str, dict]:
    """Merge per-pass exact-kernel-symbol stats across all passes into one
    calls/mean/p95/total summary per kernel name."""
    all_names = set()
    for p in passes:
        all_names.update(p)
    out = {}
    for name in all_names:
        durations_all = []
        calls_total = 0
        for p in passes:
            stat = p.get(name)
            if stat is None:
                continue
            calls_total += stat.calls
            # mean/p95 per pass are already aggregated; reconstruct an
            # approximate pooled mean weighted by call count (exact
            # per-dispatch durations are not retained past parse_kernel_trace
            # -- pooling means is the best available without re-parsing raw
            # rows here, sufficient for this diagnostic's purpose).
            durations_all.extend([stat.mean_us] * stat.calls)
        if not durations_all:
            continue
        out[name] = {
            "calls_total": calls_total,
            "pooled_mean_us": statistics.fmean(durations_all),
            "passes_present": sum(1 for p in passes if name in p),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt = build_prompt(config["prompt_repeat_sentence"], config["prompt_repeat_count"])
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

    control_times: dict[str, list[float]] = {"native": [], "replay": []}
    pass_stats: dict[str, list[dict]] = {"native": [], "replay": []}

    for pass_idx in range(1, args.passes + 1):
        # HI171/TEST.md's own "never run arms in a fixed order" rule: a
        # fixed native-then-replay order every pass confounds arm identity
        # with GPU thermal/clock-state drift within each pass (whichever
        # arm runs SECOND inherits any warm-up drift from the arm that ran
        # immediately before it). Alternate which arm goes first each pass.
        ordered_arms = list(arms.items())
        if pass_idx % 2 == 0:
            ordered_arms = list(reversed(ordered_arms))
        for arm_name, arm in ordered_arms:
            print(f"[hi172] pass {pass_idx}/{args.passes} {arm_name}: control (pre)", flush=True)
            control_times[arm_name].append(run_unprofiled_control(
                binary=arm["binary"], model=model, common_args=common_args,
                env=arm["env"], prompt=prompt, n_predict=n_predict,
                label=f"{arm_name}-p{pass_idx}-pre", log_dir=out_dir,
            ))

            print(f"[hi172] pass {pass_idx}/{args.passes} {arm_name}: profiled", flush=True)
            pass_dir = out_dir / f"{arm_name}-p{pass_idx}"
            timings, stats = run_profiled_pass(
                binary=arm["binary"], model=model, common_args=common_args,
                env=arm["env"], prompt=prompt, n_predict=n_predict,
                label=arm_name, out_dir=pass_dir,
            )
            pass_stats[arm_name].append(stats)
            (pass_dir / "completion_timings.json").write_text(
                json.dumps(timings, indent=2), encoding="utf-8",
            )

            print(f"[hi172] pass {pass_idx}/{args.passes} {arm_name}: control (post)", flush=True)
            control_times[arm_name].append(run_unprofiled_control(
                binary=arm["binary"], model=model, common_args=common_args,
                env=arm["env"], prompt=prompt, n_predict=n_predict,
                label=f"{arm_name}-p{pass_idx}-post", log_dir=out_dir,
            ))

    summary: dict = {"passes": args.passes, "arms": {}}
    environment_stable = True
    for arm_name in arms:
        times = control_times[arm_name]
        spread_pct = (max(times) - min(times)) / statistics.fmean(times) * 100 if times else 0.0
        if spread_pct > ENVIRONMENT_DRIFT_THRESHOLD_PCT:
            environment_stable = False
        summary["arms"][arm_name] = {
            "control_prompt_ms": times,
            "control_spread_pct": spread_pct,
            "kernel_stats": aggregate(pass_stats[arm_name]),
        }
    summary["environment_stable"] = environment_stable

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[hi172] environment_stable={environment_stable}")
    print(f"[hi172] wrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
