"""PVPS10: kernel-coverage profile of a patch (control vs subject).

Validation evidence proves a patch path ran once (activation marker). This
profiles what actually executes: which kernels, how many calls and what
share of GPU time, on the control (validated BC) and the subject (validated
BC + patch) for one workload, so a verdict can be checked against what the
patch changed and a neutral result on a rarely-exercised path is visible as
such.

Sources and builds are the standard scaffold's own (``campaign.scaffold``):
control/subject are materialized by content hash under the worktree root and
built under ``<build-root>/<source-hash>/{control,validation-subject}``, so a
shared per-arch build root makes the control a no-op build and only the
patch's own tree compiles. The benchmark runs under rocprofv3 kernel tracing
and each trace is summarized with ``bigcherry kernel-fraction``.

    python -m bigcherry.patch.campaign.profile --patch <id> --arch gfx1100 \\
        --device 0 --model m.gguf --workload prefill --hip-path <rocm> \\
        --worktree-root W --build-root B --out O [--common-patches a,b] \\
        [--env K=V ...]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from bigcherry.patch.campaign.build import _print, build_tree
from bigcherry.patch.campaign.scaffold import _materialize_scaffold_sources

WORKLOADS = {
    "prefill": ("-p", "512", "-n", "0", "-r", "3"),
    "decode": ("-p", "0", "-n", "128", "-r", "3"),
}


def build_pair(*, patch_id: str, arch: str, hip_path: Path, worktree_root: Path, build_root: Path,
               common_patches: tuple[str, ...], baseline_source: str) -> dict[str, Path]:
    from bigcherry.core import config as campaign_config
    from bigcherry.core import paths as bc_paths

    cfg = campaign_config.load(bc_paths.RECIPES)
    sources = _materialize_scaffold_sources(
        patch_id=patch_id, base_ref=cfg.pinned, baseline_source=baseline_source,
        common_patches=common_patches, worktree_root=worktree_root,
    )
    exe = ".exe" if sys.platform == "win32" else ""
    bins = {}
    for role, src, name in (("control", sources.control_src, "control"),
                            ("subject", sources.patched_src, "validation-subject")):
        bin_dir = build_tree(
            name=name, hip_path=hip_path, amdgpu_targets=arch, workdir=build_root / src.name,
            targets=["llama-bench"], source=src, extra_cmake_args=[],
        )
        bins[role] = bin_dir / f"llama-bench{exe}"
    return bins


def profile_arm(*, binary: Path, model: Path, workload: str, out: Path, env: dict[str, str]) -> dict[str, object]:
    out.mkdir(parents=True, exist_ok=True)
    trace_dir = out / "trace"
    command = ["rocprofv3", "--kernel-trace", "--output-format", "csv", "-d", str(trace_dir), "-o", "trace",
               "--", str(binary), "-m", str(model), "-ngl", "99", *WORKLOADS[workload]]
    with (out / "bench.log").open("w", encoding="utf-8") as log:
        rc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=False).returncode
    traces = sorted(trace_dir.rglob("*kernel_trace.csv"))
    report = out / "kernel-fraction.json"
    if rc == 0 and traces:
        with (out / "kernel-fraction.txt").open("w", encoding="utf-8") as summary:
            subprocess.run([sys.executable, "-m", "bigcherry", "kernel-fraction", "--phase", workload,
                            "--output", str(report), *map(str, traces)],
                           stdout=summary, stderr=subprocess.STDOUT, env=env, check=False)
    return {"returncode": rc, "traces": [str(t) for t in traces],
            "report": str(report) if report.is_file() else None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bigcherry.patch.campaign.profile", description=__doc__.splitlines()[0])
    parser.add_argument("--patch", required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--device", required=True, help="HIP_VISIBLE_DEVICES for the benchmark")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--workload", required=True, choices=sorted(WORKLOADS))
    parser.add_argument("--hip-path", required=True, type=Path)
    parser.add_argument("--worktree-root", required=True, type=Path)
    parser.add_argument("--build-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--baseline-source", default="bigcherry-tuning")
    parser.add_argument("--common-patches", type=lambda raw: tuple(p for p in raw.split(",") if p), default=())
    parser.add_argument("--env", action="append", default=[], help="K=V for both arms (e.g. a patch opt-in)")
    args = parser.parse_args(argv)

    bins = build_pair(patch_id=args.patch, arch=args.arch, hip_path=args.hip_path,
                      worktree_root=args.worktree_root, build_root=args.build_root,
                      common_patches=args.common_patches, baseline_source=args.baseline_source)
    env = dict(os.environ)
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env["HIP_VISIBLE_DEVICES"] = args.device
    env["BIGCHERRY_PATCH_TRACE"] = "1"
    for item in args.env:
        key, _, value = item.partition("=")
        env[key] = value
    result = {"patch": args.patch, "arch": args.arch, "workload": args.workload, "model": str(args.model),
              "env": args.env, "binaries": {r: str(b) for r, b in bins.items()}, "arms": {}}
    for role, binary in bins.items():
        _print(f"profiling {role}: {binary}")
        result["arms"][role] = profile_arm(binary=binary, model=args.model, workload=args.workload,
                                           out=args.out / role, env=env)
    (args.out / "profile.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    ok = all(a["returncode"] == 0 and a["report"] for a in result["arms"].values())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
