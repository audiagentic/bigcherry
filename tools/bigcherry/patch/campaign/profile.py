"""PVPS10: kernel-coverage profile of a patch (control vs subject).

Validation evidence proves a patch path ran once (activation marker). This
profiles what actually executes: which kernels, how many calls and what share
of GPU time, on the control (validated BC) and the subject (validated BC +
patch) for one workload.

The CLI supports a two-phase form for queue schedulers:

    # CPU/build preparation under a shared-build lock
    python -m bigcherry.patch.campaign.profile ... \
        --prepare-only --prepared-manifest prepared.json

    # GPU profiling later, with no CMake/Ninja mutation
    python -m bigcherry.patch.campaign.profile ... \
        --prepared-manifest prepared.json

The prepared manifest binds the exact control/subject binary bytes and all
build-affecting selector inputs.  The second phase re-hashes both binaries and
fails closed on drift.  Invocations without --prepared-manifest retain the
legacy build+profile behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from bigcherry.patch.campaign.build import _atomic_write_json, _print, build_tree
from bigcherry.patch.campaign.scaffold import _materialize_scaffold_sources

WORKLOADS = {
    "prefill": ("-p", "512", "-n", "0", "-r", "3"),
    "decode": ("-p", "0", "-n", "128", "-r", "3"),
}
_PREPARED_SCHEMA = "bigcherry.profile-prepared.v1"


def _sha256(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            state.update(chunk)
    return state.hexdigest()


def build_pair(*, patch_id: str, arch: str, hip_path: Path, worktree_root: Path, build_root: Path,
               common_patches: tuple[str, ...], baseline_source: str,
               allow_rejected: bool = False) -> dict[str, Path]:
    from bigcherry.core import config as campaign_config
    from bigcherry.core import paths as bc_paths

    cfg = campaign_config.load(bc_paths.RECIPES)
    sources = _materialize_scaffold_sources(
        patch_id=patch_id, base_ref=cfg.pinned, baseline_source=baseline_source,
        common_patches=common_patches, worktree_root=worktree_root, allow_rejected=allow_rejected,
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


def _prepared_identity(opts: argparse.Namespace) -> dict[str, object]:
    return {
        "patch": opts.patch,
        "arch": opts.arch,
        "hip_path": str(opts.hip_path.resolve()),
        "build_root": str(opts.build_root.resolve()),
        "baseline_source": opts.baseline_source,
        "common_patches": list(opts.common_patches),
        "allow_rejected": bool(opts.allow_rejected),
    }


def _write_prepared_manifest(path: Path, opts: argparse.Namespace, bins: dict[str, Path]) -> None:
    root = opts.build_root.resolve()
    binary_rows: dict[str, dict[str, object]] = {}
    for role in ("control", "subject"):
        binary = bins.get(role)
        if binary is None or not binary.is_file():
            raise RuntimeError(f"prepared profile {role} binary is missing: {binary}")
        resolved = binary.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                f"prepared profile {role} binary escapes build root: {resolved}"
            ) from exc
        binary_rows[role] = {
            "path": str(resolved),
            "size": resolved.stat().st_size,
            "sha256": _sha256(resolved),
        }
    document = {
        "schema": _PREPARED_SCHEMA,
        **_prepared_identity(opts),
        "binaries": binary_rows,
    }
    _atomic_write_json(path, document)


def _load_prepared_manifest(path: Path, opts: argparse.Namespace) -> dict[str, Path]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"prepared profile manifest unavailable/invalid: {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema") != _PREPARED_SCHEMA:
        raise RuntimeError(f"unsupported prepared profile manifest: {path}")
    expected = _prepared_identity(opts)
    for key, value in expected.items():
        if document.get(key) != value:
            raise RuntimeError(
                f"prepared profile identity mismatch for {key}: "
                f"{document.get(key)!r} != {value!r}"
            )
    rows = document.get("binaries")
    if not isinstance(rows, dict):
        raise RuntimeError("prepared profile manifest has no binaries object")
    root = opts.build_root.resolve()
    bins: dict[str, Path] = {}
    for role in ("control", "subject"):
        row = rows.get(role)
        if not isinstance(row, dict):
            raise RuntimeError(f"prepared profile manifest missing {role} binary")
        binary = Path(str(row.get("path", ""))).resolve()
        try:
            binary.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                f"prepared profile {role} binary escapes build root: {binary}"
            ) from exc
        if not binary.is_file():
            raise RuntimeError(f"prepared profile {role} binary no longer exists: {binary}")
        if binary.stat().st_size != int(row.get("size", -1)):
            raise RuntimeError(f"prepared profile {role} binary size drifted: {binary}")
        if _sha256(binary) != row.get("sha256"):
            raise RuntimeError(f"prepared profile {role} binary bytes drifted: {binary}")
        bins[role] = binary
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
    # rocprofv3 can crash during atexit after valid trace generation. Remove a
    # stale report first and let kernel-fraction's own successful parse prove a
    # complete usable trace rather than trusting the wrapper exit status.
    report.unlink(missing_ok=True)
    kf_rc: int | None = None
    if traces:
        with (out / "kernel-fraction.txt").open("w", encoding="utf-8") as summary:
            kf_rc = subprocess.run([sys.executable, "-m", "bigcherry", "kernel-fraction", "--phase", workload,
                                    "--output", str(report), *map(str, traces)],
                                   stdout=summary, stderr=subprocess.STDOUT, env=env, check=False).returncode
    report_ok = kf_rc == 0 and report.is_file()
    return {"returncode": rc, "traces": [str(t) for t in traces],
            "report": str(report) if report_ok else None,
            "wrapper_crashed_after_measurement": rc != 0 and bool(traces) and report_ok}


def _parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--allow-rejected", action="store_true",
                        help="admit an explicitly named rejected/superseded patch (re-examination)")
    parser.add_argument("--prepare-only", action="store_true",
                        help="build and bind binaries, write --prepared-manifest, then exit")
    parser.add_argument("--prepared-manifest", type=Path,
                        help="immutable build handoff used to separate shared build prep from GPU profiling")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    opts = parser.parse_args(argv)
    if opts.prepare_only and opts.prepared_manifest is None:
        parser.error("--prepare-only requires --prepared-manifest")

    if opts.prepared_manifest is not None and opts.prepared_manifest.is_file() and not opts.prepare_only:
        bins = _load_prepared_manifest(opts.prepared_manifest, opts)
        _print(f"reusing verified prepared profile binaries: {opts.prepared_manifest}")
    else:
        bins = build_pair(patch_id=opts.patch, arch=opts.arch, hip_path=opts.hip_path,
                          worktree_root=opts.worktree_root, build_root=opts.build_root,
                          common_patches=opts.common_patches, baseline_source=opts.baseline_source,
                          allow_rejected=opts.allow_rejected)
        if opts.prepared_manifest is not None:
            _write_prepared_manifest(opts.prepared_manifest, opts, bins)
            _print(f"prepared profile binaries: {opts.prepared_manifest}")

    if opts.prepare_only:
        return 0

    env = dict(os.environ)
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env["HIP_VISIBLE_DEVICES"] = opts.device
    env["BIGCHERRY_PATCH_TRACE"] = "1"
    for item in opts.env:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise RuntimeError(f"--env must be K=V, got {item!r}")
        env[key] = value
    result = {"patch": opts.patch, "arch": opts.arch, "workload": opts.workload, "model": str(opts.model),
              "env": opts.env, "prepared_manifest": str(opts.prepared_manifest) if opts.prepared_manifest else None,
              "binaries": {r: str(b) for r, b in bins.items()}, "arms": {}}
    for role, binary in bins.items():
        _print(f"profiling {role}: {binary}")
        result["arms"][role] = profile_arm(binary=binary, model=opts.model, workload=opts.workload,
                                           out=opts.out / role, env=env)
    opts.out.mkdir(parents=True, exist_ok=True)
    (opts.out / "profile.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    ok = all(a["report"] for a in result["arms"].values())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
