"""RHP20 forced-move acceptance, externally leased hardware only.

'prepare', 'verify', and 'kld-config' are card-free. ONLY 'run' starts
an engine or sends requests, and it explicitly requires --lease-held.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
from bigcherry.bench.rhp20_forced import (
    GateError,
    experiment_evidence,
    manifest_evidence,
    receipt_evidence,
    recipe_evidence,
)

LAYOUTS = {
    "flashnext-2xtx-rdna2-tail": "tail",
    "flashnext-2xtx-rdna2-tail-combine": "tail-combine",
}
LAB = Path(__file__).resolve().parent
HERE = LAB / "rhp20"
from bigcherry.core.context import resolve_work_root

DEFAULT_LOGS = resolve_work_root() / "flashnext"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def experiment(layout: str) -> Path:
    return HERE / (LAYOUTS[layout] + ".experiment.json")


def manifest_candidates(directory: Path, layout: str, *, after: float = 0,
                        forced: dict | None = None) -> list[Path]:
    result = []
    for path in directory.glob("manifest-*.json"):
        if path.name.startswith("manifest-measuring-") or path.stat().st_mtime < after:
            continue
        try:
            data = read(path)
            if data["spec"]["intent"]["layout"] != layout or data["spec"]["plan"]["state"] != "measured":
                continue
            if forced is not None and data["spec"].get("experiment") != forced:
                continue
            result.append(path)
        except (OSError, KeyError, ValueError):
            continue
    return sorted(result, key=lambda p: p.stat().st_mtime, reverse=True)


def baseline(args) -> tuple[Path, bool]:
    if args.manifest:
        return args.manifest, False
    matches = manifest_candidates(args.logs, args.layout)
    if matches:
        return matches[0], False
    # Only a provisional format/owner-map witness: the real combine run
    # MUST supply a matching, measured manifest after start, or fail.
    if args.layout.endswith("-combine"):
        parent = args.layout.removesuffix("-combine")
        matches = manifest_candidates(args.logs, parent)
        if matches:
            return matches[0], True
    raise GateError("no recorded measured manifest; --manifest required")


def check(args) -> dict:
    exp = read(experiment(args.layout))
    schedule = experiment_evidence(exp)
    path, provisional = baseline(args)
    doc = read(path)
    proof, recipe_digest = recipe_evidence(Path(doc["spec"]["intent"]["model"]))
    layout = args.layout.removesuffix("-combine") if provisional else args.layout
    owner = manifest_evidence(doc, layout, proof)
    if owner["mtp_depth"] < 1:
        raise GateError("MTP is disabled in recorded manifest")
    return {"manifest": str(path), "provisional_combine": provisional,
            "recipe_sha256": recipe_digest, "owner": owner, "schedule": schedule,
            "experiment": str(experiment(args.layout)),
            "note": "Proposals are NOT commits. Receiver staging and actual migration remain hardware gates."}


def run_command(cmd: list[str], log: Path, *, timeout: int) -> None:
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(cmd, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT,
                                timeout=timeout, check=False)
    if result.returncode:
        raise GateError(f"command exited {result.returncode}: {cmd}; see {log}")


def wait_work(p: subprocess.Popen, logfile, deadline: float, name: str) -> None:
    try:
        p.wait(timeout=max(1, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
        raise GateError(f"{name} exceeded bounded workload runtime")
    finally:
        logfile.close()
    if p.returncode:
        raise GateError(f"{name} returned {p.returncode}")


def find_run_logs(directory: Path, path: Path) -> dict[int, Path]:
    parts = path.name.split("-")
    if len(parts) < 4:
        raise GateError("cannot correlate measured manifest with helper log timestamp")
    stamp = "-".join(parts[1:3])
    paths = {i: directory / f"helper{i}-{stamp}.log" for i in range(3)}
    if any(not p.is_file() for p in paths.values()):
        raise GateError(f"missing matching helper logs for {stamp}")
    return paths


def validate_workload(out: Path) -> dict:
    docs = list((out / "workloads").glob("*/result.json"))
    if len(docs) != 1:
        raise GateError("expected one canonical prefill-doc-44k result")
    doc = read(docs[0])
    # The indexed workload stores per-repetition measured prompt token counts.
    def tokens(node):
        if isinstance(node, dict):
            for key, val in node.items():
                if key == "prompt_tokens" and type(val) is int:
                    yield val
                else:
                    yield from tokens(val)
        elif isinstance(node, list):
            for value in node:
                yield from tokens(value)
    observed = list(tokens(doc))
    if not observed or max(observed) < 42000:
        raise GateError(f"42K prefill missing; observed prompt tokens {observed[:12]}")
    stream = read(out / "stream-bench.json")
    if not stream.get("ok"):
        raise GateError("concurrent six-stream decode benchmark failed")
    return {"prefill_prompt_tokens": observed, "stream_bench_ok": True}


def verify(args) -> dict:
    evidence = check(args)
    if evidence["provisional_combine"]:
        raise GateError("combine manifest not measured: cannot verify this layout")
    if not args.helper_dir:
        raise GateError("--helper-dir required for verification")
    logfiles = {i: args.helper_dir / f"helper{i}.log" for i in range(3)}
    body = {i: file.read_text(errors="replace") for i, file in logfiles.items()}
    identity = read(args.identity) if args.identity else None
    verdict = receipt_evidence(body, combine=args.layout.endswith("-combine"), identity=identity)
    if args.out:
        verdict["workload"] = validate_workload(args.out)
    verdict["owner_evidence"] = evidence
    return verdict


def hardware(args) -> dict:
    if not args.lease_held:
        raise GateError("only the hardware owner may run this: acquire lease externally and pass --lease-held")
    if not args.build or not args.build.is_dir() or not args.out:
        raise GateError("--build (tested directory) and --out required")
    if "BC_WORK_ROOT" not in os.environ or "BC_MODEL_ROOT" not in os.environ:
        raise GateError("source tools/env/bigcherry-env.sh before running")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    pre = check(args)
    (out / "preflight.json").write_text(json.dumps(pre, indent=2) + "\n")
    since = time.time()
    start = ["bash", str(LAB / "launch-flashnext-best.sh"), "start", "--build", str(args.build),
             "--layout", args.layout, "--port", str(args.port),
             "--experiment", str(experiment(args.layout)), "--record-period", "1"]
    if args.tail_budget_mib is not None:
        start += ["--budget", f"gfx1030={args.tail_budget_mib}"]
    stop = ["bash", str(LAB / "launch-flashnext-best.sh"), "stop"]
    launched = False
    try:
        launched = True
        run_command(start, out / "start.log", timeout=args.timeout_s)
        matching = manifest_candidates(args.logs, args.layout, after=since,
                                       forced=read(experiment(args.layout)))
        if len(matching) != 1:
            raise GateError(f"expected exactly one new measured forced manifest, found {len(matching)}")
        live = matching[0]
        model = Path(read(live)["spec"]["intent"]["model"])
        protected, _ = recipe_evidence(model)
        runtime = manifest_evidence(read(live), args.layout, protected)
        if runtime["mtp_depth"] < 1:
            raise GateError("MTP is not enabled in running manifest")
        (out / "live-manifest.json").write_bytes(live.read_bytes())
        url = f"http://127.0.0.1:{args.port}"
        stream_cmd = [sys.executable, "-m", "bigcherry", "stream-bench", "--url", url,
                      "--streams", "6", "--prompt-tokens", "2048", "--max-tokens", "1024",
                      "--reps", "12", "--temperature", "0", "--seed", "7",
                      "--output", str(out / "stream-bench.json")]
        doc_cmd = [sys.executable, "-m", "bigcherry", "workload", "run", "prefill-doc-44k",
                   "--url", url, "--manifest", str(live), "--repo-root", str(REPO),
                   "--runs-root", str(out / "workloads"), "--index", str(out / "index.jsonl")]
        env = dict(os.environ, PYTHONPATH=str(REPO / "tools"))
        end = time.monotonic() + args.timeout_s
        with (out / "stream.log").open("w") as slog, (out / "prefill.log").open("w") as dlog:
            stream = subprocess.Popen(stream_cmd, cwd=REPO, env=env, stdout=slog, stderr=subprocess.STDOUT)
            try:
                doc = subprocess.Popen(doc_cmd, cwd=REPO, env=env, stdout=dlog, stderr=subprocess.STDOUT)
            except Exception:
                stream.kill()
                stream.wait()
                raise
            try:
                for proc, title in ((doc, "prefill-doc-44k"), (stream, "stream-bench")):
                    remaining = max(1, end - time.monotonic())
                    try:
                        proc.wait(timeout=remaining)
                    except subprocess.TimeoutExpired:
                        raise GateError(f"{title} timed out")
                    if proc.returncode:
                        raise GateError(f"{title} exit={proc.returncode}")
            finally:
                for proc in (doc, stream):
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
        (out / "runtime-owner.json").write_text(json.dumps(runtime, indent=2) + "\n")
        paths = find_run_logs(args.logs, live)
    finally:
        # Stop always, even on failed workload/manifest checks. It is never
        # safe to claim acceptance before helper 0 prints its exit receipts.
        if launched:
            run_command(stop, out / "stop.log", timeout=args.timeout_s)
    if not launched:
        raise GateError("launcher did not start")
    helper_dir = out / "helpers"
    helper_dir.mkdir()
    for i, path in paths.items():
        (helper_dir / f"helper{i}.log").write_bytes(path.read_bytes())
    args.helper_dir = helper_dir
    args.manifest = out / "live-manifest.json"
    result = verify(args)
    (out / "acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def quality_config(args) -> dict:
    path = args.out
    if not path:
        raise GateError("--out is required for KLD config")
    config = read(HERE / "kld.template.json")
    for arm in config["arms"]:
        arm["serving"]["layout"] = args.layout
        if "experiment" in arm["serving"]:
            arm["serving"]["experiment"] = str(experiment(args.layout))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")
    return {"kld_config": str(path), "command": f'PYTHONPATH=tools python3 -m bigcherry ab-benchmark --server-config "{path}" --output "{path.parent / "kld"}" --pairs 2'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "verify", "kld-config"])
    parser.add_argument("--layout", choices=sorted(LAYOUTS), required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--logs", type=Path, default=DEFAULT_LOGS)
    parser.add_argument("--helper-dir", type=Path)
    parser.add_argument("--identity", type=Path, help="recorded per-expert contribution hashes before/after a same-ISA move")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--port", type=int, default=36080)
    parser.add_argument("--timeout-s", type=int, default=5400, help="RHP06 KLD envelope; bounds the combined workload")
    parser.add_argument("--lease-held", action="store_true", help="explicit attestation; this script never obtains a lease")
    parser.add_argument("--tail-budget-mib", type=int, help="only to reproduce a measured tail budget")
    args = parser.parse_args()
    try:
        result = (check(args) if args.action == "prepare" else
                  hardware(args) if args.action == "run" else
                  verify(args) if args.action == "verify" else quality_config(args))
    except (GateError, OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        if args.action == "run" and args.out and args.out.is_dir():
            (args.out / "verdict.json").write_text(json.dumps({"status": "FAIL", "reason": str(exc)}, indent=2) + "\n")
        print(f"RHP20 FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "PREPARED" if args.action in ("prepare", "kld-config") else "PASS",
                      "result": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
