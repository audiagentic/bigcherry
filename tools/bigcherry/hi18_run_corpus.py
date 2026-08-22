"""HI18: generate a SPLIT_REDUCE correctness corpus for one real production
reduction signature, drive it through the native test-hip-reduce probe for
all three provider arms, evaluate every case, and write the aggregate
reduce-correctness.jsonl evidence.

This is the orchestration layer that was previously run ad hoc on Brutus;
committing it makes the run reproducible and keeps the evidence-generation
command itself under review, matching HI67's correctness_evidence.py
precedent of "the evaluation policy lives in the repo, not in someone's
shell history."

Usage (on the GPU host, after building test-hip-reduce):

    python3 -m bigcherry.hi18_run_corpus \\
        --probe /path/to/bin/test-hip-reduce \\
        --element-count 8192 --slice-shape 4096,2,1,1 \\
        --topology-key n2:peer1001 --peer-access partial \\
        --devices 0,1 --seeds 1,2,3 \\
        --source-revision <bigcherry HEAD sha> --manifest-hash <build manifest hash> \\
        --out-dir /tmp/hi18-real-sig-corpus \\
        --jsonl /tmp/hi18-real-sig-reduce-correctness.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import reduce_correctness as rc

PROVIDERS = ("rccl", "meta", "auto")


def _run_probe(
    probe_path: Path, *, case_dir: Path, plan: str, devices: str, out_path: Path,
) -> None:
    env = dict(os.environ)
    env["GGML_HIP_REDUCE_PLAN"] = plan
    result = subprocess.run(
        [str(probe_path), "--case", str(case_dir), "--plan", plan,
         "--devices", devices, "--out", str(out_path)],
        env=env, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise rc.CorrectnessError(
            f"test-hip-reduce --plan {plan} --case {case_dir} exited "
            f"{result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", required=True, type=Path)
    ap.add_argument("--element-count", required=True, type=int)
    ap.add_argument("--slice-shape", required=True,
                     help="comma-separated 4 ints, product must equal --element-count")
    ap.add_argument("--topology-key", required=True)
    ap.add_argument("--peer-access", required=True, choices=("full", "partial", "none", "unknown"))
    ap.add_argument("--devices", required=True, help="e.g. 0,1")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--source-revision", required=True)
    ap.add_argument("--manifest-hash", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--jsonl", required=True, type=Path)
    args = ap.parse_args(argv)

    slice_shape = tuple(int(x) for x in args.slice_shape.split(","))
    if len(slice_shape) != 4:
        raise SystemExit(f"--slice-shape must have exactly 4 components, got {slice_shape}")
    device_count = len(args.devices.split(","))
    seeds = [int(s) for s in args.seeds.split(",")]

    signature_key = rc.make_reduction_signature_key(
        element_type="f32", element_count=args.element_count,
        slice_shape=slice_shape, topology_key=args.topology_key,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    per_provider_results: dict[str, list[rc.CaseResult]] = {p: [] for p in PROVIDERS}

    total = len(rc.PATTERNS) * len(seeds)
    done = 0
    for pattern in rc.PATTERNS:
        for seed in seeds:
            case_id = f"hi18-real-sig:{args.topology_key}:{signature_key}:{pattern}:{seed}"
            case_dir = args.out_dir / f"{pattern}-{seed}"
            devices = rc.generate_case(
                seed=seed, pattern=pattern, element_count=args.element_count,
                device_count=device_count,
            )
            manifest_obj = rc.write_case(
                case_dir, case_id=case_id, seed=seed, pattern=pattern,
                reduction_signature_key=signature_key, topology_key=args.topology_key,
                peer_access=args.peer_access, devices=devices, slice_shape=slice_shape,
            )
            manifest = json.loads((case_dir / "case.json").read_text())

            runs: dict[str, rc.ProviderRun] = {}
            for provider in PROVIDERS:
                out_path = case_dir / f"result-{provider}.json"
                _run_probe(
                    args.probe, case_dir=case_dir, plan=provider,
                    devices=args.devices, out_path=out_path,
                )
                runs[provider] = rc.load_probe_run(out_path, manifest=manifest, provider=provider)

            results = rc.evaluate_case(manifest, devices, runs)
            for result in results:
                per_provider_results[result.provider].append(result)
                rows.append(rc.case_result_to_row(
                    result, source_revision=args.source_revision,
                    manifest_hash=args.manifest_hash,
                    reduction_signature_key=signature_key,
                    topology_key=args.topology_key, peer_access=args.peer_access,
                    element_count=args.element_count, seed=seed,
                ))

            done += 1
            print(f"[{done}/{total}] {case_id}: "
                  + ", ".join(f"{p}={'OK' if any(r.case_id == case_id and r.provider == p and r.valid and r.correct for r in results) else 'FAIL'}"
                              for p in PROVIDERS),
                  file=sys.stderr)

    rc.write_reduce_correctness_jsonl(args.jsonl, rows)

    overall_ok = True
    for provider in PROVIDERS:
        agg = rc.aggregate_case_results(signature_key, provider, per_provider_results[provider])
        status = "PASS" if (agg.all_valid and agg.all_correct) else "FAIL"
        if status == "FAIL":
            overall_ok = False
        print(f"{provider}: {status} case_count={agg.case_count} "
              f"worst_nmse={agg.worst_nmse} worst_max_abs={agg.worst_max_abs} "
              f"failing={agg.failing_case_ids}", file=sys.stderr)

    print(f"signature_key={signature_key}", file=sys.stderr)
    print(f"jsonl written: {args.jsonl}", file=sys.stderr)
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
