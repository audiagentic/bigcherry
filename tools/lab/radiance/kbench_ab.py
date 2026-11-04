#!/usr/bin/env python3
"""A/B rad-kbench runner and card-free report comparator.

The caller must hold the GPU lease while invoking run; this tool never
acquires/releases it. Timing-only runs DO NOT check correctness.
"""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from lab.radiance import kbench_step

SPEED = "op kernel library geometry device_us host_us min p90 n ci mhz operands notes".split()
CORRECT = "op kernel library domain geometry tuned n verdict rel_l2 tol".split()
ENV_BASE = ("PATH", "HOME", "LD_LIBRARY_PATH", "ROCM_PATH", "HIP_PATH")

def sha256_file(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def report_fixture(path):
    """Require identical single-fixture provenance in standalone comparison."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    matches = [line for line in lines if line.startswith("- fixture: ")]
    if len(matches) != 1 or " case(s) recorded " not in matches[0] or " from " not in matches[0]:
        raise ValueError("report requires one well-formed recorded fixture")
    return matches[0]



def section(text, title, headings):
    lines = text.splitlines()
    indices = [i for i, l in enumerate(lines) if l == "## " + title]
    if len(indices) != 1:
        raise ValueError("missing or duplicated " + title + " section")
    start = indices[0] + 1
    end = next((i for i in range(start, len(lines)) if lines[i].startswith("## ")), len(lines))
    block = lines[start:end]
    def cells(line):
        return [s.strip().strip(chr(96)) for s in line.strip().strip("|").split("|")]
    headers = [i for i, line in enumerate(block) if line.startswith("| op | kernel |")
               and [x.lower().replace(" ", "_").replace("±%", "ci").replace("mhz", "mhz")
                    for x in cells(line)] == headings]
    if len(headers) != 1:
        raise ValueError(title + " table header does not match pinned report format")
    h = headers[0]
    if h + 1 >= len(block) or len(cells(block[h+1])) != len(headings) or any(
        not re.fullmatch(r":?-{3,}:?", s) for s in cells(block[h+1])
    ):
        raise ValueError(title + " table separator malformed")
    result = []
    for line in block[h+2:]:
        if not line.startswith("|"):
            if result:
                break
            continue
        if not line.endswith("|") or len(cells(line)) != len(headings):
            raise ValueError(title + " malformed data row")
        result.append(dict(zip(headings, cells(line))))
    if not result:
        raise ValueError(title + " table empty")
    return result


def number(value, description, positive=False):
    if value == "-":
        return None
    try:
        n = float(value)
    except ValueError as exc:
        raise ValueError("invalid " + description + ": " + value) from exc
    if not math.isfinite(n) or n < 0 or (positive and n == 0):
        raise ValueError("out-of-range " + description + ": " + value)
    return n


def geometry(value):
    tokens = value.split()
    if not tokens or any(t.count("=") != 1 or not all(t.split("=", 1)) for t in tokens):
        raise ValueError("malformed geometry: " + value)
    fields = kbench_step.params_of(value)
    if len(fields) != len(tokens):
        raise ValueError("duplicated geometry key: " + value)
    keys = [k for k in kbench_step.SIZE_KEYS if k in fields]
    if len(keys) > 1:
        raise ValueError("multiple step keys: " + value)
    size_key = keys[0] if keys else ""
    try:
        size = int(fields.pop(size_key)) if size_key else None
    except ValueError as exc:
        raise ValueError("non-integer step: " + value) from exc
    if size is not None and size < 0:
        raise ValueError("negative step size")
    return tuple(sorted(fields.items())), size_key, size


def ident(row):
    return (row["op"], row["kernel"], row["library"], *geometry(row["geometry"]))


def casekey(row):
    # Kernel family is exact selected kernel identity; NO heuristic cross-library
    # name guessing. Any different family remains explicitly unmatched.
    return row["op"], row["kernel"], *geometry(row["geometry"]), row["operands"]


def key_object(k):
    return dict(op=k[0], kernel_family=k[1], geometry=dict(k[2]),
                step_key=k[3], step=k[4], operands=k[5])


def parse_report(path):
    path = Path(path)
    data = path.read_text(encoding="utf-8")
    speed = section(data, "Speed", SPEED)
    correct = section(data, "Correctness", CORRECT)
    by_correct = {}
    for row in correct:
        key = ident(row)
        if key in by_correct:
            raise ValueError("ambiguous tuned/correctness case: " + str(key))
        status = row["verdict"].replace("**", "").replace(" (synth w)", "")
        if status not in ("ok", "timed", "vacuous", "FAIL"):
            raise ValueError("unknown correctness verdict: " + status)
        by_correct[key] = dict(verdict=status,
                               rel_l2=number(row["rel_l2"], "rel_l2"),
                               tolerance=number(row["tol"], "tolerance"))
    # Reuse kbench_step's timing extraction, not its logic cloned here.
    parsed = kbench_step.report_rows(str(path))
    expected = Counter((r["op"], r["kernel"], geometry(r["geometry"])[0],
                        geometry(r["geometry"])[2], float(r["device_us"]))
                       for r in speed if r["device_us"] != "-")
    found = Counter((op, kernel, tuple(sorted(params.items())), size, us)
                    for op, kernel, params, size, us in parsed)
    if expected != found:
        raise ValueError("report incompatible with kbench_step.report_rows")
    cases = {}
    for row in speed:
        source = ident(row)
        if source not in by_correct:
            raise ValueError("speed row lacks correctness verdict: " + str(source))
        key = casekey(row)
        if key in cases:
            raise ValueError("ambiguous duplicate kernel/geometry/operand shape: " + str(key))
        us = number(row["device_us"], "device us", positive=True)
        if us is None and row["host_us"] != "-":
            number(row["host_us"], "host us", positive=True)
        cases[key] = dict(us=us, library=row["library"], **by_correct[source])
    # Failed, skipped and untimed correctness cases may have no Speed row.
    # Keep them visible with an explicitly unknown footprint, never invent a time.
    unreported = set(by_correct) - {ident(r) for r in speed}
    for source in sorted(unreported):
        op, kernel, library, geom, size_key, size = source
        key = (op, kernel, geom, size_key, size, "unknown (not in Speed)")
        if key in cases:
            raise ValueError("ambiguous untimed correctness case: " + str(key))
        cases[key] = dict(us=None, library=library, **by_correct[source])
    if not any(v["us"] is not None for v in cases.values()):
        raise ValueError("no device timings")
    return cases


def margin(record):
    if record["rel_l2"] is None or record["tolerance"] is None:
        return None
    return record["tolerance"] - record["rel_l2"]


def graph_calls(graph, cases, rows, layers, draft_passes):
    instances = kbench_step.graph_instances(str(graph))
    if not instances:
        raise ValueError("empty graph")
    calls = Counter()
    missing = []
    for op, params, names in instances:
        cls = kbench_step.classify(names, layers)
        if cls == "vision":
            continue
        options = [key for key, row in cases.items()
                   if key[0] == op and row["us"] is not None
                   and all(dict(key[2]).get(p) == value for p, value in params.items())
                   and key[4] is not None and key[4] <= rows]
        if not options:
            missing.append(dict(op=op, geometry=params, cls=cls))
            continue
        max_size = max(key[4] for key in options)
        chosen = [key for key in options if key[4] == max_size]
        if len(chosen) != 1:
            raise ValueError("ambiguous graph instance: " + op + " " + str(params))
        calls[chosen[0]] += draft_passes if cls == "mtp" else 1
    return calls, missing


def compare(a, b, again=None, calls=None):
    shared = sorted(a.keys() & b.keys())
    groups = defaultdict(lambda: dict(a_us=0., b_us=0., cases=0, calls=0,
                                      invalid=0, unverified=0, noise_missing=0))
    rows = []
    for k in shared:
        left, right = a[k], b[k]
        us_a, us_b = left["us"], right["us"]
        ratio = us_b / us_a if us_a is not None and us_b is not None else None
        spread = None
        if again is not None and k in again and again[k]["us"] is not None and us_a is not None:
            spread = abs(math.log(again[k]["us"] / us_a))
        valid = (left["verdict"] not in ("FAIL", "vacuous")
                 and right["verdict"] not in ("FAIL", "vacuous")
                 and all(margin(r) is None or margin(r) >= 0 for r in (left, right)))
        checked = left["verdict"] == right["verdict"] == "ok"
        if not valid:
            status = "invalid"
        elif ratio is None:
            status = "untimed"
        elif again is not None and spread is None:
            status = "noise-unavailable"
        elif spread is not None and abs(math.log(ratio)) <= spread:
            status = "indistinguishable"
        elif not checked:
            status = "unverified"
        elif ratio < 1:
            status = "B faster"
        elif ratio > 1:
            status = "A faster"
        else:
            status = "equal"
        weight = (calls.get(k, 0) if calls is not None else 1)
        rows.append(dict(**key_object(k), a_us=us_a, b_us=us_b,
                         ratio_b_over_a=ratio, a_margin=margin(left), b_margin=margin(right),
                         a_verdict=left["verdict"], b_verdict=right["verdict"],
                         noise_log_spread=spread, status=status, calls_per_step=weight))
        g = groups[k[0]]
        if not valid:
            g["invalid"] += 1
        elif ratio is not None and weight:
            g["a_us"] += us_a * weight
            g["b_us"] += us_b * weight
            g["cases"] += 1
            g["calls"] += weight
            g["unverified"] += not checked
            g["noise_missing"] += again is not None and spread is None
    summaries = {op: dict(**g, ratio_b_over_a=(g["b_us"]/g["a_us"] if g["a_us"] else None))
                 for op, g in sorted(groups.items())}
    def unmatched(keys, side):
        return [dict(**key_object(k), us=side[k]["us"], verdict=side[k]["verdict"])
                for k in sorted(keys)]
    return dict(cases=rows, summaries=summaries,
                weighting="graph calls per step" if calls is not None else "equal calls per case",
                only_a=unmatched(a.keys()-b.keys(), a),
                only_b=unmatched(b.keys()-a.keys(), b),
                noise_only_a=unmatched(a.keys()-again.keys(), a) if again is not None else [],
                noise_only_again=unmatched(again.keys()-a.keys(), again) if again is not None else [])


def fmt(x, precision=3):
    return "-" if x is None else f"{x:.{precision}f}"


def markdown(result):
    lines = ["# Kernel A/B report", "",
             "Weighting: " + result["weighting"] + ". B/A < 1 means B is faster.",
             "Timing-only verdicts are unverified. Negative correctness margins or FAIL/vacuous are not wins.",
             "Noise is the per-case absolute log ratio of A-again/A, not a fixed threshold.",
             "", "| op | kernel family | geometry | operand bytes | A us | B us | B/A | A margin | B margin | A verdict | B verdict | noise log spread | status |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---|---|---:|---|"]
    for r in result["cases"]:
        geom = dict(r["geometry"])
        if r["step_key"]:
            geom[r["step_key"]] = str(r["step"])
        display = " ".join(f"{k}={v}" for k, v in sorted(geom.items()))
        lines.append("| " + " | ".join(map(str, [
            r["op"], r["kernel_family"], display, r["operands"], fmt(r["a_us"]),
            fmt(r["b_us"]), fmt(r["ratio_b_over_a"]), fmt(r["a_margin"], 6),
            fmt(r["b_margin"], 6), r["a_verdict"], r["b_verdict"],
            fmt(r["noise_log_spread"], 6), r["status"]])) + " |")
    lines += ["", "## Op-family summary", "",
              "| family | weighted B/A | cases | calls/step | unchecked cases | invalid cases | missing noise |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for op, s in result["summaries"].items():
        lines.append(f"| {op} | {fmt(s['ratio_b_over_a'])} | {s['cases']} | {s['calls']} | "
                     f"{s['unverified']} | {s['invalid']} | {s['noise_missing']} |")
    for key, label in (("only_a","Only A"),("only_b","Only B"),
                       ("noise_only_a","Only A vs A-again"),
                       ("noise_only_again","Only A-again vs A")):
        lines += ["", f"## {label} ({len(result[key])})", ""]
        for r in result[key]:
            lines.append(f"- {r['op']} / {r['kernel_family']} / {r['geometry']} / "
                         f"{r['step_key']}={r['step']} / {r['operands']}: {fmt(r['us'])} us "
                         f"({r['verdict']})")
        if not result[key]:
            lines.append("None.")
    for key, label in (("missing_ops_a", "Missing ops on A"),
                       ("missing_ops_b", "Missing ops on B")):
        if key in result:
            lines += ["", f"## {label} ({len(result[key])})", ""]
            lines.extend("- " + op for op in result[key])
            if not result[key]:
                lines.append("None.")
    if "graph_unmatched" in result:
        lines += ["", f"## Unmatched graph instances ({len(result['graph_unmatched'])})", ""]
        lines.extend("- " + str(r) for r in result["graph_unmatched"])
        if not result["graph_unmatched"]:
            lines.append("None.")
    return "\n".join(lines) + "\n"


def resolve_sides(sides_path):
    sides = json.loads(sides_path.read_text(encoding="utf-8"))
    if not isinstance(sides, dict) or set(sides) != {"a", "b"}:
        raise ValueError("side JSON must have exactly a and b")
    seen = set()
    resolved = {}
    for label in ("a", "b"):
        side = sides[label]
        if (not isinstance(side, dict) or not isinstance(side.get("name"), str)
                or not re.fullmatch(r"[a-zA-Z0-9_-]+", side["name"])
                or side["name"] in seen):
            raise ValueError("sides must have unique safe names")
        name = side["name"]
        seen.add(name)
        if bool(side.get("radiance_home")) == bool(side.get("build_dir")):
            raise ValueError("specify exactly one of radiance_home/build_dir")
        if side.get("build_dir"):
            build = Path(side["build_dir"]).resolve()
            home, binary = build / "radiance_home", build / "bin/rad-kbench"
        else:
            home = Path(side["radiance_home"]).resolve()
            binary = home / "bin/rad-kbench"
        if not home.is_dir() or not binary.is_file():
            raise ValueError("home/binary missing for " + name)
        kernels = side.get("kernels")
        if isinstance(kernels, list) and kernels and all(isinstance(k, str) and k for k in kernels):
            kernels = ":".join(kernels)
        if not isinstance(kernels, str) or not kernels or not all(kernels.split(":")):
            raise ValueError("invalid kernel library list")
        overrides = side.get("env", {})
        if not isinstance(overrides, dict) or any(
            not isinstance(k, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", k)
            or not isinstance(v, str) for k, v in overrides.items()
        ):
            raise ValueError("env must be a mapping of environment strings")
        if "HIP_VISIBLE_DEVICES" in overrides or "RADIANCE_HOME" in overrides:
            raise ValueError("HIP_VISIBLE_DEVICES/RADIANCE_HOME controlled by runner")
        resolved[label] = dict(name=name, kernels=kernels, env=overrides,
                               home=str(home), binary=str(binary),
                               binary_sha256=sha256_file(binary))
    return resolved


def validate_ops(ops):
    if not ops or any(not isinstance(op, str) or not re.fullmatch(
            r"[A-Za-z0-9_][A-Za-z0-9_.+-]*", op) for op in ops):
        raise ValueError("ops must be nonempty safe single path components")
    if len(set(ops)) != len(ops):
        raise ValueError("duplicate --op is not allowed")


def report_path(out, name, op):
    return out / name / (op + ".report.md")


def run_configuration(args, sides, ops, max_cases):
    if not args.fixture.is_file():
        raise ValueError("fixture missing")
    if not args.device:
        raise ValueError("--device required")
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        raise ValueError("--timeout-s must be finite and positive")
    validate_ops(ops)
    if max_cases is not None and max_cases <= 0:
        raise ValueError("--max-cases must be positive")
    return dict(fixture=str(args.fixture.resolve()),
                fixture_sha256=sha256_file(args.fixture),
                ops=ops, sides=sides, repeat_a=getattr(args, "repeat_a", False),
                hip_visible_devices=args.device,
                seed=args.seed, max_cases=max_cases,
                bench_warm_ms=args.bench_warm_ms, bench_reps=args.bench_reps,
                bench_flush_mib=args.bench_flush,
                gpu_lease="caller holds GPU lease; tool never takes/releases it")


def prepare_manifest(args, configuration, paths):
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "run.json"
    if manifest_path.exists():
        if not args.resume:
            raise ValueError("run.json exists; use --resume or choose a clean output")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("configuration") != configuration:
            raise ValueError("run.json configuration differs; refuse mixed resume")
        if not isinstance(manifest.get("executions"), list):
            raise ValueError("run.json executions malformed")
        manifest.setdefault("skipped", [])
    else:
        manifest = dict(configuration=configuration, executions=[], skipped=[])
    if not args.resume and any(path.exists() for path in paths):
        raise ValueError("report destination already exists; use --resume")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def save_manifest(out, manifest):
    (out / "run.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                  encoding="utf-8")


def output_text(value):
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def execute_op(args, side, op, report, manifest, max_cases):
    report.parent.mkdir(parents=True, exist_ok=True)
    cmd = [side["binary"], "--fixture", str(args.fixture.resolve()),
           "--home", side["home"], "--kernels", side["kernels"],
           "--op", op, "--timing-only"]
    for flag, value in (("--seed", args.seed), ("--max-cases", max_cases),
                        ("--bench-warm-ms", args.bench_warm_ms),
                        ("--bench-reps", args.bench_reps),
                        ("--bench-flush", args.bench_flush)):
        if value is not None:
            cmd += [flag, str(value)]
    cmd += ["--report", str(report)]
    env = {key: os.environ[key] for key in ENV_BASE if key in os.environ}
    env.update(side["env"])
    env.update(HIP_VISIBLE_DEVICES=args.device, RADIANCE_HOME=side["home"])
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    timed_out = False
    try:
        completed = subprocess.run(cmd, env=env, capture_output=True,
                                   text=True, check=False, timeout=args.timeout_s)
        exit_code = completed.returncode
        stdout, stderr = completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code, stdout, stderr = None, exc.stdout, exc.stderr
    except OSError as exc:
        exit_code, stdout, stderr = 127, "", str(exc)
    elapsed = time.monotonic() - start
    prefix = report.name.removesuffix(".report.md")
    outlog = report.parent / (prefix + ".stdout.log")
    errlog = report.parent / (prefix + ".stderr.log")
    outlog.write_text(output_text(stdout), encoding="utf-8")
    errlog.write_text(output_text(stderr), encoding="utf-8")
    status = ("timed_out" if timed_out else "ok" if exit_code == 0
              and report.is_file() and report.stat().st_size > 0 else "failed")
    record = dict(name=side["name"], op=op, kernels=side["kernels"],
                  command=cmd, env=env, binary_sha256=side["binary_sha256"],
                  wall_time_s=elapsed, started_at=started_at, exit_code=exit_code,
                  timed_out=timed_out, timeout_s=args.timeout_s, status=status,
                  report=str(report), stdout=str(outlog), stderr=str(errlog))
    manifest["executions"].append(record)
    save_manifest(args.out, manifest)
    # One complete TSV record per finished invocation, flushed for live monitors.
    with (args.out / "progress.log").open("a", encoding="utf-8") as progress:
        code = "timeout" if timed_out else str(exit_code)
        progress.write(f"{side['name']}\t{op}\t{elapsed:.6f}\t{code}\n")
        progress.flush()
        os.fsync(progress.fileno())
    return record


def run_sides(args):
    sides = resolve_sides(args.sides)
    sequence = [(sides[label], op) for label in ("a", "b") for op in args.op]
    if args.repeat_a:
        sequence += [(dict(sides["a"], name=sides["a"]["name"] + ".again"), op)
                     for op in args.op]
    paths = [report_path(args.out, side["name"], op) for side, op in sequence]
    configuration = run_configuration(args, sides, args.op, args.max_cases)
    manifest = prepare_manifest(args, configuration, paths)
    failed = False
    for (side, op), report in zip(sequence, paths):
        if args.resume and report.is_file() and report.stat().st_size > 0:
            manifest["skipped"].append(dict(name=side["name"], op=op,
                                            report=str(report), reason="nonempty report"))
            save_manifest(args.out, manifest)
            continue
        record = execute_op(args, side, op, report, manifest, args.max_cases)
        if record["status"] != "ok":
            failed = True
            print(f"rad-kbench {record['status']} for {side['name']}/{op}; "
                  "consult run.json/logs", file=sys.stderr)
    print(args.out / "run.json")
    return 1 if failed else 0


def probe(args):
    sides = resolve_sides(args.sides)
    side = sides[args.side]
    configuration = run_configuration(args, {args.side: side}, [args.op], 1)
    report = report_path(args.out, side["name"], args.op)
    manifest = prepare_manifest(args, configuration, [report])
    record = execute_op(args, side, args.op, report, manifest, 1)
    count = 0
    if record["status"] == "ok":
        report_fixture(report)
        parse_report(report)
        count = len(section(report.read_text(encoding="utf-8"), "Speed", SPEED))
    print(f"{side['name']}/{args.op}: {record['wall_time_s']:.3f} s, "
          f"{count} Speed rows; status={record['status']}")
    return 0 if record["status"] == "ok" else 1


def load_reports(path):
    directory = path.is_dir()
    paths = sorted(path.glob("*.report.md")) if directory else [path]
    cases, provenances = {}, []
    for item in paths:
        provenances.append(report_fixture(item))
        parsed = parse_report(item)
        if directory:
            op = item.name.removesuffix(".report.md")
            validate_ops([op])
            if {key[0] for key in parsed} != {op}:
                raise ValueError("report op does not match filename: " + str(item))
        duplicate = cases.keys() & parsed.keys()
        if duplicate:
            raise ValueError("duplicate case across reports: " + str(sorted(duplicate)[0]))
        cases.update(parsed)
    return cases, provenances


def manifest_expected_ops(left, right):
    """Use the matching run manifest to expose ops missing from both sides."""
    if left.parent != right.parent:
        return set()
    path = left.parent / "run.json"
    if not path.is_file():
        return set()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    config = manifest.get("configuration", {})
    sides = config.get("sides", {})
    if not isinstance(sides, dict) or set(sides) != {"a", "b"}:
        return set()
    if {left.name, right.name} != {sides["a"]["name"], sides["b"]["name"]}:
        return set()
    ops = config.get("ops")
    validate_ops(ops)
    return set(ops)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="requires GPU lease held by caller")
    p = sub.add_parser("probe", help="measure one op on one side, max-cases=1")
    for parser in (r, p):
        for key in ("sides", "fixture", "out"):
            parser.add_argument("--" + key, required=True, type=Path)
        parser.add_argument("--device", required=True, help="HIP_VISIBLE_DEVICES")
        parser.add_argument("--timeout-s", required=True, type=float,
                            help="per-invocation wall timeout in seconds; no default")
        for key in ("seed", "bench-warm-ms", "bench-reps", "bench-flush"):
            parser.add_argument("--" + key, type=int)
    r.add_argument("--op", action="append", required=True)
    r.add_argument("--max-cases", type=int)
    r.add_argument("--repeat-a", action="store_true")
    r.add_argument("--resume", action="store_true")
    p.add_argument("--side", choices=("a", "b"), required=True)
    p.add_argument("--op", required=True)
    p.set_defaults(resume=False)
    c = sub.add_parser("compare")
    c.add_argument("a", type=Path, help="report file or per-op report directory")
    c.add_argument("b", type=Path, help="report file or per-op report directory")
    c.add_argument("--again", type=Path, help="report file or per-op report directory")
    c.add_argument("--expected-op", action="append", help="include ops absent from both reports")
    c.add_argument("--graph", type=Path)
    c.add_argument("--rows", type=int, default=4)
    c.add_argument("--layers", type=int, default=0)
    c.add_argument("--draft-passes", type=int, default=3)
    c.add_argument("--out", type=Path, required=True, help="prefix (.md and .json)")
    args = ap.parse_args(argv)
    try:
        if args.command == "run":
            return run_sides(args)
        if args.command == "probe":
            return probe(args)
        a, provenances_a = load_reports(args.a)
        b, provenances_b = load_reports(args.b)
        again, provenances_again = load_reports(args.again) if args.again else (None, [])
        provenances = provenances_a + provenances_b + provenances_again
        if not provenances:
            raise ValueError("no reports on either side")
        if len(set(provenances)) != 1:
            raise ValueError("A, B and A-again reports name different fixtures")
        calls, missing = graph_calls(args.graph, a, args.rows, args.layers,
                                     args.draft_passes) if args.graph else (None, [])
        result = compare(a, b, again, calls)
        ops_a, ops_b = {k[0] for k in a}, {k[0] for k in b}
        if args.expected_op:
            validate_ops(args.expected_op)
        expected = set(args.expected_op or []) | manifest_expected_ops(args.a, args.b)
        result["missing_ops_a"] = sorted((ops_b | expected) - ops_a)
        result["missing_ops_b"] = sorted((ops_a | expected) - ops_b)
        if args.graph:
            result["graph_unmatched"] = missing
            result["graph"] = str(args.graph)
        rendered = markdown(result)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.with_suffix(".md").write_text(rendered, encoding="utf-8")
        args.out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n",
                                                  encoding="utf-8")
        print(rendered, end="")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("kbench_ab: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
