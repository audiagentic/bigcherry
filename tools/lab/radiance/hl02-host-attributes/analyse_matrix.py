#!/usr/bin/env python3
"""Strict offline conversion of HL02 lab JSONL into path-free managed CSV receipts."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics

RESULT = re.compile(
    r"HL02_RESULT device=(\d+) mode=(\S+) direction=(\S+) flags=(\d+) "
    r"mib=(\d+) rounds=(\d+) device_gbps=([\d.]+) wall_gbps=([\d.]+) "
    r"device_ms=([\d.]+) wall_s=([\d.]+) gpu_word_errors=(\d+) status=(\w+)"
)
PUSH = re.compile(
    r"RAD50_AB_COMPARE device=(\d+) bytes=(\d+) method=(\S+) "
    r"baseline_seq_p50_us=([\d.]+) push_seq_p50_us=([\d.]+) "
    r"push_speedup=([\d.]+) delta_us=([\d.]+)"
)


def parse_file(path):
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    if not rows or rows[0].get("kind") != "header" or rows[-1].get("kind") != "complete":
        raise ValueError(f"incomplete experiment: {path.name}")
    return rows


def csv_receipt(path, records):
    if not records:
        raise ValueError(f"no records for {path.name}")
    with path.open("w", newline="") as output:
        w = csv.DictWriter(output, fieldnames=list(records[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    data, backgrounds, comparisons = [], [], []
    for path in sorted(args.input_dir.glob("*.jsonl")):
        if path.name.startswith("push-"):
            for x in parse_file(path):
                if x["kind"] == "comparison":
                    matches = PUSH.findall(x["stdout"])
                    if (len(matches) != 1 or x["returncode"] or
                            "RAD50_AB_COMPLETE" not in x["stdout"] or
                            "RAD50 RESULT test=dmabuf-ab exit=0" not in x["stderr"]):
                        raise ValueError(f"bad RAD50 comparison in {path.name}")
                    dev, bytes_, method, pull, push, speedup, delta = matches[0]
                    comparisons.append(dict(session=path.stem, receiver=dev,
                        primary_action=x["arm"], payload_bytes=bytes_,
                        method=method, baseline_p50_us=pull,
                        push_p50_us=push, speedup=speedup, delta_us=delta))
                elif x["kind"] == "background":
                    matches = RESULT.findall(x["stdout"])
                    if len(matches) != 1 or matches[0][-2:] != ("0", "PASS"):
                        raise ValueError(f"background failed in {path.name}")
                    backgrounds.append(dict(session=path.stem, receiver="serial",
                        primary_action=x["arm"], device_gbps=matches[0][6],
                        errors=matches[0][-2]))
        else:
            rows = parse_file(path)
            if sum(x.get("kind") == "measurement" for x in rows) != 9:
                raise ValueError(f"missing mapped-host arms: {path.name}")
            for x in rows:
                if x["kind"] not in ("measurement", "background"):
                    continue
                matches = RESULT.findall(x["stdout"])
                if len(matches) != 1 or matches[0][-2:] != ("0", "PASS"):
                    raise ValueError(f"invalid GPU verification: {path.name}")
                fields = matches[0]
                if x["kind"] == "measurement":
                    if x["returncode"] != 0:
                        raise ValueError(f"nonzero experiment: {path.name}")
                    data.append(dict(session=path.stem, receiver=x["victim"],
                        direction=x["direction"], primary_action=x["primary_action"],
                        host_mode=x["mode"], hip_flags=fields[3],
                        mib=fields[4], rounds=fields[5],
                        device_gbps=fields[6], wall_gbps=fields[7],
                        gpu_word_errors=fields[-2]))
                else:
                    backgrounds.append(dict(session=path.stem, receiver=x["victim"],
                        primary_action=x["primary_action"], device_gbps=fields[6],
                        errors=fields[-2]))
    # For this experiment, demand three independent sessions for all four
    # receiver/direction matrices and for both RAD50 receiver controls.
    if len(data) != 108 or len(comparisons) != 24 or len(backgrounds) != 36:
        raise ValueError(f"incomplete matrix: {len(data)} measured, "
                         f"{len(comparisons)} RAD50, {len(backgrounds)} background")
    grouped = {}
    for x in data:
        key=(x["receiver"],x["direction"],x["primary_action"],x["host_mode"])
        grouped.setdefault(key,[]).append(float(x["device_gbps"]))
    if len(grouped) != 36 or any(len(v) != 3 for v in grouped.values()):
        raise ValueError("mode/direction pairing is not complete")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_receipt(args.output_dir/"attributes.csv", data)
    csv_receipt(args.output_dir/"backgrounds.csv", backgrounds)
    csv_receipt(args.output_dir/"rad50-serial-push.csv", comparisons)
    for (recv,direction,action,mode),v in sorted(grouped.items()):
        print(f"HL02_MEDIAN receiver={recv} victim={direction} background={action} "
              f"mode={mode} gbps={statistics.median(v):.3f} min={min(v):.3f} max={max(v):.3f}")
    print(f"HL02_COMPLETE measures={len(data)} matched_groups={len(grouped)} "
          f"rad50_sessions={len(comparisons)} background_samples={len(backgrounds)} "
          f"zero_gpu_word_errors=1")


if __name__=="__main__":
    main()
