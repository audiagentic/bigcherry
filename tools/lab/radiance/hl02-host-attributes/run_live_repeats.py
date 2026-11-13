#!/usr/bin/env python3
"""Bounded RAD50 GPU-live verification repeats on the TP3 receiver cards.

Requires an externally acquired exclusive GPU lease and KFD preflight.
Does not modify the production engine, share results across GPU processes,
or claim a three-way concurrent collective.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess

SUMMARY = re.compile(
    r"RAD50_PARALLEL_SUMMARY device=(\d+) rows=(\d+) bytes=(\d+) "
    r"iterations=(\d+) verified=(\d+) .*?p50_us=([\d.]+) "
    r"p95_us=([\d.]+) p99_us=([\d.]+) max_us=([\d.]+) .*? status=(\d+)"
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--devices", type=int, nargs="+", required=True)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--rows", type=int, default=768)
    args = ap.parse_args()
    if args.iterations < 10 or args.iterations > 200 or args.rows not in (1, 16, 64, 256, 768):
        ap.error("RAD50 permits 10..200 iterations and rows 1,16,64,256,768")
    if len(set(args.devices)) != len(args.devices) or any(d < 0 or d > 3 for d in args.devices):
        ap.error("devices must be distinct and known")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(args.binary.read_bytes()).hexdigest()
    record = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "binary_sha256": digest, "method": "dmabuf-live-parallel",
        "consumer_armed_before_write": True, "dma_buf_sync": True,
        "devices": args.devices, "iterations": args.iterations, "rows": args.rows,
        "results": []
    }
    receipt = args.output_dir/"live-summary.json"
    try:
        for device in args.devices:
            cmd = [str(args.binary), "--test", "dmabuf-live-parallel",
                   "--device", str(device), "--iterations", str(args.iterations),
                   "--rows", str(args.rows), "--lease-confirmed"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            (args.output_dir/f"live-device-{device}.stdout.log").write_text(result.stdout)
            (args.output_dir/f"live-device-{device}.stderr.log").write_text(result.stderr)
            matches = SUMMARY.findall(result.stdout)
            passed = (result.returncode == 0 and len(matches) == 1
                      and "RAD50_PARALLEL_COMPLETE" in result.stdout
                      and "RAD50 RESULT test=dmabuf-live-parallel exit=0" in result.stderr)
            values = matches[0] if matches else None
            if values:
                passed &= (int(values[0]) == device and int(values[1]) == args.rows
                           and int(values[3]) == args.iterations
                           and int(values[4]) == args.iterations
                           and int(values[9]) == 0)
            r = {"device": device, "returncode": result.returncode,
                 "summary_fields": values, "passed": bool(passed)}
            record["results"].append(r)
            receipt.write_text(json.dumps(record, indent=2) + "\n")
            if not passed:
                raise RuntimeError(f"device {device}: incomplete/incorrect RAD50 verification")
    finally:
        receipt.write_text(json.dumps(record, indent=2) + "\n")
    print("HL02_LIVE_REPEATS", json.dumps(record, sort_keys=True))

if __name__ == "__main__":
    main()
