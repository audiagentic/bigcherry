"""VTP05 resumable campaign for Qwen3.8-27B-shaped Q6/Q8 kernel probes.

The default only prints a card-free matrix. 'prepare' validates existing
same-build plugins and writes a frozen manifest OUTSIDE the checkout.
'next' lists one pending invocation; 'run-next --run' executes exactly one
case through xtx27_quant_kernel.py's exclusive GPU lease. Never runs a
model, silently skips a case or automatically loops over the matrix.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import xtx27_quant_kernel as q

CAMPAIGN_ROOT = q.SCRATCH / "campaigns"
TIERS = ("pilot", "decode", "placement", "prefill", "r9700-local")
ROLES = tuple(q.ARCH)
QUANTS = ("iq6r", "w8a8")
PROJECTIONS = ("gate_up", "down")


def entries(tier: str) -> list[dict]:
    """Strictly bounded, deterministic screening ladder, no GPU interaction."""
    if tier not in TIERS:
        raise ValueError("unknown tier")
    entries_out: list[dict] = []
    if tier == "pilot":
        candidates, rows, roles = ("capacity",), (4,), ROLES
        parts = ("physical",)
    elif tier == "decode":
        candidates, rows, roles = ("capacity",), (1, 8), ROLES
        parts = ("physical",)
    elif tier == "placement":
        candidates, rows, roles = q.CANDIDATE_NAMES, (4, 8), ROLES
        parts = ("physical",)
    elif tier == "prefill":
        candidates, rows, roles = ("capacity", "bandwidth"), (64, 512), ROLES
        parts = ("physical",)
    else:
        candidates, rows, roles = ("capacity", "bandwidth"), (4, 64), ("r9700",)
        parts = ("physical", "logical2", "logical3")

    # Prioritise Q6 decode, then Q8, and record physical identity in case id.
    for quant in QUANTS:
        for candidate in candidates:
            for m in rows:
                for gpu in roles:
                    for projection in PROJECTIONS:
                        for part in parts:
                            geo = q.geometry(candidate, gpu, projection, part)
                            cold = 256 if quant == "iq6r" else 0
                            case_id = "-".join(
                                (quant, candidate, gpu, projection, part, f"m{m}", f"c{cold}")
                            )
                            item = {
                                "case_id": case_id,
                                "quant": quant, "candidate": candidate, "gpu": gpu,
                                "projection": projection, "part": part, "m": m,
                                "iters": 3 if m <= 64 else 2,
                                "cold_mib": cold,
                                "geometry": geo, "selected_kernel": q.selected_row(quant, m),
                            }
                            q.command(Path("selftest"), Path("plugin"), quant=quant,
                                      geo=geo, m=m, iters=item["iters"], cold_mib=cold)
                            entries_out.append(item)
    if len({x["case_id"] for x in entries_out}) != len(entries_out):
        raise ValueError("duplicate case ids")
    return entries_out


def cmake_field(cache: Path, key: str) -> str:
    if not cache.is_file():
        raise ValueError(f"no build-target evidence: {cache}")
    matches = re.findall(r"^" + re.escape(key) + r":[^=\n]+=(.*)$",
                         cache.read_text(), re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"missing/ambiguous target {key}: {cache}")
    return matches[0].strip()


def preflight(build_dir: Path) -> dict:
    build = build_dir.resolve()
    r4 = cmake_field(build / "CMakeCache.txt", "RAD_GPU_TARGETS")
    r3 = cmake_field(build / "components/libr3/CMakeCache.txt", "R3_TARGETS")
    if "gfx1201" not in re.split(r"[;,: ]+", r4) or "gfx1100" not in re.split(r"[;,: ]+", r3):
        raise ValueError("prebuilt gfx1201/gfx1100 architecture targets unproven or mismatched")
    src = cmake_field(build / "components/libr3/CMakeCache.txt", "RADIANCE_SRC")
    paths = {}
    for quant in QUANTS:
        for gpu in ROLES:
            exe, plugin = q.build_paths(build, gpu, quant)
            paths[f"{quant}:{gpu}"] = {
                "binary_sha256": q.sha256(exe),
                "plugin_sha256": q.sha256(plugin),
            }
    metadata = build / "bigcherry-build-metadata-radiance.json"
    return {
        "build_dir": str(build),
        "build_targets": {"RAD_GPU_TARGETS": r4, "R3_TARGETS": r3,
                          "RADIANCE_SRC": src},
        "build_metadata_sha256": q.sha256(metadata) if metadata.is_file() else None,
        "objects": paths,
    }


def make_manifest(tier: str, build_dir: Path) -> dict:
    provenance = preflight(build_dir)
    return {
        "schema": 1,
        "kind": "VTP05_TIMING_ONLY_CAMPAIGN_NOT_27B_QUALIFICATION",
        "created_utc": datetime.now(UTC).isoformat(),
        "tier": tier, "checkout": subprocess.run(
            ["git", "-C", str(q.ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip(),
        "runner_sha256": q.sha256(Path(q.__file__)),
        "campaign_sha256": q.sha256(Path(__file__)),
        "provenance": provenance,
        "cases": entries(tier),
    }


def manifest_at(campaign_dir: Path) -> dict:
    doc = json.loads((campaign_dir / "campaign.json").read_text())
    if doc.get("schema") != 1 or doc.get("tier") not in TIERS:
        raise ValueError("unsupported campaign manifest")
    if not isinstance(doc.get("cases"), list) or doc["cases"] != entries(doc["tier"]):
        raise ValueError("campaign case manifest drift")
    return doc


def validate_frozen(doc: dict) -> None:
    if doc["runner_sha256"] != q.sha256(Path(q.__file__)):
        raise ValueError("quant runner changed: create a fresh campaign")
    if doc["campaign_sha256"] != q.sha256(Path(__file__)):
        raise ValueError("campaign runner changed: create a fresh campaign")
    if doc["provenance"] != preflight(Path(doc["provenance"]["build_dir"])):
        raise ValueError("prebuilt kernel artifacts changed: refuse mixed ABI evidence")


def receipt_matches(item: dict, receipt: dict, doc: dict) -> bool:
    key = f'{item["quant"]}:{item["gpu"]}'
    target = doc["provenance"]["objects"][key]
    return (
        receipt.get("status") == "TIMING_OBTAINED_NUMERICAL_UNCHECKED"
        and receipt.get("geometry") == item["geometry"]
        and receipt.get("m") == item["m"]
        and receipt.get("quant") == item["quant"]
        and receipt.get("cold_mib") == item["cold_mib"]
        and receipt.get("selected_kernel") == item["selected_kernel"]
        and receipt.get("binary_sha256") == target["binary_sha256"]
        and receipt.get("plugin_sha256") == target["plugin_sha256"]
        and isinstance(receipt.get("device_us"), (int, float))
        and 0 < receipt["device_us"] < 1e9
    )


def progress(directory: Path, doc: dict) -> tuple[list[dict], list[dict]]:
    done, todo = [], []
    for item in doc["cases"]:
        path = directory / "receipts" / (item["case_id"] + ".json")
        if not path.is_file():
            todo.append(item)
            continue
        data = json.loads(path.read_text())
        if not receipt_matches(item, data, doc):
            raise ValueError(f"inconsistent/bad timing receipt: {path}")
        done.append({"case_id": item["case_id"], "device_us": data["device_us"],
                     "numerical_oracle": data["numerical_oracle"]})
    return done, todo



def ffn_lower_bounds(doc: dict, completed: list[dict]) -> list[dict]:
    """Rank-local ideal FFN bounds, NEVER a whole-model token/s prediction.

    The down projection depends on gate/up completion. Ideal stage time
    cannot be shorter than max(gate/up by GPU) + max(down by GPU).
    """
    by_id = {case["case_id"]: case for case in doc["cases"]}
    times = {}
    for entry in completed:
        case = by_id[entry["case_id"]]
        times[(case["quant"], case["candidate"], case["m"],
               case["gpu"], case["projection"], case["part"])] = entry["device_us"]
    keys = sorted({(x["quant"], x["candidate"], x["m"]) for x in doc["cases"]})
    out = []
    for quant, candidate, m in keys:
        selected = {(gpu, proj): times.get((quant, candidate, m, gpu, proj, "physical"))
                    for gpu in ROLES for proj in PROJECTIONS}
        if any(v is None for v in selected.values()):
            continue
        per_device = {gpu: sum(selected[(gpu, proj)] for proj in PROJECTIONS) for gpu in ROLES}
        ideal = sum(max(selected[(gpu, proj)] for gpu in ROLES) for proj in PROJECTIONS)
        record = {
            "label": "FFN_TWO_PROJECTIONS_IDEAL_BARRIER_LOWER_BOUND_NOT_TOKENS_PER_SECOND",
            "quant": quant, "candidate": candidate, "m": m,
            "cache_mode": "IQR_COLD_FLUSH" if quant == "iq6r" else "W8A8_WARM_ONLY",
            "per_device_sequential_us": per_device,
            "minimum_card_occupancy_us": max(per_device.values()),
            "ideal_two_projection_barrier_us": ideal,
            "exclusions": "no transport, attention, GDN, KV, interleaving or engine scheduler",
        }
        if (quant, candidate, m, "r9700", "gate_up", "logical2") in times:
            logical = [times.get((quant, candidate, m, "r9700", proj, rank))
                       for proj in PROJECTIONS for rank in ("logical2", "logical3")]
            if all(x is not None for x in logical):
                record["r9700_separate_logical_shards_sequential_us"] = sum(logical)
                record["r9700_combined_physical_us"] = per_device["r9700"]
                record["r9700_parallel_streams_unmeasured"] = True
        out.append(record)
    return out


def one_command(doc: dict, case: dict, scratch: Path) -> list[str]:
    return [
        sys.executable, str(Path(q.__file__).resolve()),
        "--phase", "run", "--run",
        "--build-dir", doc["provenance"]["build_dir"],
        "--scratch", str(scratch),
        "--quant", case["quant"], "--candidate", case["candidate"],
        "--gpu", case["gpu"], "--part", case["part"],
        "--projection", case["projection"], "--m", str(case["m"]),
        "--iters", str(case["iters"]), "--cold-mib", str(case["cold_mib"]),
        "--timeout-s", "30",
    ]


def run_one(directory: Path, doc: dict, case: dict) -> int:
    cmd = one_command(doc, case, directory)
    # This inner tool takes/releases the lease; do not spawn it in the background.
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    attempts = directory / "attempts"
    attempts.mkdir(exist_ok=True)
    stem = case["case_id"] + "-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    (attempts / (stem + ".stdout")).write_text(result.stdout)
    (attempts / (stem + ".stderr")).write_text(result.stderr)
    if result.returncode:
        print(f"case failed/lease unavailable; see {attempts / stem}", file=sys.stderr)
        return result.returncode
    found = re.findall(r'"receipt"\s*:\s*"([^"]+)"', result.stdout)
    if len(found) != 1:
        raise ValueError("child returned no unique receipt")
    path = Path(found[0]).resolve()
    if not path.is_relative_to((directory / "runs").resolve()):
        raise ValueError("child receipt outside campaign scratch")
    receipt = json.loads(path.read_text())
    if not receipt_matches(case, receipt, doc):
        raise ValueError("child receipt does not match the expected frozen case")
    receipts = directory / "receipts"
    receipts.mkdir(exist_ok=True)
    target = receipts / (case["case_id"] + ".json")
    if target.exists():
        raise ValueError("refusing to overwrite completed case")
    target.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": "TIMING_ONLY_COMPLETED", "case_id": case["case_id"],
                      "device_us": receipt["device_us"], "receipt": str(target)}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("plan", "prepare", "next", "status", "run-next"),
                   default="plan")
    p.add_argument("--tier", choices=TIERS, default="pilot")
    p.add_argument("--build-dir", type=Path)
    p.add_argument("--campaign-dir", type=Path)
    p.add_argument("--run", action="store_true")
    args = p.parse_args(argv)
    if args.phase == "plan":
        cases = entries(args.tier)
        print(json.dumps({"kind": "CARD_FREE_VTP05_MATRIX", "tier": args.tier,
                          "count": len(cases), "first_four": cases[:4],
                          "last_case": cases[-1], "gpu_used": False,
                          "numerical_oracle": "NOT_PROVEN_TIMING_ONLY"}, indent=2))
        return 0
    if args.phase == "prepare":
        if args.build_dir is None or args.campaign_dir is None:
            p.error("prepare requires --build-dir and --campaign-dir")
        directory = args.campaign_dir.resolve()
        if directory == q.ROOT or directory.is_relative_to(q.ROOT):
            p.error("campaign scratch must be outside the checkout")
        if directory.exists():
            p.error("refusing existing campaign directory: do not mix runs")
        doc = make_manifest(args.tier, args.build_dir)
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "campaign.json").write_text(json.dumps(doc, indent=2) + "\n")
        print(json.dumps({"status": "PREPARED_CARD_FREE", "campaign": str(directory),
                          "cases": len(doc["cases"]), "build": doc["provenance"]["build_dir"]}, indent=2))
        return 0
    if args.campaign_dir is None:
        p.error("next/status/run-next require --campaign-dir")
    directory = args.campaign_dir.resolve()
    doc = manifest_at(directory)
    validate_frozen(doc)
    done, todo = progress(directory, doc)
    info = {"status": "ALL_CASES_TIMED_NOT_NUMERICALLY_QUALIFIED" if not todo
            else "PENDING", "tier": doc["tier"], "timed": len(done),
            "remaining": len(todo), "total": len(doc["cases"])}
    if args.phase == "status":
        print(json.dumps({**info, "completed": done,
                          "ffn_lower_bounds": ffn_lower_bounds(doc, done)}, indent=2))
        return 0
    if not todo:
        print(json.dumps(info, indent=2))
        return 0
    case = todo[0]
    cmd = one_command(doc, case, directory)
    if args.phase == "next":
        print(json.dumps({**info, "next_case": case, "argv": cmd,
                          "requires_exclusive_gpu_lease": True}, indent=2))
        return 0
    if not args.run:
        p.error("run-next requires explicit --run")
    return run_one(directory, doc, case)


if __name__ == "__main__":
    raise SystemExit(main())
