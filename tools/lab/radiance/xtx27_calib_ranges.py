"""VTP06 bounded original-BF16 calibration: plan, one range, verify, resume.

No production changes. A plan is card-free and never starts an engine.
A single leased GPU run must explicitly name one planned 1-4 layer
range and an attempt number; failed/partial attempts are never reused.
Passing Hessians validate the collection pipeline, NOT IQ6R quality.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = "VTP06_BF16_RANGE_CAMPAIGN_V1"
MIB = 1024 * 1024
GIB = 1024 * MIB
OUTPUT_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)\.safetensors$")
SUMMARY_RE = re.compile(
    r"calib: (\d+) documents, (\d+) tokens in [0-9.]+ s, (\d+) preemption\(s\)"
)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(4 * MIB), b""):
            digest.update(block)
    return digest.hexdigest()


def mem_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise ValueError("MemAvailable not present")


def sha_audit(path: Path, checkpoint: Path) -> tuple[dict, str]:
    audit = json.loads(path.read_text())
    if (audit.get("status") != "STRUCTURALLY_VALID_FULL_SHARDS_SHA256_FINGERPRINTED"
            or audit.get("payload_integrity") !=
            "FULL_SHARD_SHA256_FINGERPRINTED_NOT_EXTERNALLY_VERIFIED"
            or audit.get("shard_count") != 18
            or len(audit.get("shards", {})) != 18
            or audit.get("total_tensor_payload_bytes") != 55562855904):
        raise ValueError("full BF16 checkpoint identity audit not qualified")
    if sha_file(checkpoint / "config.json") != audit.get("config_sha256"):
        raise ValueError("BF16 config does not match full-shard audit")
    if sha_file(checkpoint / "model.safetensors.index.json") != audit.get("index_sha256"):
        raise ValueError("BF16 index does not match full-shard audit")
    for name, item in audit["shards"].items():
        if not re.fullmatch(r"[0-9a-f]{64}", item.get("full_shard_sha256", "")):
            raise ValueError(f"invalid full-shard identity: {name}")
        if not (checkpoint / name).is_file():
            raise ValueError(f"missing shard: {name}")
        if (checkpoint / name).stat().st_size != item.get("bytes"):
            raise ValueError(f"shard size changed after audit: {name}")
    # Not re-streaming 55.56 GB; explicitly distinguish trust from recheck.
    return audit, sha_file(path)


def read_reference_size(directory: Path) -> tuple[int, dict]:
    if not directory.is_dir():
        raise ValueError("missing measured layer-0 Hessian reference directory")
    sizes = {}
    for file in sorted(directory.glob("*.safetensors")):
        if not file.name.startswith("model.language_model.layers.0."):
            raise ValueError("reference must contain only layer-0 Hessians")
        sizes[file.name] = file.stat().st_size
    if len(sizes) != 4 or min(sizes.values()) <= 1024:
        raise ValueError("expected four real, nonempty layer-0 Hessians")
    return sum(sizes.values()), sizes


def bundle_at(directory: Path, checkpoint: Path) -> tuple[dict, str]:
    doc = json.loads((directory / "bundle.json").read_text())
    if doc.get("schema") != "VTP06_CORPUS_V1":
        raise ValueError("not an admitted VTP06 corpus bundle")
    if doc.get("overview", {}).get("mode") not in ("pilot", "candidate"):
        raise ValueError("corpus mode unknown")
    if Path(doc.get("checkpoint", "")).resolve() != checkpoint.resolve():
        raise ValueError("corpus belongs to another model")
    for name, field in (
        ("calibration.jsonl", "calibration_sha256"),
        ("holdout.jsonl", "holdout_sha256"),
        ("document_provenance.jsonl", "provenance_sha256"),
    ):
        if sha_file(directory / name) != doc.get(field):
            raise ValueError(f"corpus bundle drift: {name}")
    for filename, field in (
        ("tokenizer.json", "tokenizer_sha256"),
        ("tokenizer_config.json", "tokenizer_config_sha256"),
    ):
        if sha_file(checkpoint / filename) != doc.get(field):
            raise ValueError(f"model tokenizer drift: {filename}")
    return doc, sha_file(directory / "bundle.json")


def read_build(build: Path) -> tuple[Path, str]:
    engine = build / "bin/radiance"
    home = build / "radiance_home"
    if not engine.is_file() or not home.is_dir():
        raise ValueError("isolated Radiance build is incomplete")
    return engine, sha_file(engine)


def manifest_at(directory: Path) -> dict:
    doc = json.loads((directory / "campaign.json").read_text())
    if doc.get("schema") != SCHEMA or doc.get("state") != "PLANNED_NO_QUALITY_APPROVAL":
        raise ValueError("campaign schema or state mismatch")
    if doc["layer_count"] != 64 or doc["quality_approved"] or doc["iq6r_conversion_approved"]:
        raise ValueError("unsafe/inconsistent campaign status")
    todo = doc["ranges"]
    if not todo or todo[0][0] != 0 or todo[-1][1] != 64:
        raise ValueError("incomplete layer ranges")
    if any(lo >= hi or hi - lo > 4 for lo, hi in todo):
        raise ValueError("range budget invalid")
    if any(a[1] != b[0] for a, b in itertools.pairwise(todo)):
        raise ValueError("gap or overlap in campaign")
    return doc


def plan(
    bundle: Path, checkpoint: Path, audit_path: Path, build: Path,
    reference: Path, output: Path, work_root: Path, *,
    chunk_size: int, tp: int, devices: str, host_pool_mib: int,
    max_model_len: int = 4096, reserve_gib: int = 16,
) -> dict:
    if not 1 <= chunk_size <= 4:
        raise ValueError("chunk_size must be 1..4")
    if tp not in (1, 2) or (tp == 1 and devices != "2") or (
        tp == 2 and devices != "0,1"
    ):
        raise ValueError("only R9700 TP1 (2) or dual-XTX TP2 (0,1) qualified")
    if not 4096 <= host_pool_mib <= 65536 or not 512 <= max_model_len <= 262144:
        raise ValueError("host pool or context outside supported bounds")
    if not 8 <= reserve_gib <= 256:
        raise ValueError("disk reserve must be 8..256 GiB")
    if not checkpoint.is_dir() or not work_root.is_dir():
        raise ValueError("checkpoint or work-root missing")
    if output.exists() or not output.parent.is_dir():
        raise ValueError("output campaign must not exist; create its parent first")
    if not output.resolve().is_relative_to(work_root.resolve()):
        raise ValueError("campaign output must be inside selected work root")
    audit, audit_hash = sha_audit(audit_path, checkpoint)
    corpus, bundle_hash = bundle_at(bundle, checkpoint)
    cfg = json.loads((checkpoint / "config.json").read_text())
    text_config = cfg.get("text_config", {})
    if (cfg.get("model_type") != "qwen3_5"
            or text_config.get("num_hidden_layers") != 64
            or len(text_config.get("layer_types", [])) != 64):
        raise ValueError("not the original dense Qwen3.8-27B 64-layer graph")
    _engine, exe_hash = read_build(build)
    reference_bytes, reference_files = read_reference_size(reference)
    full_estimate = reference_bytes * 64  # EXTRAPOLATION, not a quota promise.
    disk_needed = math.ceil(full_estimate * 1.25) + reserve_gib * GIB
    available_disk = shutil.disk_usage(output.parent).free
    if available_disk < disk_needed:
        raise ValueError(f"disk reserve fails: {available_disk} < {disk_needed}")
    available_ram = mem_available()
    per_range_peak = math.ceil(reference_bytes * chunk_size * 1.5)
    recommended_ram = host_pool_mib * MIB + per_range_peak + 4 * GIB
    ranges = [[lo, min(lo + chunk_size, 64)] for lo in range(0, 64, chunk_size)]
    doc = {
        "schema": SCHEMA,
        "state": "PLANNED_NO_QUALITY_APPROVAL",
        "quality_approved": False,
        "iq6r_conversion_approved": False,
        "native_token_counts_verified": False,
        "source_payload_rehashed_now": False,
        "work_root": str(work_root.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "audit_receipt": str(audit_path.resolve()),
        "audit_receipt_sha256": audit_hash,
        "audit_full_payload_bytes": audit["total_tensor_payload_bytes"],
        "build": str(build.resolve()),
        "engine_sha256": exe_hash,
        "corpus_bundle": str(bundle.resolve()),
        "corpus_bundle_sha256": bundle_hash,
        "calibration_sha256": corpus["calibration_sha256"],
        "holdout_sha256": corpus["holdout_sha256"],
        "tokenizer_sha256": corpus["tokenizer_sha256"],
        "corpus_mode": corpus["overview"]["mode"],
        "document_count": corpus["overview"]["documents"]["calibration"],
        "layer_count": 64,
        "ranges": ranges,
        "tp": tp,
        "devices": devices,
        "host_pool_mib": host_pool_mib,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": 128,
        "reserve_gib": reserve_gib,
        "layer0_observed_hessian_bytes": reference_bytes,
        "layer0_observed_hessian_file_bytes": reference_files,
        "all_64_layers_projection_is_extrapolated_bytes": full_estimate,
        "disk_requirement_with_25pct_and_reserve_bytes": disk_needed,
        "available_disk_at_plan_bytes": available_disk,
        "available_ram_at_plan_bytes": available_ram,
        "minimum_admission_ram_bytes": recommended_ram,
        "ram_admission_at_plan": "PASS" if available_ram >= recommended_ram
                                  else "BLOCKED_LOW_MEMORY",
        "warning": "No hardware run in plan step. The 64-layer estimate is based "
                   "on one layer, not measured all-layer allocation. Every range "
                   "needs a fresh exclusive GPU lease and RAM check. Pilot Hessians "
                   "do not qualify an IQ6R checkpoint.",
    }
    output.mkdir(exist_ok=False)
    (output / "campaign.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return doc


def pre_run_identity(doc: dict) -> tuple[Path, Path, Path]:
    checkpoint = Path(doc["checkpoint"])
    audit_path = Path(doc["audit_receipt"])
    if sha_file(audit_path) != doc["audit_receipt_sha256"]:
        raise ValueError("BF16 identity receipt drift")
    sha_audit(audit_path, checkpoint)
    bundle = Path(doc["corpus_bundle"])
    if sha_file(bundle / "bundle.json") != doc["corpus_bundle_sha256"]:
        raise ValueError("corpus metadata drift")
    corpus, _ = bundle_at(bundle, checkpoint)
    if corpus["calibration_sha256"] != doc["calibration_sha256"]:
        raise ValueError("calibration corpus drift")
    engine, exe_hash = read_build(Path(doc["build"]))
    if exe_hash != doc["engine_sha256"]:
        raise ValueError("experimental runtime binary drift")
    return checkpoint, bundle / "calibration.jsonl", engine


def check_gpu_lease(purpose: str) -> str:
    if not purpose.startswith("VTP06/") or len(purpose) < 12:
        raise ValueError("lease purpose must be unique and begin VTP06/")
    status = subprocess.run(
        ["agent-lease", "status"], capture_output=True, text=True, check=True, timeout=8
    ).stdout.strip()
    if f'for "{purpose}"' not in status or "expires" not in status:
        raise ValueError(f"exclusive GPU lease purpose not matched: {status}")
    gpu = subprocess.run(
        ["rocm-smi", "--showpids"], capture_output=True, text=True, check=True, timeout=8
    ).stdout
    if "No KFD PIDs currently running" not in gpu:
        raise ValueError("GPU PIDs active; cannot start a calibration range")
    return status


def _stop_child(proc: subprocess.Popen) -> int:
    if proc.poll() is not None:
        return proc.returncode
    proc.send_signal(signal.SIGINT)
    try:
        return proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            return proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            return proc.wait(timeout=10)


def run_range(
    campaign_dir: Path, lo: int, hi: int, attempt: int, *,
    lease_purpose: str, timeout_s: int = 3600,
    min_runtime_ram_gib: int = 4,
) -> dict:
    doc = manifest_at(campaign_dir)
    if [lo, hi] not in doc["ranges"]:
        raise ValueError("range is not in the immutable campaign")
    if not 1 <= attempt <= 99 or not 60 <= timeout_s <= 21600:
        raise ValueError("attempt or timeout outside bounds")
    if not 2 <= min_runtime_ram_gib <= 64:
        raise ValueError("memory floor outside bounds")
    _, corpus, engine = pre_run_identity(doc)
    available = mem_available()
    peak = math.ceil(doc["layer0_observed_hessian_bytes"] * (hi - lo) * 1.5)
    required = doc["host_pool_mib"] * MIB + peak + min_runtime_ram_gib * GIB
    if available < required:
        raise ValueError(f"RAM preflight BLOCKED: available={available}, required={required}")
    disk_free = shutil.disk_usage(campaign_dir).free
    if disk_free < math.ceil(peak * 1.5) + doc["reserve_gib"] * GIB:
        raise ValueError("disk preflight BLOCKED")
    lease = check_gpu_lease(lease_purpose)
    run_dir = campaign_dir / "ranges" / f"{lo:02d}-{hi:02d}" / f"attempt-{attempt:02d}"
    if run_dir.exists():
        raise FileExistsError(f"attempt already exists; never mix or overwrite: {run_dir}")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(exist_ok=False)
    hessian_dir = run_dir / "hessians"
    hessian_dir.mkdir()
    argv = [
        str(engine), "--model", doc["checkpoint"],
        "--radiance-home", str(Path(doc["build"]) / "radiance_home"),
        "--tp", str(doc["tp"]), "--tp-wire", "exact", "--p2p", "off",
        "--embedding-placement", "ram",
        "--host-pool-mib", str(doc["host_pool_mib"]),
        "--placement", "layer_offload", "--max-num-seqs", "1",
        "--max-model-len", str(doc["max_model_len"]),
        "--max-num-batched-tokens", str(doc["max_num_batched_tokens"]),
        "--kv-cache-dtype", "bf16", "--num-speculative-tokens", "0",
        "--calib-corpus", str(corpus), "--calib-dir", str(hessian_dir),
        "--calib-layers", f"{lo}:{hi}",
    ]
    env = os.environ.copy()
    for key in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                "GPU_DEVICE_ORDINAL"):
        env.pop(key, None)
    env["ROCR_VISIBLE_DEVICES"] = doc["devices"]
    receipt = {
        "schema": "VTP06_RANGE_RUN_V1",
        "state": "STARTED",
        "range": [lo, hi],
        "attempt": attempt,
        "lease_status": lease,
        "run_start_utc": datetime.now(UTC).isoformat(),
        "argv": argv,
        "gpu_devices": doc["devices"],
        "engine_sha256": doc["engine_sha256"],
        "calibration_sha256": doc["calibration_sha256"],
        "audit_receipt_sha256": doc["audit_receipt_sha256"],
        "ram_admission_bytes": available,
        "ram_minimum_bytes": required,
        "quality_approved": False,
    }
    receipt_path = run_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    aborted = None
    start = time.monotonic()
    with (run_dir / "engine.log").open("wb") as out:
        child = subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT, env=env)
        receipt["pid"] = child.pid
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        last_lease_check = 0.0
        try:
            while child.poll() is None:
                elapsed = time.monotonic() - start
                if elapsed > timeout_s:
                    aborted = "TIMEOUT"
                    break
                if mem_available() < min_runtime_ram_gib * GIB:
                    aborted = "RAM_FLOOR_CROSSED"
                    break
                # A detached child does not renew its parent's GPU lease. Fail
                # closed if the lease expires/reassigns while it is running.
                if elapsed - last_lease_check >= 20:
                    lease_now = subprocess.run(
                        ["agent-lease", "status"], capture_output=True,
                        text=True, check=True, timeout=8
                    ).stdout
                    if f'for "{lease_purpose}"' not in lease_now or "expires" not in lease_now:
                        aborted = "EXCLUSIVE_LEASE_LOST"
                        break
                    last_lease_check = elapsed
                time.sleep(2)
        except KeyboardInterrupt:
            aborted = "OWNER_INTERRUPTED"
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            receipt["monitor_error"] = str(exc)
            aborted = "SAFETY_MONITOR_FAILED"
        exit_code = _stop_child(child) if aborted else child.wait()
    receipt["elapsed_s"] = round(time.monotonic() - start, 2)
    receipt["exit_code"] = exit_code
    receipt["state"] = aborted or (
        "EXIT_ZERO_UNVERIFIED" if exit_code == 0 else "EXIT_NONZERO"
    )
    receipt["ended_utc"] = datetime.now(UTC).isoformat()
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def scan_f32(path: Path, offset: int, size: int) -> tuple[int, int, str]:
    nonfinite = nonzero = 0
    sha = hashlib.sha256()
    with path.open("rb") as src:
        src.seek(0)
        remaining = offset + size
        while remaining > 0:
            block = src.read(min(4 * MIB, remaining))
            if not block:
                raise ValueError("early EOF in Hessian")
            sha.update(block)
            remaining -= len(block)
            # Header is not numeric.
        src.seek(offset)
        remaining = size
        while remaining:
            data = src.read(min(1 * MIB, remaining))
            if len(data) % 4:
                raise ValueError("bad float alignment")
            words = array.array("I")
            words.frombytes(data)
            if sys.byteorder != "little":
                words.byteswap()
            for bit in words:
                if (bit & 0x7F800000) == 0x7F800000:
                    nonfinite += 1
                elif (bit & 0x7FFFFFFF):
                    nonzero += 1
            remaining -= len(data)
    return nonfinite, nonzero, sha.hexdigest()


def validate_hessian(path: Path, expected_rows: int) -> dict:
    with path.open("rb") as fh:
        len_bytes = fh.read(8)
        if len(len_bytes) != 8:
            raise ValueError(f"truncated safetensors header: {path.name}")
        header_len = struct.unpack("<Q", len_bytes)[0]
        if not 32 <= header_len <= 65536:
            raise ValueError("unsafe safetensors header size")
        header_raw = fh.read(header_len)
        if len(header_raw) != header_len:
            raise ValueError("truncated header")
        header = json.loads(header_raw)
    if set(header) != {"__metadata__", "hessian_upper"}:
        raise ValueError("unexpected safetensors tensor keys")
    md = header["__metadata__"]
    tensor = header["hessian_upper"]
    if md.get("layout") != "upper triangle, row-major, packed":
        raise ValueError("wrong Hessian layout")
    if not isinstance(md.get("linears"), str) or not md["linears"].strip():
        raise ValueError("missing linear identity")
    if int(md["rows"]) != expected_rows or expected_rows < 1:
        raise ValueError("Hessian row count differs from actual calibrated token count")
    k = int(md["k"])
    if k < 128 or k > 32768:
        raise ValueError("Hessian width outside model bounds")
    elements = k * (k + 1) // 2
    if tensor.get("dtype") != "F32" or tensor.get("shape") != [elements]:
        raise ValueError("wrong Hessian dtype or packed dimensions")
    if tensor.get("data_offsets") != [0, elements * 4]:
        raise ValueError("invalid tensor byte offsets")
    offset = 8 + header_len
    size = elements * 4
    if path.stat().st_size != offset + size:
        raise ValueError("Hessian payload length mismatch")
    nonfinite, nonzero, digest = scan_f32(path, offset, size)
    if nonfinite or nonzero == 0:
        raise ValueError("Hessian contains NaN/Inf or is all zero")
    negative_diag = 0
    positive_diag = 0
    with path.open("rb") as stream:
        for j in range(k):
            index = j * k - j * (j - 1) // 2
            stream.seek(offset + index * 4)
            value, = struct.unpack("<f", stream.read(4))
            if not math.isfinite(value):
                raise ValueError("nonfinite Hessian diagonal")
            if value < -1e-5:
                negative_diag += 1
            if value > 0:
                positive_diag += 1
    if negative_diag or positive_diag == 0:
        raise ValueError("Hessian diagonal has negative entries or no positives")
    return {
        "name": path.name, "k": k, "rows": expected_rows, "elements": elements,
        "bytes": offset + size, "sha256": digest, "nonfinite": nonfinite,
        "nonzero": nonzero, "negative_diagonal": negative_diag,
        "positive_diagonal": positive_diag,
        "linears": md["linears"],
    }


def verify_range(campaign_dir: Path, lo: int, hi: int, attempt: int) -> dict:
    doc = manifest_at(campaign_dir)
    if [lo, hi] not in doc["ranges"]:
        raise ValueError("range is not in the planned campaign")
    folder = campaign_dir / "ranges" / f"{lo:02d}-{hi:02d}" / f"attempt-{attempt:02d}"
    receipt = json.loads((folder / "receipt.json").read_text())
    if (receipt.get("schema") != "VTP06_RANGE_RUN_V1"
            or receipt.get("state") != "EXIT_ZERO_UNVERIFIED"
            or receipt.get("exit_code") != 0
            or receipt.get("range") != [lo, hi]
            or receipt.get("calibration_sha256") != doc["calibration_sha256"]
            or receipt.get("engine_sha256") != doc["engine_sha256"]):
        raise ValueError("range receipt not eligible for Hessian validation")
    if (folder / "validation.json").exists():
        raise FileExistsError("validation already exists; no re-certification or overwrite")
    log = (folder / "engine.log").read_text(errors="replace")
    stats = SUMMARY_RE.findall(log)
    if len(stats) != 1:
        raise ValueError("missing/ambiguous Radiance calibration completion summary")
    docs, rows, preemptions = (int(v) for v in stats[0])
    if docs != doc["document_count"] or rows < docs or preemptions:
        raise ValueError("calibration run had wrong document count or preemption")
    files = sorted((folder / "hessians").glob("*.safetensors"))
    if not files:
        raise ValueError("exit-zero produced no Hessians")
    # The original qwen35_bf16 graph declares EXACTLY four taps per layer:
    # input, attention output, MLP gate/up input, and MLP down input.
    # Full-attention and GDN layers have different attention-output names.
    config = json.loads((Path(doc["checkpoint"]) / "config.json").read_text())
    types = config["text_config"]["layer_types"]
    expected = set()
    for layer in range(lo, hi):
        prefix = f"model.language_model.layers.{layer}."
        attention = ("self_attn.o_proj.input" if types[layer] == "full_attention"
                     else "linear_attn.out_proj.input")
        expected.update({
            prefix + "input.safetensors",
            prefix + attention + ".safetensors",
            prefix + "post_attention.input.safetensors",
            prefix + "mlp.down_proj.input.safetensors",
        })
    # CalibFull::site(-1, "lm_head.input") is attached ONLY to a range
    # whose HI equals 64. Missing this head silently breaks GPTQ.
    if hi == 64:
        expected.add("lm_head.input.safetensors")
    actual = {p.name for p in files}
    if actual != expected:
        raise ValueError(
            f"missing or unexpected Hessian tap(s): missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    covered = {layer: 0 for layer in range(lo, hi)}
    checks = []
    for path in files:
        match = OUTPUT_RE.fullmatch(path.name)
        if match:
            covered[int(match.group(1))] += 1
        elif path.name != "lm_head.input.safetensors" or hi != 64:
            raise ValueError(f"unrecognised Hessian file: {path.name}")
        checks.append(validate_hessian(path, rows))
    if any(count != 4 for count in covered.values()):
        raise ValueError(f"one or more layers lack exactly four Hessian taps: {covered}")
    extra = list((folder / "hessians").iterdir())
    if len(extra) != len(files):
        raise ValueError("unexpected files in Hessian output directory")
    out = {
        "schema": "VTP06_VALIDATED_RANGE_V1",
        "status": "HESSIAN_PIPELINE_PASS_NOT_QUALITY_APPROVAL",
        "range": [lo, hi], "attempt": attempt,
        "engine_sha256": doc["engine_sha256"],
        "calibration_sha256": doc["calibration_sha256"],
        "audit_receipt_sha256": doc["audit_receipt_sha256"],
        "documents": docs, "tokens": rows,
        "hessians": checks, "per_layer_file_count": covered,
        "quality_approved": False,
    }
    (folder / "validation.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def campaign_status(directory: Path, *, full_verify: bool = False) -> dict:
    doc = manifest_at(directory)
    work = []
    validated_layers = set()
    for lo, hi in doc["ranges"]:
        base = directory / "ranges" / f"{lo:02d}-{hi:02d}"
        attempts = sorted(base.glob("attempt-*/receipt.json")) if base.is_dir() else []
        verified = []
        for receipt in attempts:
            f = receipt.with_name("validation.json")
            if not f.is_file():
                continue
            validation = json.loads(f.read_text())
            if (validation.get("schema") != "VTP06_VALIDATED_RANGE_V1"
                    or validation.get("status") !=
                    "HESSIAN_PIPELINE_PASS_NOT_QUALITY_APPROVAL"
                    or validation.get("range") != [lo, hi]
                    or validation.get("calibration_sha256") != doc["calibration_sha256"]
                    or validation.get("engine_sha256") != doc["engine_sha256"]):
                raise ValueError("invalid, mismatched range validation")
            if full_verify:
                for item in validation["hessians"]:
                    path = receipt.parent / "hessians" / item["name"]
                    if not path.is_file() or path.stat().st_size != item["bytes"]:
                        raise ValueError(f"missing/changed validated Hessian {path.name}")
                    if sha_file(path) != item["sha256"]:
                        raise ValueError(f"validated Hessian digest changed: {path.name}")
            verified.append(validation)
        if len(verified) > 1:
            raise ValueError("multiple separately validated attempts need reconciliation")
        if verified:
            validated_layers.update(range(lo, hi))
        work.append({"range": [lo, hi], "state": "VERIFIED" if verified else "PENDING",
                     "attempts": len(attempts)})
    return {
        "schema": SCHEMA, "validated_layers": len(validated_layers),
        "total_layers": doc["layer_count"], "ranges": work,
        "complete_hessian_pipeline": len(validated_layers) == doc["layer_count"],
        "iq6r_quality_approved": False,
        "output_bytes_reverified": full_verify,
        "warning": "Even 64/64 Hessians do not prove IQ6R conversion or model quality."
                   " Use --full-verify before conversion to rehash all recorded bytes.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    x = sub.add_parser("plan", help="card-free plan, disk and model provenance preflight")
    for name in ("bundle", "checkpoint", "audit", "build", "reference", "out", "work-root"):
        x.add_argument("--" + name, type=Path, required=True)
    x.add_argument("--chunk-size", type=int, default=2)
    x.add_argument("--tp", type=int, default=1)
    x.add_argument("--devices", default="2")
    x.add_argument("--host-pool-mib", type=int, default=32768)
    x.add_argument("--max-model-len", type=int, default=4096)
    x.add_argument("--reserve-gib", type=int, default=16)
    x = sub.add_parser("run-range", help="one explicit lease-gated GPU attempt, no loop")
    x.add_argument("--campaign", type=Path, required=True)
    x.add_argument("--lo", type=int, required=True)
    x.add_argument("--hi", type=int, required=True)
    x.add_argument("--attempt", type=int, default=1)
    x.add_argument("--lease-purpose", required=True)
    x.add_argument("--timeout-s", type=int, default=3600)
    x.add_argument("--min-runtime-ram-gib", type=int, default=4)
    x = sub.add_parser("verify-range", help="full streamed Hessian integrity gate")
    x.add_argument("--campaign", type=Path, required=True)
    x.add_argument("--lo", type=int, required=True)
    x.add_argument("--hi", type=int, required=True)
    x.add_argument("--attempt", type=int, default=1)
    x = sub.add_parser("status", help="inspect completed range identity and coverage")
    x.add_argument("--campaign", type=Path, required=True)
    x.add_argument("--full-verify", action="store_true",
                   help="rehash every validated Hessian before conversion")
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = plan(args.bundle, args.checkpoint, args.audit, args.build,
                          args.reference, args.out, args.work_root,
                          chunk_size=args.chunk_size, tp=args.tp,
                          devices=args.devices, host_pool_mib=args.host_pool_mib,
                          max_model_len=args.max_model_len,
                          reserve_gib=args.reserve_gib)
        elif args.command == "run-range":
            result = run_range(args.campaign, args.lo, args.hi, args.attempt,
                               lease_purpose=args.lease_purpose,
                               timeout_s=args.timeout_s,
                               min_runtime_ram_gib=args.min_runtime_ram_gib)
        elif args.command == "verify-range":
            result = verify_range(args.campaign, args.lo, args.hi, args.attempt)
        else:
            result = campaign_status(args.campaign, full_verify=args.full_verify)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f"VTP06 range gate FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
