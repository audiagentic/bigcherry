"""Card-free, fail-closed post-conversion RAD2 artifact structure audit.

Run only after the managed converter completed and atomically named the model.
This checks metadata/encoding counts, original recipe and artifact identity.
It does NOT qualify inference, numerics, calibration quality or TP3 serving.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

RECIPE_SHA256 = "0c41614302d448ba61c39f31eb6da1ee4970fcd4fa1a6011ec479d51b64e597f"
REQUIRED_META = ("architecture  qwen35", "format        RAD2", "source quant  w8a8_fwht128")
HEADER_RE = re.compile(r"^\s*weight directory \((\d+) entries")
CODE_RE = re.compile(r"^\s*trellis:codes=(u[4567])\b.*?\s+(\d+)\s+[\d.]+\s+(?:KiB|MiB|GiB)\b")
BY_ENCODING = re.compile(r"^\s*by encoding\s*$")
SECTION_END = re.compile(r"^\s*by what made it\s*$")


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def recipe_codes(text: str) -> Counter[str]:
    kinds = Counter()
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            raise ValueError("malformed recipe line")
        if parts[1] != "gptq":
            continue
        codes = [token.split("=", 1)[1] for token in parts if token.startswith("codes=")]
        if len(codes) != 1 or codes[0] not in ("u4", "u5", "u6", "u7"):
            raise ValueError(f"invalid recipe quantization: {line}")
        kinds[codes[0]] += 1
    if sum(kinds.values()) != 292:
        raise ValueError(f"expected 292 calibrated GPTQ weights; got {sum(kinds.values())}")
    if "token_embd.weight rtn codes=u6" not in text:
        raise ValueError("missing separately quantized token embedding")
    return kinds


def parse_radinfo(text: str) -> dict:
    if any(marker not in text for marker in REQUIRED_META):
        raise ValueError("RAD2/qwen35/quantization metadata missing")
    if not re.search(r"^\s*n_layers\s+64\s*$", text, re.MULTILINE):
        raise ValueError("unexpected layer count")
    directory = [int(m.group(1)) for line in text.splitlines()
                 if (m := HEADER_RE.match(line))]
    if len(directory) != 1 or directory[0] < 1000:
        raise ValueError("incomplete or implausible RAD2 directory")
    in_section = False
    found = Counter()
    for line in text.splitlines():
        if BY_ENCODING.match(line):
            in_section = True
            continue
        if in_section and SECTION_END.match(line):
            in_section = False
            continue
        if not in_section:
            continue
        match = CODE_RE.match(line)
        if match:
            found[match.group(1)] += int(match.group(2))
    if not found:
        raise ValueError("no trellis weight encodings found")
    return {"directory_entries": directory[0], "trellis_codes": dict(found)}


def audit(model: Path, receipt: Path, recipe: Path, rad_info: Path) -> dict:
    state = json.loads(receipt.read_text())
    if state.get("stage") != "experimental_RAD_ready_unqualified":
        raise ValueError("model conversion is not complete")
    if str(model) != state.get("artifact"):
        raise ValueError("artifact path differs from immutable converter receipt")
    if not model.is_file() or model.stat().st_size != state["artifact_bytes"]:
        raise ValueError("RAD file absent or wrong size")
    digest = sha256(model)
    if digest != state.get("artifact_sha256"):
        raise ValueError("RAD file checksum differs from receipt")
    if sha256(recipe) != RECIPE_SHA256:
        raise ValueError("recipe source is not the pinned IQ6R-V4 recipe")
    desired = recipe_codes(recipe.read_text())
    runner = subprocess.run([str(rad_info), str(model)], text=True,
                            capture_output=True, timeout=300, check=False)
    if runner.returncode != 0:
        raise ValueError(f"rad-info exited {runner.returncode}: {runner.stderr[-300:]}")
    parsed = parse_radinfo(runner.stdout)
    if Counter(parsed["trellis_codes"]) != desired:
        raise ValueError(f"unexpected trellis encoding counts: {parsed['trellis_codes']} != {dict(desired)}")
    return {
        "status": "STRUCTURE_PASS_NUMERIC_QUALITY_NOT_TESTED",
        "artifact_sha256": digest,
        "artifact_bytes": model.stat().st_size,
        "dir_entries": parsed["directory_entries"],
        "expected_trellis_weights": dict(desired),
        "actual_trellis_weights": parsed["trellis_codes"],
        "model_quality_approved": False,
        "tp3_approved": False,
        "production_promotion_approved": False,
        "radinfo_text": runner.stdout,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--rad-info", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.model, args.receipt, args.recipe, args.rad_info)
    raw = result.pop("radinfo_text")
    args.output.with_suffix(".radinfo.txt").write_text(raw)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
