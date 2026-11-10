"""VTP06 corpus admission for Qwen3.8-27B BF16 calibration.

Card-free. This is a strict provenance and leakage validator, NOT evidence
that a dataset is representative. No dataset is fetched or invented here.
The generated calibration JSONL is the only file the engine may consume;
holdout documents are kept separate for BF16-vs-IQ6R quality evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "VTP06_CORPUS_V1"
CATEGORIES = frozenset(("language", "code", "math", "multilingual", "long_context"))
SPLITS = frozenset(("calibration", "holdout"))
HEX = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def canonical_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def checked_records(source: Path, *, mode: str) -> tuple[list[dict], dict]:
    if mode not in ("pilot", "candidate"):
        raise ValueError("mode must be pilot or candidate")
    groups: dict[str, list[dict]] = defaultdict(list)
    hashes: set[str] = set()
    ids: set[str] = set()
    source_ids = set()
    counts: Counter[str] = Counter()
    chars: Counter[str] = Counter()
    if not source.is_file():
        raise ValueError("source JSONL is missing")
    if source.stat().st_size > 256 * 1024 * 1024:
        raise ValueError("source exceeds 256 MiB metadata guard")
    with source.open("r", encoding="utf-8") as stream:
        for lineno, raw in enumerate(stream, 1):
            if not raw.strip():
                raise ValueError(f"line {lineno}: blank corpus row")
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {lineno}: invalid JSON") from exc
            if not isinstance(item, dict):
                raise TypeError(f"line {lineno}: expected object")
            fields = ("id", "prompt", "source", "license", "category", "split")
            for field in fields:
                value = item.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"line {lineno}: nonempty string {field} required")
            if item["category"] not in CATEGORIES or item["split"] not in SPLITS:
                raise ValueError(f"line {lineno}: unsupported category or split")
            if len(item["prompt"]) < 32 or len(item["prompt"].encode()) > 131072:
                raise ValueError(f"line {lineno}: prompt must be 32–131072 UTF-8 bytes")
            if any(ord(c) < 32 and c not in "\n\r\t" for c in item["prompt"]):
                raise ValueError(f"line {lineno}: control character in prompt")
            if len(item["id"]) > 128 or len(item["source"]) > 256:
                raise ValueError(f"line {lineno}: excessive identifier length")
            if "score_from" in item and (
                item["split"] != "holdout"
                or isinstance(item["score_from"], bool)
                or not isinstance(item["score_from"], int)
                or not 0 <= item["score_from"] <= 262144
            ):
                raise ValueError(
                    f"line {lineno}: score_from is a native token index "
                    "supported only on held-out evaluation documents"
                )
            if item["license"].strip().casefold() in (
                "unknown", "none", "unreviewed", "n/a", "unspecified"
            ):
                raise ValueError(f"line {lineno}: unreviewed or unknown license")
            ident = item["id"]
            digest = hashlib.sha256(canonical_text(item["prompt"]).encode()).hexdigest()
            if ident in ids or digest in hashes:
                raise ValueError(f"line {lineno}: duplicate ID or prompt across splits")
            ids.add(ident)
            hashes.add(digest)
            source_ids.add(item["source"])
            counts[f'{item["split"]}:{item["category"]}'] += 1
            chars[item["split"]] += len(item["prompt"])
            groups[item["split"]].append(item)
    if not groups["calibration"]:
        raise ValueError("at least one calibration document is required")
    if mode == "candidate":
        if len(groups["calibration"]) < 200 or len(groups["holdout"]) < 50:
            raise ValueError("candidate requires >=200 calibration and >=50 holdout docs")
        if len(source_ids) < 3:
            raise ValueError("candidate must have >=3 independently identified sources")
        for category in CATEGORIES:
            if counts[f"calibration:{category}"] < 20:
                raise ValueError(f"candidate needs >=20 calibration docs of {category}")
            if counts[f"holdout:{category}"] < 5:
                raise ValueError(f"candidate needs >=5 holdout docs of {category}")
        if not any(len(d["prompt"]) >= 8192 for d in groups["calibration"]
                   if d["category"] == "long_context"):
            raise ValueError("candidate needs an >=8192-character long-context example")
    overview = {
        "mode": mode,
        "documents": {"calibration": len(groups["calibration"]),
                      "holdout": len(groups["holdout"])},
        "categories": dict(sorted(counts.items())),
        "characters_not_native_tokens": dict(chars),
        "unique_sources": len(source_ids),
        "source_identities": sorted(source_ids),
        "deduplication": "NFKC/casefold/collapse-whitespace SHA256 across both splits",
    }
    return groups["calibration"] + groups["holdout"], overview


def prepare(source: Path, checkpoint: Path, output: Path, mode: str) -> dict:
    """Validate everything before creating output. Never clobber a bundle."""
    records, overview = checked_records(source, mode=mode)
    if not checkpoint.is_dir():
        raise ValueError("checkpoint directory is missing")
    tokenizer = checkpoint / "tokenizer.json"
    tokenizer_config = checkpoint / "tokenizer_config.json"
    for p in (tokenizer, tokenizer_config):
        if not p.is_file():
            raise ValueError(f"model tokenizer input missing: {p.name}")
    if output.resolve() == source.resolve() or output.resolve().is_relative_to(checkpoint.resolve()):
        raise ValueError("output cannot overwrite source or checkpoint")
    if output.exists():
        raise FileExistsError("bundle already exists; never overwrite provenance")
    if not output.parent.is_dir():
        raise ValueError("create parent output directory explicitly")
    tokenizer_hash = sha256_file(tokenizer)
    tokenizer_config_hash = sha256_file(tokenizer_config)
    upstream_hash = sha256_file(source)
    if not HEX.fullmatch(upstream_hash):
        raise ValueError("invalid SHA")
    output.mkdir(exist_ok=False)
    for split in ("calibration", "holdout"):
        # Keep the metadata for review, but engine receives only prompt+source.
        path = output / f"{split}.jsonl"
        with path.open("x", encoding="utf-8") as stream:
            for doc in records:
                if doc["split"] != split:
                    continue
                row = {"prompt": doc["prompt"], "source": doc["source"], "id": doc["id"]}
                if split == "holdout" and "score_from" in doc:
                    row["score_from"] = doc["score_from"]
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    document_manifest = output / "document_provenance.jsonl"
    with document_manifest.open("x", encoding="utf-8") as stream:
        for doc in records:
            stream.write(json.dumps(
                {key: doc[key] for key in ("id", "source", "license", "category", "split")},
                sort_keys=True
            ) + "\n")
    result = {
        "schema": SCHEMA,
        "status": "STRUCTURAL_CANDIDATE_NOT_NUMERICALLY_QUALIFIED"
                  if mode == "candidate" else "PILOT_ONLY_NOT_QUALITY_QUALIFIED",
        "source_jsonl": str(source.resolve()),
        "source_sha256": upstream_hash,
        "checkpoint": str(checkpoint.resolve()),
        "tokenizer_sha256": tokenizer_hash,
        "tokenizer_config_sha256": tokenizer_config_hash,
        "calibration_jsonl": str((output / "calibration.jsonl").resolve()),
        "calibration_sha256": sha256_file(output / "calibration.jsonl"),
        "holdout_jsonl": str((output / "holdout.jsonl").resolve()),
        "holdout_sha256": sha256_file(output / "holdout.jsonl"),
        "provenance_sha256": sha256_file(document_manifest),
        "overview": overview,
        "native_token_counts_verified": False,
        "licensed_corpus_suitability_manually_approved": False,
        "iq6r_quality_approved": False,
        "warning": "Corpus counts are characters and documents, NOT native tokens. "
                   "Candidate structural checks do not establish representativity, "
                   "source licence validity, or IQ6R quality. No holdout may be fed to calibration.",
    }
    (output / "bundle.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def inspect_bundle(path: Path, checkpoint: Path) -> dict:
    bundle = json.loads((path / "bundle.json").read_text())
    if bundle.get("schema") != SCHEMA or Path(bundle.get("checkpoint", "")).resolve() != checkpoint.resolve():
        raise ValueError("bundle model/schema mismatch")
    for name, key in (
        ("calibration.jsonl", "calibration_sha256"),
        ("holdout.jsonl", "holdout_sha256"),
        ("document_provenance.jsonl", "provenance_sha256"),
    ):
        if sha256_file(path / name) != bundle.get(key):
            raise ValueError(f"bundle digest drift: {name}")
    if sha256_file(checkpoint / "tokenizer.json") != bundle["tokenizer_sha256"]:
        raise ValueError("tokenizer digest drift")
    if sha256_file(checkpoint / "tokenizer_config.json") != bundle["tokenizer_config_sha256"]:
        raise ValueError("tokenizer config digest drift")
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    cmd = parser.add_subparsers(dest="cmd", required=True)
    a = cmd.add_parser("audit", help="read-only candidate / pilot corpus check")
    a.add_argument("--input", required=True, type=Path)
    a.add_argument("--mode", choices=("pilot", "candidate"), default="candidate")
    a = cmd.add_parser("prepare", help="freeze corpus, tokenizer, holdout and provenance")
    a.add_argument("--input", required=True, type=Path)
    a.add_argument("--checkpoint", required=True, type=Path)
    a.add_argument("--out", required=True, type=Path)
    a.add_argument("--mode", choices=("pilot", "candidate"), default="candidate")
    a = cmd.add_parser("verify", help="verify the immutable bundle")
    a.add_argument("--bundle", required=True, type=Path)
    a.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cmd == "audit":
            _, result = checked_records(args.input, mode=args.mode)
        elif args.cmd == "prepare":
            result = prepare(args.input, args.checkpoint, args.out, args.mode)
        else:
            result = inspect_bundle(args.bundle, args.checkpoint)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, KeyError, FileExistsError) as exc:
        print(f"VTP06 corpus gate FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
