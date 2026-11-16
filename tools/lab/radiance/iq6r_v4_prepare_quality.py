"""Card-free audit of the pinned held-out reference corpus and golden layout.

Validates structural alignment only. Upstream reference quality is NOT presumed
to be BF16-equivalent; tokenizer/model identity and scored logits must be
independently established before a KL comparison is accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

PUBLIC_CORPUS_SHA256 = "144243ceba1e6bde89f2a94fda39991a6e027847c797590fc1da5aa6d0ecb93e"
EXPECTED_DOCS = 596
EXPECTED_TOKENS = 1_694_621
EXPECTED_ROWS = EXPECTED_TOKENS - EXPECTED_DOCS
EXPECTED_TOP = 512
EXPECTED_VOCAB = 248_320


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_docs(corpus: list[dict], golden: dict) -> dict:
    docs = golden.get("docs")
    if not isinstance(docs, list) or len(docs) != EXPECTED_DOCS:
        raise ValueError("incorrect golden document count")
    if len(corpus) != EXPECTED_DOCS:
        raise ValueError("incorrect held-out corpus document count")
    if golden.get("rows") != EXPECTED_ROWS:
        raise ValueError("incorrect golden scoring rows")
    if golden.get("top") != EXPECTED_TOP or golden.get("n_vocab") != EXPECTED_VOCAB:
        raise ValueError("golden vocabulary or top-K does not match source")
    offsets = 0
    categories = Counter()
    for idx, (record, entry) in enumerate(zip(corpus, docs, strict=True)):
        length = record.get("tokens")
        if not isinstance(length, int) or length < 2:
            raise ValueError(f"bad token count in document {idx}")
        if not isinstance(record.get("prompt"), str) or not record["prompt"]:
            raise ValueError(f"missing text in document {idx}")
        if entry.get("tokens") != length or entry.get("source") != record.get("source"):
            raise ValueError(f"golden document {idx} source/token mismatch")
        skip = entry.get("score_from")
        if not isinstance(skip, int) or not 0 <= skip < length - 1:
            raise ValueError(f"invalid score_from in document {idx}")
        if entry.get("row0") != offsets:
            raise ValueError(f"golden document {idx} row-offset mismatch")
        offsets += length - skip - 1
        categories[record.get("category", "uncategorized")] += 1
    if sum(row["tokens"] for row in corpus) != EXPECTED_TOKENS or offsets != EXPECTED_ROWS:
        raise ValueError("held-out corpus and golden scored row counts differ")
    return {
        "documents": len(corpus),
        "tokens": EXPECTED_TOKENS,
        "scored_rows": offsets,
        "categories": dict(sorted(categories.items())),
        "category_count": len(categories),
        "golden_top_k": golden["top"],
        "golden_vocab": golden["n_vocab"],
        "golden_corpus_fnv1a64_advertised": golden.get("corpus_fnv1a64"),
    }


def audit(corpus: Path, golden_dir: Path, *, check_corpus_hash: bool = True) -> dict:
    if check_corpus_hash and sha256(corpus) != PUBLIC_CORPUS_SHA256:
        raise ValueError("pinned public held-out corpus SHA-256 mismatch")
    lines = [json.loads(line) for line in corpus.read_text().splitlines() if line.strip()]
    golden = json.loads((golden_dir / "kld.json").read_text())
    if golden.get("format") != "radiance-kld-top-1" or golden.get("model") != "Qwen/Qwen3.8-27B":
        raise ValueError("unexpected golden reference format/model")
    info = validate_docs(lines, golden)
    expected = {
        "tokens.i32": EXPECTED_TOKENS * 4,
        "gen.u8": EXPECTED_ROWS,
        "rows.f32": EXPECTED_ROWS * 3 * 4,
        "top.bin": EXPECTED_ROWS * EXPECTED_TOP * 8,
    }
    for name, byte_count in expected.items():
        path = golden_dir / name
        if not path.is_file() or path.stat().st_size != byte_count:
            raise ValueError(f"golden {name} missing or malformed (expected {byte_count} bytes)")
    return {
        "status": "PUBLIC_GOLDEN_LAYOUT_PASS_NOT_BF16_QUALIFIED",
        "source_corpus_sha256": PUBLIC_CORPUS_SHA256 if check_corpus_hash else None,
        "reference_status": "unverified model/tokenizer identity; no BF16 comparison authorized",
        "aligned_score_layout": info,
        "expected_golden_file_bytes": expected,
        "quality_accepted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--golden-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.corpus, args.golden_dir)
    args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
