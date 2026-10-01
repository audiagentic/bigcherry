#!/usr/bin/env python3
"""PRBE20/RD26: pairwise ubatch-size bisection using the already-built
subject llama-results binary. See README.md for the question this answers.

Usage (on Brutus, inside the repo root):
    BC_HIP_PATH=/mnt/vault/tmp/bc-rocm python3 tools/lab/prbe20-rd26-bisect/bisect_ubatch.py \
        --binary work/worktrees/builds/1210_rd26_bitidentical_decode_verify_standalone-subject-gfx1100+gfx1201+gfx1030/bin/llama-results \
        --model /mnt/data/llm-models/qwen3.6-35B-A3B/gguf/mtp/Qwen3.6-35B-A3B-APEX-MTP-I-Compact.gguf \
        --device 0
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_PROMPT = (
    "RD26 determinism probe. The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. Sphinx of black quartz, judge my vow. "
    "How vexingly quick daft zebras jump. Bright vixens jump; dozy fowl quack."
)
_CTX_SIZE = 256
_UBATCH_SIZES = (1, 2, 3, 4, 5)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_diff_offset(left: Path, right: Path) -> int | None:
    with left.open("rb") as lf, right.open("rb") as rf:
        offset = 0
        while True:
            lchunk = lf.read(1024 * 1024)
            rchunk = rf.read(1024 * 1024)
            if lchunk == rchunk:
                if not lchunk:
                    return None
                offset += len(lchunk)
                continue
            limit = min(len(lchunk), len(rchunk))
            for i in range(limit):
                if lchunk[i] != rchunk[i]:
                    return offset + i
            return offset + limit


def _run_one(binary: Path, model: Path, ubatch: int, out: Path, env: dict) -> None:
    argv = [
        str(binary), "--model", str(model), "--output", str(out),
        "--prompt", _PROMPT, "--ctx-size", str(_CTX_SIZE),
        "--batch-size", str(_CTX_SIZE), "--ubatch-size", str(ubatch),
        "-ngl", "99",
    ]
    result = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=300, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ubatch={ubatch} failed: {(result.stderr or result.stdout).strip()[:500]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    env = dict(os.environ)
    env["HIP_VISIBLE_DEVICES"] = args.device
    env.pop("ROCR_VISIBLE_DEVICES", None)

    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        outputs: dict[int, Path] = {}
        for ub in _UBATCH_SIZES:
            out = td / f"ubatch{ub}.gguf"
            print(f"running ubatch={ub} ...", file=sys.stderr)
            _run_one(args.binary, args.model, ub, out, env)
            outputs[ub] = out
            print(f"  ubatch={ub}: sha256={_sha256(out)}")

        print("\npairwise comparison (ubatch A vs ubatch B):")
        identical_pairs = []
        diverging_pairs = []
        for i, a in enumerate(_UBATCH_SIZES):
            for b in _UBATCH_SIZES[i + 1:]:
                same = _sha256(outputs[a]) == _sha256(outputs[b])
                if same:
                    identical_pairs.append((a, b))
                    print(f"  {a} vs {b}: IDENTICAL")
                else:
                    offset = _first_diff_offset(outputs[a], outputs[b])
                    diverging_pairs.append((a, b, offset))
                    print(f"  {a} vs {b}: DIFFERS at byte offset {offset}")

        print("\nsummary:")
        print(f"  identical pairs: {identical_pairs}")
        print(f"  diverging pairs: {[(a, b) for a, b, _ in diverging_pairs]}")
        if not diverging_pairs:
            print("  ALL ubatch sizes bit-identical -- the earlier contract failure may be "
                  "non-reproducible here or model/prompt/device-state-dependent.")
        elif all(a == 1 for a, b, _ in diverging_pairs):
            print("  Only ubatch=1 diverges from every other size -- consistent with "
                  "something specific to the single-token decode path.")
        elif any(a != 1 for a, b, _ in diverging_pairs):
            print("  Divergence exists between ubatch sizes BOTH >= 2 -- this is a general "
                  "batch-width-dependent kernel issue, not specific to the decode(1)-vs-"
                  "verify(>1) boundary RD26 targets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
