#!/usr/bin/env python3
"""Cut the lab's KLD text into a corpus for radiance's KL mode (rad-kld.sh).

radiance reads JSONL, one {"prompt", "score_from", "source"} a line. Each line here is one piece of the text, taken
at even spacing through the file so the pieces do not overlap, and scored from its first position.

Usage: rad_kld_corpus.py <out.jsonl> <pieces> <characters a piece> [text file]
"""
from __future__ import annotations

import json
import sys


def main() -> int:
    out, pieces, chars = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    text_path = sys.argv[4] if len(sys.argv) > 4 else "/mnt/data/bigcherry-work/corpus/kld-docs.txt"
    text = open(text_path, errors="replace").read()
    if len(text) < pieces * chars:
        print(f"TEXT_TOO_SHORT: {len(text)} characters for {pieces} x {chars}")
        return 1
    stride = len(text) // pieces
    with open(out, "w") as f:
        for i in range(pieces):
            piece = text[i * stride: i * stride + chars]
            f.write(json.dumps({"prompt": piece, "score_from": 0, "source": f"kld-docs-{i:02d}"}) + "\n")
    print(f"wrote {pieces} prompts of {chars} characters to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
