#!/usr/bin/env python3
"""Compare the R9700 helper's result rows with the engine's captured rows for one dumped request
(R3_MOE_HELPER_DUMP, mask-only mode). Usage: fn-helper-dump-compare.py <dump-dir> <top_k> <hidden>"""
import struct
import sys
from pathlib import Path

import numpy as np


def bf16(path):
    raw = np.fromfile(path, dtype=np.uint16).astype(np.uint32) << 16
    return raw.view(np.float32)


def main():
    d, top_k, hidden = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    for rank, k in [(r, k) for k in list(range(8)) + [f"L{i}" for i in range(64)] for r in (0, 1)]:
        rp = d / f"r{rank}.{k}.routes.bin"
        if not rp.exists():
            continue
        routes = [struct.unpack_from("<iiHH", rp.read_bytes(), o) for o in range(0, rp.stat().st_size, 12)]
        helper = bf16(d / f"r{rank}.{k}.helper.bin").reshape(-1, hidden)
        engine = bf16(d / f"r{rank}.{k}.engine.bin").reshape(-1, hidden)
        seen = bf16(d / f"r{rank}.{k}.seen.bin").reshape(-1, hidden)
        sbad = []
        for token, _e, _w, slot in routes:
            flat = token * top_k + slot
            r = float(np.linalg.norm(seen[flat] - helper[flat]) / max(np.linalg.norm(helper[flat]), 1e-30))
            if r >= 2e-3:
                sbad.append((token, slot, r))
        print(f"rank {rank}: add-kernel reads differing from helper buffer: {len(sbad)} {sbad[:6]}")
        print(f"rank {rank} req {k}: {len(routes)} routes")
        bad = 0
        for token, expert, _w, slot in routes:
            flat = token * top_k + slot
            h, e = helper[flat], engine[flat]
            rel = float(np.linalg.norm(h - e) / max(np.linalg.norm(e), 1e-30))
            if rel >= 2e-3:
                # Which engine row does the helper row best match? Exposes a row/slot offset.
                best = int(np.argmin(np.linalg.norm(engine - h, axis=1)))
                print(f"  {k} tok {token} slot {slot} expert {expert} flat {flat} rel {rel:.3e} "
                      f"|h| {np.linalg.norm(h):.3g} |e| {np.linalg.norm(e):.3g} best-engine-flat {best}")
                bad += rel >= 2e-3
        print(f"  bad rows {bad}")


if __name__ == "__main__":
    main()
