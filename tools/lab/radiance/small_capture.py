#!/usr/bin/env python3
"""RAD06: derive small-lane (decode-sized) helper captures from real large-lane captures, for `r3_moe_worker --replay[-sweep]`.

A capture (format 2, r3_moe_worker.cpp CaptureHeader) is: header {magic, version, layer, rows, route_count, hidden} as six
uint32, the layer's 512-entry slot map (int32), route_count routes of 12 bytes {int32 token, int32 expert, uint16 weight,
uint16 slot}, then rows x hidden E4M3 activation codes and rows x (hidden/128) float scales. This keeps the first N
tokens: their routes (token < N) and their activation rows, with the header rewritten. N <= 8 takes the helper's small
(decode) lane, so the replay exercises the pre-queued GEMV path a decode step uses.

  python3 small_capture.py OUT_DIR [--rows 8] CAPTURE.bin...
"""
import argparse
import os
import struct
import sys

HIDDEN = 2560
ACT_GROUP = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--rows', type=int, default=8)
    ap.add_argument('files', nargs='+')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for path in a.files:
        b = open(path, 'rb').read()
        magic, ver, layer, rows, rc, hidden = struct.unpack_from('<6I', b, 0)
        if ver != 2 or hidden != HIDDEN:
            print('skip', path, '(version %d, hidden %d)' % (ver, hidden))
            continue
        n = min(a.rows, rows)
        off = 24
        slots = b[off:off + 512 * 4]
        off += 512 * 4
        routes = [b[off + 12 * i: off + 12 * (i + 1)] for i in range(rc)]
        off += 12 * rc
        q = b[off: off + rows * HIDDEN]
        off += rows * HIDDEN
        s = b[off: off + rows * (HIDDEN // ACT_GROUP) * 4]
        keep = [r for r in routes if struct.unpack_from('<i', r, 0)[0] < n]
        out = struct.pack('<6I', magic, ver, layer, n, len(keep), hidden) + slots + b''.join(keep)
        out += q[: n * HIDDEN] + s[: n * (HIDDEN // ACT_GROUP) * 4]
        dst = os.path.join(a.out, 'small-' + os.path.basename(path))
        open(dst, 'wb').write(out)
        print('%s: layer %d, %d rows, %d routes (of %d)' % (dst, layer, n, len(keep), rc))


if __name__ == '__main__':
    main()
