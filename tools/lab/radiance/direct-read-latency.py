#!/usr/bin/env python3
"""Latency of random 4 KiB O_DIRECT reads of a file, the read libavx's n-gram gather makes on a row-cache miss
(radiance libavx/avx_ngram.cpp). Says whether the filesystem under a container is what paces `embed_lookup_q`.

Usage: direct-read-latency.py <file> [reads=4000] [threads=1]
Offsets are uniform over the file, block aligned, from a fixed seed, so two filesystems are read the same way.
"""
import mmap
import os
import random
import sys
import threading
import time


def worker(path, offsets, out):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    buf = mmap.mmap(-1, 4096)
    lat = []
    for off in offsets:
        t = time.perf_counter()
        os.preadv(fd, [buf], off)
        lat.append(time.perf_counter() - t)
    os.close(fd)
    out.extend(lat)


def main():
    path = sys.argv[1]
    reads = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    blocks = os.path.getsize(path) // 4096
    rng = random.Random(1234)
    offsets = [rng.randrange(blocks) * 4096 for _ in range(reads)]
    lat = []
    start = time.perf_counter()
    ts = [threading.Thread(target=worker, args=(path, offsets[i::threads], lat)) for i in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.perf_counter() - start
    lat.sort()
    us = [x * 1e6 for x in lat]
    print(f"{path}")
    print(f"  {reads} reads, {threads} thread(s): p50 {us[len(us) // 2]:.0f} us, p90 {us[len(us) * 9 // 10]:.0f} us, "
          f"p99 {us[len(us) * 99 // 100]:.0f} us; {reads / wall:.0f} reads/s")


if __name__ == "__main__":
    main()
