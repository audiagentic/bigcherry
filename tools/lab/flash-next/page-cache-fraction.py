#!/usr/bin/env python3
"""Print the percentage (0-100) of a file that is resident in the page cache (mincore over a read-only mapping).
Usage: page-cache-fraction.py <file>"""
import ctypes
import os
import sys

PROT_READ, MAP_SHARED = 1, 1


def main() -> int:
    fd = os.open(sys.argv[1], os.O_RDONLY)
    size = os.fstat(fd).st_size
    if size == 0:
        print(100)
        return 0
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    addr = libc.mmap(None, size, PROT_READ, MAP_SHARED, fd, 0)
    if addr in (None, ctypes.c_void_p(-1).value):
        return 1
    page = os.sysconf("SC_PAGESIZE")
    pages = (size + page - 1) // page
    vec = (ctypes.c_ubyte * pages)()
    if libc.mincore(addr, size, vec) != 0:
        libc.munmap(addr, size)
        return 1
    resident = sum(b & 1 for b in bytes(vec))
    libc.munmap(addr, size)
    print(100 * resident // pages)
    return 0


if __name__ == "__main__":
    sys.exit(main())
