#!/usr/bin/env python3
"""Stubs for the entry points of the units a harness plugin leaves out (the plugin must still export what the selftest
looks up): every r4d symbol the reference plugin exports, or the objects leave undefined, and the objects do not define,
returns RAD_E_UNSUPPORTED (-18).

Usage: unit_stubs.py <nm> <out.S> <reference plugin> <object>...
"""
import subprocess
import sys


def names(nm, args, flag):
    out = subprocess.run([nm, flag, "--format=posix", *args], capture_output=True, text=True, check=True).stdout
    return {line.split()[0] for line in out.splitlines() if len(line.split()) >= 2 and not line.endswith(":")}


def main() -> int:
    nm, out, reference, objects = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    defined = names(nm, objects, "--defined-only")
    wanted = names(nm, objects, "--undefined-only") | {n for n in names(nm, ["-D", reference], "--defined-only")}
    stubs = sorted(n for n in wanted - defined if "r4d" in n.lower() and not n.startswith("_Z"))
    lines = ['        .section .text.uh_stubs,"ax",@progbits']
    for n in stubs:
        lines += [f"        .globl {n}", f"        .type {n},@function", f"{n}:", "        movl $-18, %eax", "        ret"]
    lines += ['        .section .note.GNU-stack,"",@progbits', ""]
    open(out, "w", encoding="utf-8", newline="\n").write("\n".join(lines))
    print(f"unit_stubs: {len(stubs)} symbol(s) stubbed")
    return 0


sys.exit(main())
