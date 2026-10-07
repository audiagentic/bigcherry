"""Bump helper: for a patch's failing edits, show where the old-pin anchor stops matching the new-pin source.

Usage: _anchor_diff.py <patch dir name> <old rev> <edit id>...
"""
import importlib.util
import re
import subprocess
import sys

sys.path.insert(0, "tools")
pid, old_rev, ids = sys.argv[1], sys.argv[2], set(sys.argv[3:])
spec = importlib.util.spec_from_file_location("p", f"patches/{pid}/patch.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
patches = getattr(P, "PATCHES", None) or [getattr(P, "PATCH")]


def show(rev, path):
    return subprocess.run(["git", "-C", "vendor/llama.cpp", "show", f"{rev}:{path}"], capture_output=True, check=True).stdout.decode("utf-8", "replace")


for fp in patches:
    for e in fp.edits:
        if e.id not in ids:
            continue
        old, new = show(old_rev, fp.path), show("HEAD", fp.path)
        m = re.search(e.anchor, old, re.S | re.M)
        print(f"== {e.id} ({fp.path}) mode={e.mode} matches old pin: {bool(m)}")
        if not m:
            continue
        t = m.group(0)
        lo, hi = 0, len(t)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if t[:mid] in new:
                lo = mid
            else:
                hi = mid - 1
        print(f"   anchor {len(t)} chars / {t.count(chr(10)) + 1} lines; prefix found in new source: {lo} chars")
        print("   --- old anchor from the break:")
        print(t[max(0, lo - 200):lo + 500])
        j = new.find(t[:lo]) if lo > 20 else -1
        if j >= 0:
            print("   --- new source from the break:")
            print(new[j + max(0, lo - 200):j + lo + 700])
        else:
            first = t.strip().split("\n")[0]
            k = new.find(first)
            print("   --- first anchor line in new source:", k)
            if k >= 0:
                print(new[k:k + 700])
