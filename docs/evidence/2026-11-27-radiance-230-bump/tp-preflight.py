#!/usr/bin/env python3
"""Scratch TP patch-chain probe, not a production composition or a qualification gate."""
import argparse
from dataclasses import replace
from pathlib import Path
import re
import runpy
import subprocess as sp
import sys
import tempfile

p = argparse.ArgumentParser()
p.add_argument("--mirror", required=True)
p.add_argument("--scratch", required=True)
a = p.parse_args()
repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo / "tools"))
from bigcherry.patcher import apply_all

ref = "3b13fd32663ef52252a7f0be255523480aa22ffe"

def git(*args):
    return sp.check_output(["git", "-C", a.mirror, *args])

def files(num):
    path, = (repo / "engines/radiance/patches").glob(num + "_*/patch.py")
    return runpy.run_path(str(path))["PATCHES"]

def literal(e):
    old = re.sub(r"\\(.)", r"\1", e.anchor, flags=re.S)
    assert re.escape(old) == e.anchor, e.id
    return old

def retarget(e, old, new):
    assert re.escape(old) in e.anchor and old in e.text, e.id
    return replace(e, anchor=e.anchor.replace(re.escape(old), re.escape(new)),
                   text=e.text.replace(old, new))

def adjusted(e):
    replacements = {
        "n1-rowmask": ("    bool                 chain_fenced = false;", "    bool                 fetches_rows = false;"),
        "n1-row-mask": ("        row.chain_fenced = chains == 2;", "        row.fetches_rows = fetches && fetches(sr.second) != 0;"),
        "n2-ctx-field": ("    int      span_pass_step;", "    int      dspark_walk;"),
        "n2-layout-tp-ownership": ("ABI_FIELD(RadBuildCtx, span_pass_step)", "ABI_FIELD(RadBuildCtx, dspark_walk)"),
    }
    bf16 = {"n2-attn-spans": "    w_o = out.w;\n", "n2-gdn-spans": "    w_out = out.w;\n",
            "n2-mlp-spans": "    w_down    = dn.w;\n"}
    if e.id in replacements:
        return retarget(e, *replacements[e.id])
    if e.id in bf16:
        return retarget(e, literal(e), bf16[e.id])
    return e

def seed(root, path):
    dest = root / path
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(git("show", ref + ":" + path))

with tempfile.TemporaryDirectory(prefix="bc-tp-preflight-", dir=a.scratch) as tmp:
    root = Path(tmp)
    for num in ("rad0016", "rad0017", "rad0018"):
        patches = []
        for f in files(num):
            if not f.create:
                seed(root, f.path)
            patches.append(replace(f, edits=tuple(adjusted(e) for e in f.edits)))
        for phase in ("apply", "reapply"):
            results = apply_all(patches, root)
            expected = "applied" if phase == "apply" else "already-applied"
            bad = [(r.path, e.id, e.status) for r in results for e in r.results if e.status != expected]
            print(num, phase, "edits", sum(len(r.results) for r in results), "bad", bad, flush=True)
            assert not bad and all(r.ok for r in results)
    # Only the target's architecture headers and ABI, plus the affected translation unit.
    for path in git("ls-tree", "-r", "--name-only", ref, "abi", "arch/common").decode().splitlines():
        seed(root, path)
    inc = root / "include"
    inc.mkdir()
    (inc / "arch").symlink_to(root / "arch/common", target_is_directory=True)
    source = root / "arch/qwen35_bf16/qwen35_bf16.cpp"
    result = sp.run(["c++", "-std=c++17", "-fsyntax-only", "-I", str(root/"abi"),
                     "-I", str(inc), str(source)], text=True, capture_output=True)
    print("BF16 Qwen translation-unit syntax", result.returncode, flush=True)
    print(result.stderr[-6000:], flush=True)
    # Audit rad0019 on the already-patched prefix, without applying incomplete package.
    names = {"rad_block_attn_gated_fp8.h": "rad_block_attn_gated_q.h",
             "rad_block_mlp_fp8.h": "rad_block_mlp_q.h", "rad_fp8.h": "rad_linear.h"}
    counts = {"exact": 0, "drift": 0}
    for f in files("rad0019"):
        path = str(Path(f.path).with_name(names.get(Path(f.path).name, Path(f.path).name)))
        seed(root, path)
        source = (root/path).read_text()
        for e in f.edits:
            n = len(re.findall(e.anchor, source, flags=re.M))
            if n == e.expect_matches:
                counts["exact"] += 1
            else:
                counts["drift"] += 1
            print("rad0019", path, e.id, "matches", n, "expected", e.expect_matches, flush=True)
    print("rad0019 anchor audit", counts, "(no application or semantic pass claimed)", flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
