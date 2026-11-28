#!/usr/bin/env python3
"""Reproduce the ABI-only feasibility probe; never edits a vendor or patch.
Run from bump checkout with --mirror pointing to the upstream Git repository.
This is an evidence fixture, not a replacement for the production patch/test gate.
"""
import argparse
from dataclasses import replace
from pathlib import Path
import re
import runpy
import subprocess as sp
import sys
import tempfile

P = argparse.ArgumentParser()
P.add_argument("--mirror", required=True)
P.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))
P.add_argument("--scratch", required=True, help="existing executable scratch directory")
a = P.parse_args()
repo = Path(a.repo)
sys.path.insert(0, str(repo / "tools"))
from bigcherry.patcher import apply_all

old = "b6b8751cccb0e834a13ae9e5afca824ee9a487aa"
new = "3b13fd32663ef52252a7f0be255523480aa22ffe"
packages = ("rad0018", "rad0020", "rad0025", "rad0048")

def git(*args):
    return sp.check_output(["git", "-C", a.mirror, *args])

def seed(root, ref):
    paths = git("ls-tree", "-r", "--name-only", ref, "abi").decode().splitlines()
    paths += ["tests/" + x for x in ("abi_c_test.c", "abi_layout_test.cpp",
              "abi_layout.h", "abi_layout_table.h", "rad_test.h")]
    for path in paths:
        dest = root / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(git("show", ref + ":" + path))

def patch_modules():
    for num in packages:
        path, = (repo / "engines/radiance/patches").glob(num + "_*/patch.py")
        yield num, runpy.run_path(str(path))

def apply_headers(root, combined):
    for num, module in patch_modules():
        files = [p for p in module["PATCHES"] if p.path.startswith("abi/")]
        adjusted = []
        for p in files:
            edits = []
            for e in p.edits:
                changes = {
                    "n2-ctx-field": ("    int      span_pass_step;", "    int      dspark_walk;"),
                    "n2-layout-tp-ownership": ("ABI_FIELD(RadBuildCtx, span_pass_step)", "ABI_FIELD(RadBuildCtx, dspark_walk)"),
                    "msp-abi": ("    const int32_t* ckpt_cu;", "    const int32_t* vl_mask;"),
                    "abi44-layout-batch": ("ABI_FIELD(RadBatch, ckpt_cu)", "ABI_FIELD(RadBatch, vl_mask)"),
                }
                if combined and e.id in changes:
                    before, after = changes[e.id]
                    assert re.escape(before) in e.anchor and before in e.text
                    e = replace(e, anchor=e.anchor.replace(re.escape(before), re.escape(after)),
                                text=e.text.replace(before, after))
                if combined and e.id == "abi44-version":
                    e = replace(e, anchor=re.escape("#define RAD_ABI_VERSION 44u\n"),
                                text="#define RAD_ABI_VERSION 45u\n",
                                guard=re.escape("#define RAD_ABI_VERSION 45u\n"))
                edits.append(e)
            adjusted.append(replace(p, edits=tuple(edits)))
        for phase in ("first", "reapply"):
            results = apply_all(adjusted, root)
            if not all(r.ok for r in results):
                raise RuntimeError((num, phase, results))
            statuses = [e.status for r in results for e in r.results]
            assert statuses and set(statuses) == {"applied" if phase == "first" else "already-applied"}, (num, phase, statuses)
        print(num, "ABI-header apply/idempotence PASS", flush=True)

def run(args, **kw):
    return sp.run([str(x) for x in args], check=True, text=True, **kw)

def measure(root, label):
    source = root / "probe.c"
    source.write_text(r"""
#include "rad_abi_layout.h"
#include <inttypes.h>
#include <stdio.h>
int main(void) {
    printf("ABI %u fingerprint %016" PRIx64 "\n", RAD_ABI_VERSION, rad_abi_layout());
#define ABI_STRUCT(T) printf("%s||%zu|%zu\n", #T, (size_t)0, sizeof(T));
#define ABI_FIELD(T, f) printf("%s|%s|%zu|%zu\n", #T, #f, offsetof(T, f), sizeof(((T*)0)->f));
#include "rad_abi_layout_rows.h"
    return 0;
}
""")
    inc = ["-I", root / "abi"]
    outputs = []
    for compiler, standard, lang in (("cc", "c11", "c"), ("c++", "c++17", "c++")):
        exe = root / ("probe-" + lang)
        run([compiler, "-std=" + standard, "-x", lang, "-O2", *inc, source, "-o", exe])
        outputs.append(sp.check_output([exe], text=True))
    assert outputs[0] == outputs[1], label
    first, *rows = outputs[0].splitlines()
    print(label, first, "C/C++ identical", len(rows), "rows", flush=True)
    return first, {tuple(row.split("|")[:2]): tuple(map(int, row.split("|")[2:])) for row in rows}

def kv_fixture(root, version):
    module = dict(patch_modules())["rad0048"]
    source = root / "kv.c"
    source.write_text(module["_FIXTURE"].replace("RAD_ABI_VERSION == 44u", f"RAD_ABI_VERSION == {version}u"))
    exe = root / "kv"
    run(["cc", "-std=c11", "-DNDEBUG", "-O2", "-I", root / "abi", source, "-o", exe])
    output = sp.check_output([exe], text=True).strip()
    print("KV descriptor fixture PASS", output, flush=True)

def native_test(root):
    run(["cc", "-std=c11", "-pedantic-errors", "-I", root / "abi", "-c", root / "tests/abi_c_test.c", "-o", root / "abi_c.o"])
    run(["c++", "-std=c++17", "-O2", "-I", root / "abi", root / "tests/abi_layout_test.cpp", root / "abi_c.o", "-o", root / "native-abi"])
    res = sp.run([root / "native-abi"], text=True, capture_output=True)
    print("native ABI suite rc", res.returncode, flush=True)
    print(res.stdout[-2500:], res.stderr[-2500:], flush=True)
    return res.returncode

with tempfile.TemporaryDirectory(prefix="bc-abi-preflight-", dir=a.scratch) as tmp:
    roots = {label: Path(tmp) / label for label in ("old-native", "old-local", "new-native", "new-combined")}
    for label, root in roots.items():
        seed(root, old if label.startswith("old") else new)
    apply_headers(roots["old-local"], False)
    apply_headers(roots["new-combined"], True)
    measured = {label: measure(root, label) for label, root in roots.items()}
    for label, v in (("old-local", 44), ("new-combined", 45)):
        kv_fixture(roots[label], v)
    keys = [("RadBuildCtx", f) for f in ("", "span_pass_step", "draft_head_off", "expert_split", "dspark_walk", "tp_ownership", "pipelined_prefill", "pipelined_prefill_min_rows", "kv_layout", "attn_prefill_fp8")]
    keys += [("RadBatch", f) for f in ("", "ckpt_cu", "vl_mask", "cu_seqlens_h")]
    keys += [("RadKVGroupDecl", ""), ("RadKVGroupDecl", "kv_layout"), ("RadArchProbe", ""), ("RadArchProbe", "kv_layout_ok")]
    for key in keys:
        print(".".join(key), *(measured[label][1].get(key, "-") for label in roots), flush=True)
    n = measured["new-native"][1]
    c = measured["new-combined"][1]
    assert all(c[key] == value for key, value in n.items() if key[1]), "native field moved"
    disk = {"RadEncPlane", "RadEncoding", "RadFileEntry", "RadFileHeader", "RadFileKV", "RadFilePlane", "RadFileProfile", "RadVocabHeader", "RadVocabStep"}
    assert {k:v for k,v in c.items() if k[0] in disk} == {k:v for k,v in n.items() if k[0] in disk}
    print("native field offsets and on-disk records unchanged PASS", flush=True)
    assert native_test(roots["new-native"]) == 0
    combined = roots["new-combined"]
    table = combined / "tests/abi_layout_table.h"
    table.write_text("/* Experimental ABI45 snapshot, derived from measured header rows. */\n" +
                     "".join(f'{chr(123)} "{t}", "{f}", {off}, {size} {chr(125)},\n'
                             for (t, f), (off, size) in c.items()))
    suite = combined / "tests/abi_layout_test.cpp"
    source = suite.read_text()
    for before, after in (("kTableAbi = 44", "kTableAbi = 45"),
                          ("sizeof(RadBuildCtx), (size_t)224", "sizeof(RadBuildCtx), (size_t)256"),
                          ("sizeof(RadBatch), (size_t)504", "sizeof(RadBatch), (size_t)512"),
                          ("sizeof(RadArchProbe), (size_t)128", "sizeof(RadArchProbe), (size_t)136")):
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    suite.write_text(source)
    print("Combined ABI suite (scratch adapted table/size expectations)", flush=True)
    assert native_test(combined) == 0
