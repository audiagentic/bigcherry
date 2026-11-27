#!/usr/bin/env python3
"""One-time rad0017 three-way source reconciliation. Historical input is aaba6b2f6."""
import subprocess as s
import re
import tempfile
import hashlib
from pathlib import Path
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--mirror", required=True)
ap.add_argument("--apply", action="store_true", help="explicitly write historical merge result to source tree")
a = ap.parse_args()
repo = Path(__file__).resolve().parents[3]
p = repo / "engines/radiance/patches/rad0017_vtp14_n3_shared_rendezvous"
old,new=[s.check_output(["git","-C",a.mirror,"show",r+":libr4d/r4d_ar_entry.hip"],text=True) for r in ("b6b8751cccb0e834a13ae9e5afca824ee9a487aa","3b13fd32663ef52252a7f0be255523480aa22ffe")]
patch=(p/"patch.py").read_text()
outputs={}
for name,pat in [("bc_ar_rendezvous_impl.h",r'(?s)#include "r4d_ar_links\.h"\n.*?(?=/\* One resolved instance\x27s view of the world:)'),("bc_ar_lifecycle_impl.h",r"(?s)struct ArInst \{.*?(?=\}  /\* namespace \*/)")]:
    b,n=[re.search(pat,t).group() for t in (old,new)]
    f=p/"src"/name
    historical=s.check_output(["git","-C",str(repo),"show","aaba6b2f6:"+str(f.relative_to(repo))],text=True)
    prefix,l=historical.split("#pragma once\n\n",1)
    with tempfile.TemporaryDirectory() as d:
        fs=[Path(d)/k for k in ("local","base","target")]
        for x,t in zip(fs,(l,b,n)): x.write_text(t)
        r=s.run(["git","merge-file","-p",*map(str,fs)],capture_output=True,text=True)
        out=r.stdout
        if name=="bc_ar_rendezvous_impl.h":
            assert r.returncode==1
            conflict=re.search(r"(?ms)^<<<<<<<[^\n]*\n(.*?)^=======\n(.*?)^>>>>>>>[^\n]*\n",out)
            assert conflict and "#include <dlfcn.h>" in conflict[1] and conflict[2]=="#include <atomic>\n"
            out=out[:conflict.start()]+conflict[1]+conflict[2]+out[conflict.end():]
        else:
            assert r.returncode==0
            oldcond="if (s.key.rank == k.rank && s.key.world == k.world && s.key.dtype == k.dtype &&"
            assert out.count(oldcond)==1
            out=out.replace(oldcond,"if (s.key.group == k.group && s.key.rank == k.rank &&\n                s.key.world == k.world && s.key.dtype == k.dtype &&")
        assert "<<<<<<<" not in out
        content=prefix+"#pragma once\n\n"+out
        outputs[f]=content
        h=hashlib.sha256(content.encode()).hexdigest()
        patch,nmatch=re.subn(r'("'+re.escape(name)+r'": ")[0-9a-f]{64}(")',r"\g<1>"+h+r"\2",patch)
        assert nmatch==1
for f,t in outputs.items():
    if a.apply:
        f.write_text(t)
    else:
        assert f.read_text() == t, f
if a.apply:
    (p/"patch.py").write_text(patch)
else:
    assert (p/"patch.py").read_text() == patch
print("Merged both owned headers; retained native delta and explicit group-key condition.")
