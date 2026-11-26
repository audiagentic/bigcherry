"""Managed, observe-only wrapper for the existing peer-copy oracle fixture."""
from __future__ import annotations

import json
import subprocess
import signal
import sys
import tempfile
import time
from pathlib import Path

from ..core import paths, workspace, run_manifest as rm, harvest
from ..core.context import resolve_work_root
from ..hardware import ai_probe as a, lease
from . import ai_qualification_cmds as q

SIZES = (8,4088,4096,4104,2**20,64*2**20)
METHODS = ("push","pull","peer","peer-async")


def register(sub):
    p=sub.add_parser("peer",help="leased explicit physical-pair copy correctness")
    p.add_argument("--probe",type=Path,required=True)
    p.add_argument("--bdf",nargs=2,required=True)
    p.add_argument("--arch",nargs=2,required=True)
    p.add_argument("--headroom-mib",type=int,required=True)
    p.add_argument("--max-hotspot-c",type=int,required=True)
    p.add_argument("--max-memory-c",type=int,required=True)
    p.add_argument("--timeout",type=int,default=120)
    p.add_argument("--work-root",type=Path)
    p.set_defaults(func=cmd_peer)


def parse_coverage(output, bdfs, arches, rc):
    identity="MANAGED_PAIR "+" ".join([*bdfs,*arches])
    rows=[]
    access=[]
    for line in output.splitlines():
        if line.startswith("MANAGED_ACCESS "):
            _,src,dst,can=line.split()
            src,dst,can=map(int,(src,dst,can))
            if (src,dst) not in ((0,1),(1,0)) or can not in (0,1):
                raise a.ScreenError("invalid directional peer capability")
            access.append({"src":src,"dst":dst,"supported":bool(can)})
            continue
        if not line.startswith("MANAGED_CASE "):
            continue
        parts=line.split()
        if len(parts)!=7:
            raise a.ScreenError("invalid peer case record")
        _,method,src,dst,size,words,status=parts
        src,dst,size,words=map(int,(src,dst,size,words))
        if (method not in METHODS or (src,dst) not in ((0,1),(1,0)) or
                size not in SIZES or words!=size//8 or status not in ("pass","mismatch")):
            raise a.ScreenError("invalid peer case coverage")
        rows.append(dict(method=method,src=src,dst=dst,bytes=size,words_checked=words,status=status))
    expected=[(m,s,1-s,b) for b in SIZES for s in (0,1) for m in METHODS]
    actual=[(r["method"],r["src"],r["dst"],r["bytes"]) for r in rows]
    identity_ok=output.splitlines().count(identity)==1
    access_ok=[(r["src"],r["dst"]) for r in access]==[(0,1),(1,0)]
    complete=(identity_ok and access_ok and all(r["supported"] for r in access) and actual==expected)
    if complete and rc==0 and all(r["status"]=="pass" for r in rows):
        status="pass"
    elif complete and rc==1 and any(r["status"]=="mismatch" for r in rows):
        status="mismatch"
    elif rc==2 and identity_ok and access_ok:
        status="unavailable"
    else:
        status="inconclusive"
    return {"status":status,"complete":complete,"access":access,"cases":rows}


def cmd_peer(args):
    run=None
    manifest=None
    interrupted=False
    def cancel(signum, frame):
        nonlocal interrupted
        interrupted=True
    previous={sig:signal.signal(sig,cancel) for sig in (signal.SIGTERM,signal.SIGINT)}
    result={"schema":1,"status":"inconclusive","production_qualified":False}
    try:
        if (len(set(args.bdf))!=2 or not all(a.BDF.fullmatch(x) for x in args.bdf)
                or not all(a.ARCH.fullmatch(x) for x in args.arch)
                or not 256<=args.headroom_mib<=65536 or not 1<=args.timeout<=300
                or not 1<=args.max_hotspot_c<=110 or not 1<=args.max_memory_c<=110):
            raise a.ScreenError("invalid peer bounds or device pair")
        if not lease.held_by_this_process():
            raise a.ScreenError("GPU lease not verified")
        q.idle_preflight()
        source=q._source(args.probe,"probe")
        binary=Path(source["uri"])
        protocol=subprocess.run([str(binary),"--protocol-version"],capture_output=True,text=True,timeout=10)
        if protocol.returncode or protocol.stdout.strip()!="1" or q._source(binary,"probe")!=source:
            raise a.ScreenError("peer probe protocol/input mismatch")
        root=resolve_work_root(args.work_root)
        before={}
        sensors=[]
        for bdf in args.bdf:
            snap,hw=a.snapshot(bdf)
            before[bdf]=snap
            j,jlimit=a.thermal_sensor(hw,args.max_hotspot_c)
            m,mlimit=a.thermal_sensor(hw,args.max_memory_c,labels=("mem","memory"))
            sensors.extend([(bdf,j,jlimit,[]),(bdf,m,mlimit,[])])
        result.update(request={"bdf":args.bdf,"arch":args.arch,"headroom_mib":args.headroom_mib,
            "max_hotspot_c":args.max_hotspot_c,"max_memory_c":args.max_memory_c,
            "probe":source,"timeout":args.timeout},before=before)
        run=workspace.create_run(root,"capture","gpu-ai-peer",plan_item="PWR13",engine="none",
            argv=sys.argv[:],purpose="Independent explicit pair copy correctness",inputs=[source],
            subkind="gpu-ai-peer",repo=paths.REPO_ROOT)
        result["run_id"]=run.run_id
        manifest=rm.update_manifest(run.path,expected_generation=run.manifest()["generation"],state="running")
        with a.DriverHealthWindow() as health, tempfile.TemporaryFile(dir=run.path) as output:
            for _,sensor,limit,samples in sensors:
                a.sample(sensor,limit,samples)
            start=time.monotonic()
            if interrupted:
                raise a.ScreenError("peer interrupted","cancelled")
            if not lease.held_by_this_process():
                raise a.ScreenError("GPU lease lost before peer launch")
            proc=subprocess.Popen([str(binary),"--managed",*args.bdf,*args.arch,str(args.headroom_mib)],
                                  stdout=output,stderr=subprocess.STDOUT)
            try:
                while True:
                    if interrupted:
                        raise a.ScreenError("peer interrupted","cancelled")
                    if not lease.held_by_this_process():
                        raise a.ScreenError("GPU lease lost","lease_lost")
                    for _,sensor,limit,samples in sensors:
                        a.sample(sensor,limit,samples)
                    health.scan()
                    if health.events:
                        raise a.ScreenError("AMDGPU fault observed","device_error")
                    if proc.poll() is not None:
                        break
                    if time.monotonic()-start>=args.timeout:
                        raise a.ScreenError("peer deadline exceeded","timeout")
                    time.sleep(a.POLL_SECONDS)
            except (a.ScreenError,KeyboardInterrupt) as exc:
                result.update(status=getattr(exc,"status","cancelled"),diagnostic=str(exc))
            finally:
                if proc.poll() is None:
                    proc.kill()
                proc.wait()
                result["driver_health"]=health.observation()
                result["temperatures"]=[{"bdf":b,"sensor":s.name,"samples_mc":v,"limit_mc":l}
                                        for b,s,l,v in sensors]
                output.seek(0)
                raw=output.read().decode(errors="replace")
                result["native_output_tail"]=raw[-32000:]
            if "diagnostic" not in result:
                result.update(parse_coverage(raw,args.bdf,args.arch,proc.returncode))
        result["after"]={bdf:a.snapshot(bdf)[0] for bdf in args.bdf}
        if (result["before"]!=result["after"] or not lease.held_by_this_process()
                or q._source(binary,"probe")!=source):
            raise a.ScreenError("peer physical identity/controls/lease/input changed")
    except (OSError,ValueError,KeyError,RuntimeError,subprocess.SubprocessError) as exc:
        result.update(status=getattr(exc,"status","inconclusive"),diagnostic=str(exc))
    except KeyboardInterrupt:
        result.update(status="cancelled",diagnostic="interrupted")
    finally:
        for sig,handler in previous.items():
            signal.signal(sig,handler)
    passed=result["status"]=="pass"
    if run is not None and manifest is not None:
        rm.write_atomic(run.path/"gpu-ai-peer.json",json.dumps(result,sort_keys=True,indent=2)+"\n")
        rm.update_manifest(run.path,expected_generation=manifest["generation"],
            state="completed" if passed else ("cancelled" if result["status"]=="cancelled" else "failed"),
            patch={"suspect":q.DIAGNOSTIC,"execution":{"exit_code":0 if passed else 2},
                   "retention":{"hold":True}})
        harvest.harvest(root,run.path)
    print(json.dumps({k:v for k,v in result.items() if k not in ("native_output_tail","temperatures")},indent=2))
    return 0 if passed else 2
