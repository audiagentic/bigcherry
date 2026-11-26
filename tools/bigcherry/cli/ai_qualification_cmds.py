"""Thin managed planner for gpu-ai-screen. Every execution is one bounded lease."""
from __future__ import annotations

import argparse
import json
import sys
import subprocess
from pathlib import Path

from ..core import harvest, paths, run_manifest as rm, workspace
from ..core.context import resolve_work_root
from ..hardware import ai_probe as a, ai_qualification as q, lease, power_probe, power_profiles

DIAGNOSTIC = "diagnostic_only_not_production_qualification"


def register(sub):
    p = sub.add_parser("gpu-ai-qualify", help="resolve and execute independent GPU qualification groups")
    sp = p.add_subparsers(dest="qualification_command", required=True)
    plan = sp.add_parser("plan", help="resolve and harvest workload groups without GPU work")
    plan.add_argument("--recipe-file", type=Path, default=paths.REPO_ROOT/"config/gpu-ai-qualification.toml")
    plan.add_argument("--recipe", required=True)
    plan.add_argument("--native-plan", type=Path, help="optional existing RCS27 unit-harness plan JSON")
    for name in ("probe", "profiles"):
        plan.add_argument("--"+name, type=Path, required=True)
    for name in ("bdf", "arch", "role"):
        plan.add_argument("--"+name, required=True)
    plan.add_argument("--profile", default="stock", help="observe-only; other names planned but not executable")
    for name in ("headroom-mib", "max-hotspot-c", "max-memory-c"):
        plan.add_argument("--"+name,type=int,required=True)
    plan.set_defaults(func=cmd_plan)
    run = sp.add_parser("run", help="execute one planned group under an existing GPU lease")
    run.add_argument("--plan",required=True)
    run.add_argument("--group",required=True)
    run.set_defaults(func=cmd_run)
    report = sp.add_parser("report", help="aggregate explicit managed group receipts; missing/nonpassing work stays visible")
    report.add_argument("--plan",required=True)
    report.add_argument("--result",action="append",default=[])
    report.add_argument("--precision-reference",action="append",default=[],
                        help="explicit previously reviewed RDNA3 observation run IDs; two per group")
    report.set_defaults(func=cmd_report)
    from .ai_peer_cmds import register as register_peer
    register_peer(sp)
    for cmd in (plan,run,report):
        cmd.add_argument("--work-root",type=Path)


def _source(path, role):
    path=path.resolve(strict=True)
    return {"role":role,"uri":str(path),"sha256":a.digest(path).removeprefix("sha256:")}


def _capture(root, topic, payload, inputs, refs=(), passed=True):
    run=workspace.create_run(root,"capture",topic,plan_item="PWR13",engine="none",
                            argv=sys.argv[:],purpose="Independent GPU qualification "+topic,
                            inputs=inputs,references=list(refs),subkind=topic,repo=paths.REPO_ROOT)
    m=rm.update_manifest(run.path,expected_generation=run.manifest()["generation"],state="running")
    payload=payload | {"run_id":run.run_id}
    rm.write_atomic(run.path/(topic+".json"),json.dumps(payload,sort_keys=True,indent=2)+"\n")
    rm.update_manifest(run.path,expected_generation=m["generation"],
                       state="completed" if passed else "failed",
                       patch={"suspect":DIAGNOSTIC,"execution":{"exit_code":0 if passed else 2},
                              "retention":{"hold":True}})
    harvest.harvest(root,run.path)
    return run.run_id


def _verified(root, rid, topic, filename):
    directory=workspace.run_dir(root,"capture",rid)
    manifest=rm.read_manifest(directory)
    bundle=harvest.bundle_dir(root,rid)
    source=bundle/"files"/filename
    q._need(not directory.is_symlink() and not bundle.is_symlink() and
            not source.is_symlink() and manifest["run_id"]==rid and
            manifest["subkind"]==topic and manifest["state"] in rm.TERMINAL_STATES
            and manifest["harvest"]["state"]=="verified" and not harvest.verify_bundle(bundle),
            "qualification input is not a verified managed capture")
    doc=json.loads((bundle/harvest.BUNDLE_NAME).read_text())
    q._need(doc["run_id"]==rid and doc["recipe"]["id"]=="capture/v1"
            and sum(f["path"]=="files/"+filename for f in doc["files"])==1,
            "qualification bundle identity/coverage mismatch")
    return manifest,source


def _plan(root,rid):
    m,path=_verified(root,rid,"gpu-ai-qualification-plan","gpu-ai-qualification-plan.json")
    doc=json.loads(path.read_text())
    q._need(m["state"]=="completed" and m["execution"]["exit_code"]==0
            and doc["schema"]==1 and doc["run_id"]==rid and doc["protocol"]==a.NATIVE_PROTOCOL
            and doc["screen_schema"]==a.SCREEN_SCHEMA and doc["production_qualified"] is False,
            "stale or invalid qualification plan")
    return doc,path


def cmd_plan(args):
    try:
        root=resolve_work_root(args.work_root)
        q._need(a.BDF.fullmatch(args.bdf) and a.ARCH.fullmatch(args.arch),"invalid physical device")
        before,_=a.snapshot(args.bdf)
        pci=Path("/sys/bus/pci/devices")/args.bdf
        total=power_probe.read_int(pci/"mem_info_vram_total").value
        used=power_probe.read_int(pci/"mem_info_vram_used").value
        q._need(type(total) is int and type(used) is int and 0<=used<total,"VRAM inventory unavailable")
        sources=[_source(args.probe,"probe"),_source(args.profiles,"profiles"),
                 _source(args.recipe_file,"recipe")]
        a.verify_probe_protocol(args.probe)
        profile=power_profiles.get_profile(power_profiles.load_profiles(args.profiles),args.profile)
        settings=profile.settings_for(args.role,args.arch)
        groups=q.resolve(q.load_recipe(args.recipe_file),args.recipe,args.arch,total,total-used,
                         args.headroom_mib,args.max_hotspot_c,args.max_memory_c)
        native=None
        if args.native_plan:
            sources.append(_source(args.native_plan,"native-plan"))
            native=json.loads(args.native_plan.read_text())
        scope=q.native_scope(native,args.arch)
        q._need(all(_source(Path(s["uri"]),s["role"])==s for s in sources),
                "qualification source changed during planning")
        payload={"schema":1,"protocol":a.NATIVE_PROTOCOL,"screen_schema":a.SCREEN_SCHEMA,
                 "recipe":args.recipe,"target":{"bdf":args.bdf,"arch":args.arch,"role":args.role},
                 "snapshot":before,"vram_bytes":total,"free_bytes_at_plan":total-used,
                 "profile":args.profile,"profile_settings":settings,"profile_digest":profile.source_digest,
                 "sources":{s["role"]:s for s in sources},"groups":groups,"native_scope":scope,
                 "production_qualified":False}
        rid=_capture(root,"gpu-ai-qualification-plan",payload,sources)
        print(json.dumps({"plan":rid,"groups":groups,"native_scope":scope,"production_qualified":False},indent=2))
        return 0
    except (OSError,ValueError,KeyError,TypeError,RuntimeError,AttributeError) as exc:
        print("gpu-ai-qualify plan: "+str(exc),file=sys.stderr)
        return 2


def idle_preflight():
    try:
        result=subprocess.run(["rocm-smi","--showpids"],capture_output=True,text=True,timeout=10,check=True)
    except (OSError,subprocess.SubprocessError) as exc:
        raise a.ScreenError("GPU process inventory unavailable") from exc
    q._need("No KFD PIDs currently running" in result.stdout, "GPU process inventory not idle")


def cmd_run(args):
    try:
        root=resolve_work_root(args.work_root)
        doc,path=_plan(root,args.plan)
        q._need(doc["profile"]=="stock","candidate execution blocked pending power-owner recovery hardening")
        group=doc["groups"][args.group]
        q._need(group["status"]=="planned","group is not executable on this target")
        q._need(lease.held_by_this_process(),"GPU lease not verified")
        idle_preflight()
        for role in ("probe","profiles"):
            source=doc["sources"][role]
            q._need(_source(Path(source["uri"]),role)==source,"planned input digest changed: "+role)
        target=doc["target"]
        ns=argparse.Namespace(**group["options"],**target,profile="stock",
            probe=Path(doc["sources"]["probe"]["uri"]),profiles=Path(doc["sources"]["profiles"]["uri"]),
            work_root=root,stock_confirmed=False,observe_current=True,apply_profile=False,
            confirm_idle=False,reference=None,
            qualification={"plan":args.plan,"group":args.group,"digest":a.digest(path)},
            qualification_path=path,qualification_expected=doc["snapshot"],
            qualification_sources=doc["sources"])
        from .ai_probe_cmds import cmd_screen
        return cmd_screen(ns)
    except (OSError,ValueError,KeyError,TypeError,RuntimeError,AttributeError) as exc:
        print("gpu-ai-qualify run: "+str(exc),file=sys.stderr)
        return 2


def precision_match(refs, results):
    """Two prior distinct managed reference runs plus a new exact match.

    Each entry is (signature, started_at, ended_at). The caller has already
    revalidated the plan binding, physical state, recipe and native counts.
    """
    return (len(refs)>=2 and bool(results) and
            all(r[0]==refs[0][0] for r in refs+results) and
            all(isinstance(r[1],str) and isinstance(r[2],str) and r[1]<=r[2]
                for r in refs+results) and
            max(r[2] for r in refs)<=min(r[1] for r in results))


def cmd_report(args):
    try:
        root=resolve_work_root(args.work_root)
        doc,plan_path=_plan(root,args.plan)
        groups={name:{"status":g["status"] if g["status"]=="not_applicable" else "not_run","runs":[]}
                for name,g in doc["groups"].items()}
        reference_ids=getattr(args,"precision_reference",[])
        all_ids=[*args.result,*reference_ids]
        q._need(len(all_ids)==len(set(all_ids)),"duplicate result/reference run ID")
        precision_refs={}
        precision_results={}
        inputs=[_source(plan_path,"qualification-plan")]
        refs=[{"type":"run","id":args.plan,"role":"qualification-plan"}]
        for rid in all_ids:
            m,path=_verified(root,rid,"gpu-ai-screen",rid+".power.json")
            receipt=json.loads(path.read_text())
            q._need(receipt.get("run_id")==rid,"receipt run identity mismatch")
            request=receipt["request"]
            binding=request["qualification"]
            name=binding["group"]
            q._need(binding=={"plan":args.plan,"group":name,"digest":a.digest(plan_path)}
                    and groups[name]["status"]!="not_applicable","receipt belongs to another plan/group")
            q._need(any(r.get("type")=="run" and r.get("id")==args.plan
                        and r.get("role")=="qualification-plan" for r in m["references"]),
                    "receipt lacks managed plan link")
            g=doc["groups"][name]["options"]
            s=receipt["observations"]["ai_screen"]
            status=s["status"]
            is_reference=rid in reference_ids
            if is_reference:
                q._need(status=="precision_observed","precision reference must be a completed observation")
            if status in ("pass","precision_observed"):
                precision=status=="precision_observed"
                q._need(precision==all(d in a.PRECISION_DTYPES for d in g["dtype"]),
                        "precision exception is restricted to named RDNA3 reference cases")
                for key in ("provider","n","seconds","warmup_seconds","min_hotspot_c","max_hotspot_c",
                            "max_memory_c","pause_ms","setup_timeout","memory_mib","resident_mib","headroom_mib","batch"):
                    q._need(request[key]==g[key],"receipt recipe mismatch: "+key)
                geometry=a.matrix_geometry(g["n"],g["matrix_shape"]) if any(d in a.MATRIX_DTYPES for d in g["dtype"]) else None
                q._need(request["matrix_geometry"]==geometry and request["dtypes"]==g["dtype"]
                        and all(request[k]==v for k,v in doc["target"].items())
                        and request["profile"]=="stock" and request["observe_current"] is True
                        and request["apply_profile"] is False and request["stock_confirmed"] is False and
                        request["probe_digest"]=="sha256:"+doc["sources"]["probe"]["sha256"]
                        and request["profile_digest"]==doc["profile_digest"],"receipt identity mismatch")
                q._need(m["state"]==("failed" if precision else "completed") and m["execution"]["exit_code"]==(2 if precision else 0)
                        and s["schema_version"]==a.SCREEN_SCHEMA and s["production_qualified"] is False
                        and receipt["observations"]["start"]==doc["snapshot"]==receipt["observations"]["stop"]
                        and receipt["transaction"]["state"]=="not_requested",
                        "passing receipt lacks required identity/restoration evidence")
                expected=[(d,p) for d in g["dtype"] for p in g["pause_ms"]]
                q._need(bool(expected) and [(c["dtype"],c["pause_ms"]) for c in s["cases"]]==expected,
                        "passing receipt lacks complete case coverage")
                for c in s["cases"]:
                    dtype=c["dtype"]
                    shape=a.matrix_geometry(g["n"],g["matrix_shape"]) if dtype in a.MATRIX_DTYPES else None
                    n=g["memory_mib"] if dtype=="memory" else shape[1] if shape else g["n"]
                    q._need(c["status"]==status and c["n"]==n,"passing receipt has nonpassing case")
                    a.validate_native(c["native"],doc["target"]["bdf"],doc["target"]["arch"],dtype,n,
                        a.case_provider(dtype,g["provider"]),g["seconds"]+g["warmup_seconds"],g["setup_timeout"],
                        g["headroom_mib"] or 512,batch=a.case_batch(dtype,g["batch"]),geometry=shape,
                        resident_mib=g["resident_mib"] if shape else 0)
                    if g["min_hotspot_c"] is not None:
                        q._need(c.get("hot_window_verified") is True and a.hot_window(c["junction_samples"],
                            c["native"]["start_ns"],c["native"]["end_ns"],g["seconds"],g["min_hotspot_c"]*1000)==c.get("hot_window")
                            and c.get("hot_window") is not None,"missing hot coverage")
                if precision:
                    q._need(doc["target"]["arch"]=="gfx1100" and isinstance(request["kernel"],str) and bool(request["kernel"])
                            and receipt["observations"]["start"]["controls"].get("vddgfx_offset_mv")==0,
                            "precision reference requires zero-offset observed gfx1100")
                    # Compare every epoch fingerprint, native runtime identity,
                    # and exact CPU error across separately captured observations.
                    signature=(request["kernel"],tuple((a.case_identity(c),
                        tuple(c["native"]["precision_fingerprints"]),
                        c["native"]["precision_max_abs_error"]) for c in s["cases"]))
                    # Timing/load-gap changes cannot acquire different allowed
                    # bytes for the same input corpus.
                    by_dtype={}
                    for c in s["cases"]:
                        observed=(tuple(c["native"]["precision_fingerprints"]),
                                  c["native"]["precision_max_abs_error"])
                        q._need(by_dtype.setdefault(c["dtype"],observed)==observed,
                                "RDNA3 reference differs between load patterns")
                    (precision_refs if is_reference else precision_results).setdefault(name,[]).append(
                        (signature,m["started_at"],m["ended_at"]))
            if not is_reference:
                groups[name]["runs"].append({"run_id":rid,"status":status})
            else:
                groups[name].setdefault("precision_references",[]).append(rid)
            inputs.append(_source(path,"screen"))
            refs.append({"type":"run","id":rid,"role":"precision-reference" if is_reference else "screen"})
        for name,group in groups.items():
            if group["runs"]:
                group["status"]="pass" if all(r["status"]=="pass" for r in group["runs"]) else "nonpass"
                matches=precision_results.get(name,[])
                references=precision_refs.get(name,[])
                if len(matches)==len(group["runs"]) and precision_match(references,matches):
                    group["status"]="allowed_precision"
        passed=all(g["status"] in ("pass","not_applicable","allowed_precision") for g in groups.values())
        payload={"schema":1,"plan":args.plan,"groups":groups,"independent_pass":passed,
                 "native_scope":doc["native_scope"],"production_qualified":False}
        rid=_capture(root,"gpu-ai-qualification-report",payload,inputs,refs,passed)
        print(json.dumps(payload|{"report":rid},indent=2))
        return 0 if passed else 2
    except (OSError,ValueError,KeyError,TypeError,RuntimeError,AttributeError) as exc:
        print("gpu-ai-qualify report: "+str(exc),file=sys.stderr)
        return 2
