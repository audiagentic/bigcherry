import json
from pathlib import Path
from argparse import Namespace
from unittest.mock import patch

import pytest

from bigcherry.hardware import ai_probe as a, ai_qualification as q
from bigcherry.cli import ai_qualification_cmds as cli
from bigcherry.core import workspace, run_manifest as rm, harvest

ROOT=Path(__file__).resolve().parents[3]
RECIPE=ROOT/"config/gpu-ai-qualification.toml"


def resolve(arch="gfx1100",raw=None):
    return q.resolve(q.load_recipe(RECIPE) if raw is None else raw,"extended",arch,
                     24*2**30,23*2**30,512,80,80)


def test_platform_recipe_architecture_capabilities_and_memory():
    for arch,provider in (("gfx1030","rocblas"),("gfx1100","rocblas"),("gfx1201","hipblaslt")):
        groups=resolve(arch)
        assert groups["matrix"]["options"]["provider"]==provider
        assert ("fp8" in groups["matrix"]["options"]["dtype"])==(arch=="gfx1201")
        assert groups["wmma-positive"]["status"]==("not_applicable" if arch=="gfx1030" else "planned")
        assert groups["dot2-signed"]["status"]=="planned"
        assert groups["dot2-signed"]["options"]["dtype"]==[a.PRECISION_CASES["dot2"] if arch=="gfx1100" else "dot2"]
        assert groups["hot"]["options"]["min_hotspot_c"] is None
        elevated=q.resolve(q.load_recipe(RECIPE),"elevated-temperature",arch,
                           24*2**30,23*2**30,512,80,80)
        assert set(elevated)=={"elevated-temperature"}
        assert elevated["elevated-temperature"]["options"] == (
            groups["hot"]["options"] | {"min_hotspot_c":60})
        assert "elevated-temperature" not in groups
        operational=q.resolve(q.load_recipe(RECIPE),"operational",arch,
                              24*2**30,23*2**30,512,80,80)
        assert operational["hot"]["options"]["min_hotspot_c"] is None
        assert operational["hot"]["options"]["max_hotspot_c"]==80
        assert operational["hot"]["options"]["max_memory_c"]==80
        assert groups["hot"]["options"]["dtype"]==["fp32" if arch=="gfx1201" else "bf16"]
        assert groups["hot"]["options"]["n"]==(1024 if arch=="gfx1030" else 768)
        assert groups["hot"]["options"]["warmup_seconds"]==(60 if arch=="gfx1030" else 30)
        assert groups["hot"]["options"]["seconds"]==60
        assert groups["burst"]["options"]["n"]==768
        mem=groups["memory"]["options"]["memory_mib"]
        res=groups["resident"]["options"]["resident_mib"]
        assert mem==res==21*1024
        assert all(g["options"]["headroom_mib"]==512 for g in groups.values() if g["status"]=="planned")


@pytest.mark.parametrize("change",[
    lambda r:r["recipe"]["extended"]["groups"].append("missing"),
    lambda r:r["recipe"]["extended"]["groups"].append("matrix"),
    lambda r:r["group"]["direct"].update(dtypes=["bogus"]),
    lambda r:r["group"]["resident"].update(resident_fraction=[8,7]),
    lambda r:r["group"]["hot"].update(min_hotspot_c=80),
    lambda r:r["group"]["direct"].update(pause_ms=[0,0]),
    lambda r:r["group"]["wmma-positive"].update(dtypes=["alu","wmma-fp16"]),
])
def test_invalid_recipe_fails_closed(change):
    raw=q.load_recipe(RECIPE)
    change(raw)
    with pytest.raises(a.ScreenError):
        resolve("gfx1030",raw)


def test_native_scope_only_uses_existing_routes_and_gaps():
    doc={"arch":"gfx1100","outcome":"diagnostic","qualification":"not_run","release":"r",
         "runs":[{"kind":"overlay","name":"iqr","units":["u"],"selftest_case":"c",
                  "other_rows_need_evidence":["row"],"reference_required":True,"build_args":["never execute"]}],
         "selected_units":["u","v"],"uncovered_units":["v"],"inventory_kind":"canonical_source_only"}
    got=q.native_scope(doc,"gfx1100")
    assert got["qualification"]=="not_run" and got["uncovered_units"]==["v"]
    assert got["routes"][0]["other_rows_need_evidence"]==["row"]
    assert "build_args" not in got["routes"][0]
    with pytest.raises(a.ScreenError):
        q.native_scope(doc,"gfx1201")


def plan_fixture(tmp_path):
    groups=resolve()
    groups["dot2-signed"]["options"]["dtype"]=["dot2"]
    # Only the two groups necessary to exercise pass and missing evidence.
    doc={"schema":1,"recipe":"fixture","protocol":a.NATIVE_PROTOCOL,"screen_schema":a.SCREEN_SCHEMA,
         "groups":{"direct":groups["direct"],"dot2-signed":groups["dot2-signed"]},
         "target":{"bdf":"0000:03:00.0","role":"helper0","arch":"gfx1100"},
         "profile":"stock","snapshot":{"observed":True},"native_scope":q.native_scope(None,"gfx1100"),
         "sources":{"probe":{"sha256":"a"*64}},"profile_digest":"sha256:"+"b"*64,"production_qualified":False}
    rid=cli._capture(tmp_path,"gpu-ai-qualification-plan",doc,[])
    return rid


def test_report_missing_or_nonpass_not_promoted(tmp_path):
    plan=plan_fixture(tmp_path)
    _,path=cli._plan(tmp_path,plan)
    binding={"plan":plan,"group":"dot2-signed","digest":a.digest(path)}
    run=workspace.create_run(tmp_path,"capture","gpu-ai-screen",plan_item="PWR13",engine="none",
         argv=[],references=[{"type":"run","id":plan,"role":"qualification-plan"}],subkind="gpu-ai-screen")
    m=rm.update_manifest(run.path,expected_generation=run.manifest()["generation"],state="running")
    a.screen_receipt(run.run_id,{"kind":"ai-power-screen","schema_version":a.SCREEN_SCHEMA,
        "status":"mismatch","request":{"qualification":binding},"production_qualified":False}).save(
        run.path/(run.run_id+".power.json"))
    rm.update_manifest(run.path,expected_generation=m["generation"],state="failed",
                       patch={"execution":{"exit_code":2}})
    harvest.harvest(tmp_path,run.path)
    with patch.object(cli,"resolve_work_root",return_value=tmp_path):
        assert cli.cmd_report(Namespace(work_root=tmp_path,plan=plan,result=[run.run_id]))==2
    reports=list((tmp_path/"runs/capture").glob("*-gpu-ai-qualification-report-*"))
    assert len(reports)==1
    result=json.loads((reports[0]/"gpu-ai-qualification-report.json").read_text())
    assert not result["production_qualified"] and not result["independent_pass"]
    assert result["groups"]["direct"]["status"]=="not_run"
    assert result["groups"]["dot2-signed"]["status"]=="nonpass"


def test_modified_plan_bundle_refused(tmp_path):
    plan=plan_fixture(tmp_path)
    _,path=cli._plan(tmp_path,plan)
    path.write_text("{}")
    with pytest.raises(a.ScreenError,match="verified"):
        cli._plan(tmp_path,plan)

def test_idle_preflight_fails_closed():
    from subprocess import CompletedProcess
    with patch.object(cli.subprocess,"run",return_value=CompletedProcess([],0,"No KFD PIDs currently running","")):
        cli.idle_preflight()
    with patch.object(cli.subprocess,"run",return_value=CompletedProcess([],0,"PID 123","")):
        with pytest.raises(a.ScreenError,match="not idle"):
            cli.idle_preflight()


def test_report_revalidates_passing_case(tmp_path):
    plan=plan_fixture(tmp_path)
    doc,path=cli._plan(tmp_path,plan)
    g=doc["groups"]["dot2-signed"]["options"]
    request={k:v for k,v in g.items() if k not in ("dtype","matrix_shape")}
    request.update(doc["target"],dtypes=g["dtype"],matrix_geometry=None,profile="stock",
        observe_current=True,stock_confirmed=False,apply_profile=False,probe_digest="sha256:"+"a"*64,
        profile_digest=doc["profile_digest"],qualification={"plan":plan,"group":"dot2-signed","digest":a.digest(path)})
    n=g["n"]
    lanes=n*a.INSTRUCTION_BLOCK
    allocated=lanes*20+512
    native=dict(protocol_version=a.NATIVE_PROTOCOL,status="pass",bdf=doc["target"]["bdf"],arch="gfx1100",
        dtype="dot2",n=n,provider="hip-native",algorithm=a.INSTRUCTION_ALGORITHMS["dot2"],batch=1,
        library_version="fixture",hip=1,driver=1,workspace=0,iterations=2,elapsed_s=g["seconds"],
        start_ns=10**9,end_ns=(1+g["seconds"])*10**9,block_size=a.INSTRUCTION_BLOCK,wave_size=32,
        host_checks=2,checked_lanes=lanes*2,host_elements_checked=lanes*8,guard_checks=4,input_checks=2,
        allocated_bytes=allocated,free_before_bytes=2**30+allocated,free_after_bytes=2**30)
    cases=[dict(dtype="dot2",pause_ms=p,status="pass",n=n,native=native) for p in g["pause_ms"]]
    run=workspace.create_run(tmp_path,"capture","gpu-ai-screen",plan_item="PWR13",engine="none",
         argv=[],references=[{"type":"run","id":plan,"role":"qualification-plan"}],subkind="gpu-ai-screen")
    m=rm.update_manifest(run.path,expected_generation=run.manifest()["generation"],state="running")
    a.screen_receipt(run.run_id,{"kind":"ai-power-screen","schema_version":a.SCREEN_SCHEMA,
        "status":"pass","request":request,"production_qualified":False,"cases":cases,
        "before":doc["snapshot"],"after":doc["snapshot"]}).save(run.path/(run.run_id+".power.json"))
    rm.update_manifest(run.path,expected_generation=m["generation"],state="completed",
                       patch={"execution":{"exit_code":0}})
    harvest.harvest(tmp_path,run.path)
    with patch.object(cli,"resolve_work_root",return_value=tmp_path):
        assert cli.cmd_report(Namespace(work_root=tmp_path,plan=plan,result=[run.run_id]))==2
    report=next((tmp_path/"runs/capture").glob("*-gpu-ai-qualification-report-*"))
    result=json.loads((report/"gpu-ai-qualification-report.json").read_text())
    assert result["groups"]["dot2-signed"]["status"]=="pass"
    assert result["groups"]["direct"]["status"]=="not_run"
    assert result["independent_pass"] is False

def test_precision_match_requires_two_prior_identical_references():
    ref=[("same","01","02"),("same","03","04")]
    result=[("same","05","06")]
    assert cli.precision_match(ref,result)
    assert not cli.precision_match(ref[:1],result)
    assert not cli.precision_match(ref,[])
    assert not cli.precision_match(ref,[("changed-one-bit","05","06")])
    assert not cli.precision_match(ref,[("same","03","06")])
    assert not cli.precision_match([ref[0],("different","03","04")],result)


def test_precision_recipe_mapping_only_gfx1100():
    raw=q.load_recipe(RECIPE)
    raw["target"]["gfx1201"]["precision_references"]=True
    with pytest.raises(a.ScreenError,match="precision"):
        resolve("gfx1201",raw)

def test_native_scope_retains_requested_rows_and_reference_identity():
    doc={"arch":"gfx1100","outcome":"diagnostic","qualification":"not_run","release":"r",
         "selected_units":["r4d_a"],"uncovered_units":[],"inventory_kind":"matching_libr3_membership",
         "reference_sha256":"a"*64,"family_policy_sha256":"b"*64,
         "runs":[{"kind":"overlay","name":"iqr","units":["r4d_a"],"selftest_case":"base",
                  "base_rows":["base","extra"],"overlay_rows":["bc_base","bc_extra"],
                  "other_rows_need_evidence":["extra"]}],
         "requested_rows":[{"row":"bc_base","status":"configured_route","routes":[{"kind":"overlay","name":"iqr"}]},
                           {"row":"unknown","status":"uncovered","routes":[]}],
         "affected_scope":{"scope":"full_inventory_conservative"}}
    got=q.native_scope(doc,"gfx1100")
    assert got["status"]=="incomplete" and got["qualification"]=="not_run"
    assert got["requested_rows"]==doc["requested_rows"] and got["reference_sha256"]=="a"*64
    assert "uncovered requested rows" in got["gaps"]
    assert "additional runtime rows need evidence" in got["gaps"]
    with pytest.raises(a.ScreenError):
        q.native_scope(doc|{"selected_units":["r4d_a","r4d_b"]},"gfx1100")
    with pytest.raises(a.ScreenError):
        q.native_scope(doc|{"requested_rows":[{"row":"bad","status":"configured_route","routes":[]}]},"gfx1100")
    with pytest.raises(a.ScreenError):
        q.native_scope(doc|{"reference_sha256":"fake"},"gfx1100")

def test_unknown_native_rows_are_visible_even_with_no_unit_route():
    doc={"arch":"gfx1201","outcome":"diagnostic","qualification":"not_run","release":"r",
         "selected_units":[],"uncovered_units":[],"runs":[],
         "requested_rows":[{"row":"unknown","status":"uncovered","routes":[]}]}
    scope=q.native_scope(doc,"gfx1201")
    assert scope["status"]=="incomplete" and scope["requested_rows"]==doc["requested_rows"]


@pytest.mark.parametrize("arch,budget",[("gfx1030",11),("gfx1100",14),("gfx1201",15)])
def test_slim_strict_budget(arch,budget):
    raw=q.load_recipe(RECIPE)
    args=(arch,24*2**30,23*2**30,512,80,80)
    slim=q.resolve(raw,"slim",*args)
    assert q.workload_seconds(slim)==budget
    assert q.workload_seconds(q.resolve(raw,"sanity",*args))==2
    assert slim["memory"]["options"]["memory_mib"]==192
    assert slim["direct"]["options"]["pause_ms"]==[2]
    assert not any(d in a.PRECISION_DTYPES for g in slim.values()
                   if g["status"]=="planned" for d in g["options"]["dtype"])
    assert q.resolve(raw,"extended",*args)["memory"]["options"]["memory_mib"]==21*1024
    assert q.resolve(raw,"screen",*args)["direct"]["options"]["seconds"]==4


@pytest.mark.parametrize("group,override",[
    ("hot",{"seconds":1}),("direct",{"min_hotspot_c":1}),
    ("direct",{"seconds":0}),("direct",{"memory_fraction":[1,128]}),
    ("missing",{"seconds":1}),
])
def test_recipe_override_refuses_invalid(group,override):
    raw=q.load_recipe(RECIPE)
    raw["recipe"]["extended"]["overrides"]={group:override}
    with pytest.raises(a.ScreenError):
        resolve("gfx1100",raw)


def test_run_all_stops_and_reports_missing(tmp_path):
    plan=plan_fixture(tmp_path)
    resolved=cli._plan(tmp_path,plan)
    args=Namespace(work_root=tmp_path,plan=plan,all=True,group=None)
    with patch.object(cli,"resolve_work_root",return_value=tmp_path), \
         patch.object(cli,"_run_group",return_value=(2,"failed-run")) as run, \
         patch.object(cli.workspace,"run_dir",return_value=tmp_path), \
         patch.object(cli.harvest,"harvest") as harvest_run, \
         patch.object(cli,"_plan",return_value=resolved), \
         patch.object(cli,"cmd_report",return_value=2) as report:
        assert cli.cmd_run(args)==2
        assert run.call_count==harvest_run.call_count==1
        assert report.call_args.args[0].result==["failed-run"]


def test_run_all_rejects_precision_before_launch(tmp_path):
    groups=resolve()
    args=Namespace(work_root=tmp_path,plan="test",all=True,group=None)
    with patch.object(cli,"resolve_work_root",return_value=tmp_path), \
         patch.object(cli,"_plan",return_value=({"profile":"stock","groups":groups},tmp_path)), \
         patch.object(cli,"_run_group") as run:
        assert cli.cmd_run(args)==2
        run.assert_not_called()


@pytest.mark.parametrize("key,value",[("hot_n",0),("hot_n",True),("hot_warmup_seconds",3600)])
def test_invalid_target_hot_geometry(key,value):
    raw=q.load_recipe(RECIPE)
    raw["target"]["gfx1030"][key]=value
    with pytest.raises(a.ScreenError):
        resolve("gfx1030",raw)
