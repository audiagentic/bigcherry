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
        assert groups["hot"]["options"]["min_hotspot_c"]==60
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
    # Only the two groups necessary to exercise pass and missing evidence.
    doc={"schema":1,"protocol":a.NATIVE_PROTOCOL,"screen_schema":a.SCREEN_SCHEMA,
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
