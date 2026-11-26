"""Resolve independent GPU workload recipes; never select/apply power settings."""
from __future__ import annotations

import re
from pathlib import Path
import tomllib

from . import ai_probe as a

MIB = 2**20
# The native probe's hipBLASLt preference limit, not a measured allocation.
LT_WORKSPACE_LIMIT = 32*MIB
GROUP_FIELDS = {"matrix", "hot", "dtypes", "n", "batch", "seconds", "pause_ms", "matrix_shape",
                "memory_fraction", "resident_fraction", "warmup_seconds",
                "min_hotspot_c", "setup_timeout"}


def _need(ok, why):
    if not ok:
        raise a.ScreenError(why)


def _integer(value, lo, hi):
    return type(value) is int and lo <= value <= hi


def _names(values):
    return (isinstance(values, list) and bool(values)
            and all(isinstance(v, str) and re.fullmatch(r"[a-z0-9-]+", v) for v in values)
            and len(values) == len(set(values)))


def _fraction(value, total):
    _need(isinstance(value, list) and len(value) == 2
          and all(type(x) is int for x in value) and 0 < value[0] < value[1] <= 1000,
          "invalid memory fraction")
    return total*value[0]//value[1]


def load_recipe(path: Path):
    return tomllib.loads(path.read_text(encoding="utf-8"))


def resolve(raw: dict, recipe: str, arch: str, total: int, free: int,
            headroom: int, max_hotspot: int, max_memory: int) -> dict:
    _need(set(raw) == {"schema", "target", "recipe", "group"} and raw["schema"] == 2,
          "invalid qualification recipe schema")
    _need(arch in raw["target"] and recipe in raw["recipe"], "unknown target/recipe")
    _need(_integer(total, 1, 2**50) and _integer(free, 1, total)
          and _integer(headroom, 256, 65536) and free > headroom*MIB,
          "invalid physical memory/headroom")
    _need(_integer(max_hotspot, 1, 110) and _integer(max_memory, 1, 110),
          "invalid thermal ceiling")
    target = raw["target"][arch]
    _need({"provider","matrix","hot_dtype"} <= set(target) <= {"provider","matrix","hot_dtype","precision_references","hot_n","hot_warmup_seconds"} and target["provider"] in a.PROVIDERS
          and _names(target["matrix"]) and set(target["matrix"]) <= set(a.MATRIX_DTYPES)
          and target["hot_dtype"] in target["matrix"]
          and ("hot_n" not in target or _integer(target["hot_n"],16,2048))
          and ("hot_warmup_seconds" not in target or _integer(target["hot_warmup_seconds"],0,3599))
          and not (target["provider"] == "rocblas" and "fp8" in target["matrix"]),
          "invalid target capability configuration")
    precision = target.get("precision_references",False)
    _need(type(precision) is bool and (not precision or arch=="gfx1100"),"invalid precision reference target")
    selected = raw["recipe"][recipe]
    _need({"groups"} <= set(selected) <= {"groups", "overrides"}
          and _names(selected["groups"]), "invalid group selection")
    overrides = selected.get("overrides", {})
    _need(isinstance(overrides, dict) and set(overrides) <= set(selected["groups"]),
          "override must name a selected group")
    groups = {}
    for name in selected["groups"]:
        _need(name in raw["group"], "unknown selected group")
        g = raw["group"][name]
        if name in overrides:
            override = overrides[name]
            _need(isinstance(override, dict) and bool(override)
                  and set(override) <= {"seconds", "pause_ms", "memory_fraction"}
                  and not g.get("hot") and "min_hotspot_c" not in g
                  and "warmup_seconds" not in g
                  and ("memory_fraction" not in override or "memory_fraction" in g),
                  "invalid short-recipe override")
            g = g | override
        _need(not set(g)-GROUP_FIELDS and sum(k in g for k in ("matrix","dtypes","hot"))==1
              and ("matrix" not in g or g["matrix"] is True)
              and ("hot" not in g or g["hot"] is True), "invalid group fields")
        dtypes = target["matrix"] if g.get("matrix") else [target["hot_dtype"]] if g.get("hot") else g["dtypes"]
        _need(_names(dtypes) and set(dtypes) <= set(a.DTYPES)
              and set(dtypes)&set(a.MATRIX_DTYPES) <= set(target["matrix"]),
              "unknown or unconfigured matrix case")
        if precision:
            dtypes = [a.PRECISION_CASES.get(d,d) for d in dtypes]
        missing = [d for d in dtypes if d in a.WMMA_CASES and arch not in a.WMMA_ARCHES]
        if missing:
            _need(missing == dtypes, "split supported and unsupported cases into separate groups")
            groups[name] = {"status": "not_applicable", "reason": "native WMMA unsupported",
                            "dtypes": dtypes}
            continue
        n, batch = g.get("n", 768), g.get("batch", 1)
        sec, warm, timeout = g["seconds"], g.get("warmup_seconds", 0), g.get("setup_timeout", 60)
        pause, hot = g["pause_ms"], g.get("min_hotspot_c")
        if g.get("hot"):
            n = target.get("hot_n",n)
            warm = target.get("hot_warmup_seconds",warm)
        _need(_integer(n,16,2048) and _integer(batch,1,256) and _integer(sec,1,3600)
              and _integer(warm,0,3600-sec) and _integer(timeout,1,300)
              and isinstance(pause,list) and bool(pause) and all(_integer(x,0,1000) for x in pause)
              and len(pause)==len(set(pause)) and (not warm or hot is not None)
              and (hot is None or _integer(hot,1,max_hotspot-1)), "invalid group bounds")
        _need(not any(d in a.PRECISION_DTYPES for d in dtypes) or (arch=="gfx1100" and n<=64 and all(d in a.PRECISION_DTYPES for d in dtypes)), "invalid precision reference scope")
        matrix = any(d in a.MATRIX_DTYPES for d in dtypes)
        shape = g.get("matrix_shape")
        _need(shape is None or matrix, "shape requires matrix case")
        geometry = a.matrix_geometry(n, shape) if matrix else None
        mem, resident = None, 0
        budget = free-headroom*MIB
        if "memory_fraction" in g:
            _need(dtypes == ["memory"] and "resident_fraction" not in g, "memory group must be isolated")
            mem = min(_fraction(g["memory_fraction"], total), budget)//(2*MIB)*2
            _need(2 <= mem <= 65536, "memory budget unavailable")
        else:
            _need("memory" not in dtypes, "memory requires explicit fraction")
        if "resident_fraction" in g:
            _need(matrix and all(d in a.MATRIX_DTYPES for d in dtypes), "resident group must be matrix-only")
            overhead = max(a.matrix_allocation(d, geometry,
                          LT_WORKSPACE_LIMIT if target["provider"]=="hipblaslt" else 0,batch)
                           for d in dtypes)
            resident = min(_fraction(g["resident_fraction"], total), budget-overhead)//MIB
            _need(1 <= resident <= 65536, "resident budget unavailable")
        # These are typed options of the existing executor, not executable argv.
        groups[name] = {"status": "planned", "options": {
            "dtype": list(dtypes), "provider": target["provider"], "n": n, "batch": batch,
            "matrix_shape": shape, "seconds": sec, "warmup_seconds": warm,
            "min_hotspot_c": hot, "max_hotspot_c": max_hotspot, "max_memory_c": max_memory,
            "pause_ms": pause, "setup_timeout": timeout, "memory_mib": mem, "resident_mib": resident,
            "headroom_mib": headroom}}
    _need(any(g["status"]=="planned" for g in groups.values()), "no executable group")
    return groups


def workload_seconds(groups: dict) -> int:
    """Nominal native load budget, excluding setup, teardown and lease waits."""
    return sum((g["options"]["seconds"] + g["options"]["warmup_seconds"])
               * len(g["options"]["dtype"]) * len(g["options"]["pause_ms"])
               for g in groups.values() if g["status"] == "planned")


def native_scope(doc: dict | None, arch: str) -> dict:
    """Retain the native planner's scope, not a second kernel taxonomy or a pass."""
    if doc is None:
        return {"status": "not_supplied", "qualification": "not_run"}
    def names(value):
        return isinstance(value,list) and all(isinstance(v,str) and bool(v) for v in value) and len(set(value))==len(value)
    _need(isinstance(doc,dict) and doc.get("arch")==arch and doc.get("outcome")=="diagnostic"
          and doc.get("qualification")=="not_run" and isinstance(doc.get("release"),str)
          and bool(doc["release"]) and isinstance(doc.get("runs"),list)
          and names(doc.get("selected_units")) and names(doc.get("uncovered_units"))
          and set(doc["uncovered_units"])<=set(doc["selected_units"]), "invalid native qualification plan")
    requested=doc.get("requested_rows",[])
    _need(isinstance(requested,list) and (bool(doc["selected_units"]) or bool(requested)),
          "empty native qualification selection")
    routes=[]
    route_keys=set()
    for item in doc["runs"]:
        _need(isinstance(item,dict) and item.get("kind") in ("family","overlay")
              and isinstance(item.get("name"),str) and bool(item["name"])
              and names(item.get("units")) and bool(item["units"])
              and bool(set(item["units"])&set(doc["selected_units"]))
              and isinstance(item.get("selftest_case"),str) and bool(item["selftest_case"])
              and names(item.get("other_rows_need_evidence",[]))
              and type(item.get("reference_required",False)) is bool, "invalid native route")
        key=(item["kind"],item["name"])
        _need(key not in route_keys, "duplicate native route")
        route_keys.add(key)
        routes.append({k:item[k] for k in ("kind","name","units","selftest_case","support_units",
                        "base_rows","overlay_rows","other_rows_need_evidence","reference_required") if k in item})
    covered={u for r in routes for u in r["units"]}
    _need(set(doc["selected_units"])-covered<=set(doc["uncovered_units"]),
          "native plan hides uncovered units")
    row_names=set()
    for row in requested:
        _need(isinstance(row,dict) and isinstance(row.get("row"),str) and bool(row["row"])
              and row["row"] not in row_names and row.get("status") in ("configured_route","uncovered")
              and isinstance(row.get("routes"),list),"invalid native row scope")
        row_names.add(row["row"])
        if row["status"]=="configured_route":
            _need(len(row["routes"])==1 and (row["routes"][0]["kind"],row["routes"][0]["name"]) in route_keys,
                  "requested row lacks planned route")
    digests={}
    for key in ("reference_sha256","family_policy_sha256"):
        if key in doc:
            _need(isinstance(doc[key],str) and re.fullmatch(r"[0-9a-f]{64}",doc[key]),"invalid native input digest")
            digests[key]=doc[key]
    gaps=[]
    graph=doc.get("model_graph")
    if graph is not None:
        _need(isinstance(graph,dict) and isinstance(graph.get("sha256"),str)
              and re.fullmatch(r"[0-9a-f]{64}",graph["sha256"])
              and isinstance(graph.get("declared"),list) and bool(graph["declared"])
              and all(isinstance(r,dict) and r.get("row") in row_names for r in graph["declared"]),
              "invalid model graph scope")
        gaps.append("model graph declaration is not execution evidence")
    if doc["uncovered_units"]: gaps.append("uncovered units")
    if any(r["status"]=="uncovered" for r in requested): gaps.append("uncovered requested rows")
    if any(r.get("other_rows_need_evidence") for r in routes): gaps.append("additional runtime rows need evidence")
    if any(r.get("reference_required") for r in routes): gaps.append("compatible base reference missing")
    if doc.get("inventory_kind")!="matching_libr3_membership": gaps.append("actual configured inventory not verified")
    if len(digests)!=2: gaps.append("native reference/policy identity incomplete")
    return {"status":"incomplete" if gaps else "planned","release":doc["release"],
            "inventory_kind":doc.get("inventory_kind"),"selected_units":doc["selected_units"],
            "uncovered_units":doc["uncovered_units"],"requested_rows":requested,
            "affected_scope":doc.get("affected_scope"),"pin_triage":doc.get("pin_triage"),
            "routes":routes,"gaps":gaps,"model_graph":graph,**digests,"qualification":"not_run"}
