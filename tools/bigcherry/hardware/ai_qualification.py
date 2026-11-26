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
    _need({"provider","matrix","hot_dtype"} <= set(target) <= {"provider","matrix","hot_dtype","precision_references"} and target["provider"] in a.PROVIDERS
          and _names(target["matrix"]) and set(target["matrix"]) <= set(a.MATRIX_DTYPES)
          and target["hot_dtype"] in target["matrix"]
          and not (target["provider"] == "rocblas" and "fp8" in target["matrix"]),
          "invalid target capability configuration")
    precision = target.get("precision_references",False)
    _need(type(precision) is bool and (not precision or arch=="gfx1100"),"invalid precision reference target")
    selected = raw["recipe"][recipe]
    _need(set(selected) == {"groups"} and _names(selected["groups"]), "invalid group selection")
    groups = {}
    for name in selected["groups"]:
        _need(name in raw["group"], "unknown selected group")
        g = raw["group"][name]
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


def native_scope(doc: dict | None, arch: str) -> dict:
    """Consume the RCS27 planned routes; never run embedded build/measure args."""
    if doc is None:
        return {"status": "not_supplied", "qualification": "not_run"}
    _need(isinstance(doc, dict) and doc.get("arch")==arch and doc.get("outcome")=="diagnostic"
          and doc.get("qualification")=="not_run" and isinstance(doc.get("release"),str)
          and bool(doc["release"]) and isinstance(doc.get("runs"),list)
          and isinstance(doc.get("selected_units"),list) and bool(doc["selected_units"])
          and isinstance(doc.get("uncovered_units"),list), "invalid native qualification plan")
    names=set()
    routes=[]
    for item in doc["runs"]:
        _need(isinstance(item,dict) and item.get("kind") in ("family","overlay")
              and isinstance(item.get("name"),str) and bool(item["name"])
              and isinstance(item.get("units"),list) and bool(item["units"])
              and isinstance(item.get("selftest_case"),str) and bool(item["selftest_case"]),
              "invalid native route")
        key=(item["kind"],item["name"])
        _need(key not in names, "duplicate native route")
        names.add(key)
        routes.append({k:item[k] for k in ("kind","name","units","selftest_case",
                        "other_rows_need_evidence","reference_required") if k in item})
    return {"status": "planned", "release": doc["release"],
            "inventory_kind": doc.get("inventory_kind"), "selected_units": doc["selected_units"],
            "uncovered_units": doc["uncovered_units"], "routes": routes,
            "qualification": "not_run"}
