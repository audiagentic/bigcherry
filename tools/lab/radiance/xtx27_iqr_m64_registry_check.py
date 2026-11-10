"""VTP09 card-free audit of gfx11 IQ6R M17..64 tiled registry selection.

Loads a built libr3.so, never runs kernels or touches GPUs. Fails closed
if the custom row is missing, its constraints drift, its function pointers
differ from upstream tiled, or narrow decode loses any original shape.
"""
from __future__ import annotations

import argparse
import ctypes as c
import hashlib
import json
from pathlib import Path


class CConstraint(c.Structure):
    _fields_ = [
        ("key", c.c_char_p), ("op", c.c_int),
        ("ival", c.c_longlong), ("sval", c.c_char_p),
    ]

class CKernel(c.Structure):
    _fields_ = [
        ("name", c.c_char_p), ("op", c.c_char_p), ("family", c.c_char_p),
        ("computes", c.c_char_p), ("shape", c.c_char_p),
        ("dtypes", c.c_char_p), ("domain", c.c_int), ("priority", c.c_int),
        ("constraints", c.POINTER(CConstraint)), ("n_constraints", c.c_int),
        ("tunables", c.c_void_p), ("n_tunables", c.c_int),
        ("tune_valid", c.c_void_p), ("init", c.c_void_p), ("fini", c.c_void_p),
        ("launch", c.c_void_p), ("scratch", c.c_void_p), ("layout", c.c_void_p),
        ("relayout", c.c_void_p), ("opd_shape", c.c_void_p),
        ("unrelayout", c.c_void_p), ("replaces", c.c_void_p),
        ("capabilities", c.c_uint64),
    ]

NARROW = "gemm_iqr_nt_m64"
TILED = "gemm_iqr_tiled"
CUSTOM = "gemm_iqr_tiled_m64_gfx11"
EARLY = "gemm_iqr_tiled_m17_63_gfx11"
REQUIRED = (NARROW, TILED, CUSTOM, EARLY)
EQ, LE, GE, DIV, IN = 0, 1, 2, 3, 4


def as_text(value: bytes | None) -> str | None:
    return value.decode() if value is not None else None


def row_summary(row: CKernel) -> dict:
    if not 0 <= row.n_constraints <= 100:
        raise ValueError("invalid kernel constraint count")
    return {
        "name": as_text(row.name), "op": as_text(row.op),
        "priority": row.priority, "domain": row.domain,
        "constraints": [
            (as_text(row.constraints[i].key), row.constraints[i].op,
             row.constraints[i].ival, as_text(row.constraints[i].sval))
            for i in range(row.n_constraints)
        ],
        "launch": row.launch, "scratch": row.scratch,
        "layout": row.layout, "relayout": row.relayout,
        "unrelayout": row.unrelayout, "replaces": row.replaces,
    }


def matches(row: dict, *, m: int, n: int, k: int, dtype: str = "i8a8h") -> bool:
    values = {"M": m, "N": n, "K": k, "group": 128, "dtype": dtype}
    for key, op, ival, sval in row["constraints"]:
        value = values.get(key)
        if value is None:
            return False
        if op == EQ and value != ival:
            return False
        if op == LE and value > ival:
            return False
        if op == GE and value < ival:
            return False
        if op == DIV and (not ival or value % ival):
            return False
        if op == IN and (sval is None or str(value) not in sval.split()):
            return False
        if op not in (EQ, LE, GE, DIV, IN):
            return False
    return True


def selected(rows: dict[str, dict], *, m: int, n: int, k: int) -> str | None:
    eligible = [row for row in rows.values()
                if matches(row, m=m, n=n, k=k)]
    return max(eligible, key=lambda row: row["priority"])["name"] if eligible else None


def validate(rows: dict[str, dict], *, active: bool) -> dict:
    if not active or set(rows) != set(REQUIRED):
        raise ValueError("gfx11 IQR tiled crossover overlays missing or unexpected")
    narrow, tiled, m64, early = (rows[name] for name in REQUIRED)
    if any(row["op"] != "gemm_nt_q" for row in (narrow, tiled, m64, early)):
        raise ValueError("unexpected kernel operation")
    if not (m64["priority"] == early["priority"] == narrow["priority"] + 1
            and narrow["priority"] > tiled["priority"]):
        raise ValueError("wrong row selection priority")
    if ("M", LE, 64, None) not in narrow["constraints"]:
        raise ValueError("original narrow M<=64 must remain intact")
    if ("N", DIV, 16, None) not in narrow["constraints"]:
        raise ValueError("original N16 fallback missing")
    if ("N", DIV, 128, None) not in tiled["constraints"]:
        raise ValueError("tiled N128 legality missing")
    if m64["constraints"] != tiled["constraints"] + [("M", EQ, 64, None)]:
        raise ValueError("M64 overlay must retain exact tiled constraints plus M==64")
    if early["constraints"] != tiled["constraints"] + [
        ("M", GE, 17, None), ("M", LE, 63, None),
        ("K", EQ, 5120, None), ("N", IN, 0, "5120 10240"),
    ]:
        raise ValueError("early overlay must retain tiled legality and measured 27B FFN geometry")
    for row in (m64, early):
        if any(not row[key] or row[key] != tiled[key]
               for key in ("launch", "scratch", "layout", "relayout")):
            raise ValueError("overlay must call identical real tiled hooks")
        if (row["domain"] != tiled["domain"]
                or row["unrelayout"] != tiled["unrelayout"]
                or row["replaces"] != tiled["replaces"]):
            raise ValueError("overlay kernel ABI differs from native tiled")
    bands = (
        (1, 10240, NARROW),
        (8, 10240, NARROW),
        (16, 10240, NARROW),
        (17, 10240, EARLY),
        (20, 10240, EARLY),
        (24, 10240, EARLY),
        (28, 10240, EARLY),
        (31, 10240, EARLY),
        (32, 10240, EARLY),
        (48, 10240, EARLY),
        (63, 10240, EARLY),
        (64, 10240, CUSTOM),
        (64, 5120, CUSTOM),
        (65, 10240, TILED),
        (128, 10240, TILED),
        (17, 48, NARROW),
        (63, 48, NARROW),
        (64, 16, NARROW),
        (64, 48, NARROW),
        (17, 128, NARROW),
        (63, 128, NARROW),
        (17, 5120, EARLY),
        (63, 5120, EARLY),
        (17, 8192, NARROW),
        (63, 8192, NARROW),
        (64, 128, CUSTOM),
    )
    for m, n, want in bands:
        actual = selected(rows, m=m, n=n, k=5120)
        if actual != want:
            raise ValueError(f"selection failed M={m} N={n}: expected {want}, got {actual}")
    invalid_k_cases = ((17, 5120, 1024), (32, 10240, 4096),
                       (63, 5120, 2560))
    for m, n, k in invalid_k_cases:
        if selected(rows, m=m, n=n, k=k) != NARROW:
            raise ValueError(f"unmeasured K={k} would steal narrow fallback")
    return {
        "status": "GFX11_IQR_M17_64_TILED_OVERLAYS_PRESENT_EXACT_AND_SHAPE_SAFE",
        "tested_bands": len(bands) + len(invalid_k_cases),
        "new_rows": [EARLY, CUSTOM],
        "narrow_fallback_remains": True,
        "device_kernels_executed": False,
    }


def read_plugin(path: Path) -> tuple[dict[str, dict], int]:
    plugin = c.CDLL(str(path.resolve()))
    plugin.rad_kernel_count.restype = c.c_int
    plugin.rad_kernel_at.argtypes = [c.c_int]
    plugin.rad_kernel_at.restype = c.c_void_p
    plugin.r3_iqr_m64_tiled_override.restype = c.c_int
    plugin.r3_iqr_m17_63_tiled_override.restype = c.c_int
    count = plugin.rad_kernel_count()
    if count <= 0 or count > 3000:
        raise ValueError("invalid plugin kernel count")
    rows: dict[str, dict] = {}
    for i in range(count):
        address = plugin.rad_kernel_at(i)
        if not address:
            raise ValueError("null plugin row")
        row = c.cast(address, c.POINTER(CKernel)).contents
        name = as_text(row.name)
        if name in REQUIRED:
            if name in rows:
                raise ValueError("duplicate required kernel row")
            rows[name] = row_summary(row)
    return rows, int(plugin.r3_iqr_m64_tiled_override() == 1
                     and plugin.r3_iqr_m17_63_tiled_override() == 1)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plugin", type=Path, required=True)
    p.add_argument("--receipt", type=Path)
    a = p.parse_args(argv)
    if a.receipt and a.receipt.exists():
        raise FileExistsError(f"refusing existing receipt: {a.receipt}")
    if not a.plugin.is_file() or a.plugin.is_symlink():
        p.error("regular existing libr3 plugin required")
    rows, status = read_plugin(a.plugin)
    result = validate(rows, active=(status == 1))
    result.update({"plugin_sha256": sha256(a.plugin),
                   "plugin": str(a.plugin.resolve()),
                   "kernel_rows": {k: {
                       "priority": row["priority"],
                       "constraints": row["constraints"],
                   } for k, row in rows.items()}})
    output = json.dumps(result, indent=2) + "\n"
    if a.receipt:
        a.receipt.parent.mkdir(parents=True, exist_ok=True)
        with a.receipt.open("x") as file:
            file.write(output)
        print(f"receipt={a.receipt} status={result['status']}")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
