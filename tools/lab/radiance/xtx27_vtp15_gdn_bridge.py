"""VTP15: original BF16 Qwen GDN projection -> convolution -> recurrence bridge.

This card-free, one-layer, short-sequence reference consumes ORIGINAL
Qwen3.8-27B layer-0 BF16 QKV/a/b/z projection outputs computed from the
checkpoint and verified by a separate three-GPU projection experiment.
It then uses the ORIGINAL BF16 conv1d, A_log, dt_bias and GDN norm
weights to compute head-local recurrent outputs, checking against an
independent global-layout computation.

This closes the synthetic-PROJECTED-input gap at the GDN core only.
Input hidden activations are deterministic, not a real prior layer;
output projection/all-reduce, engine scheduling, prefill KKT, cache
continuation and long-context serving remain open.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from datetime import UTC, datetime
from pathlib import Path

import xtx27_vtp13_gdn_pack as gdn
import xtx27_vtp14_gdn_exact as exact
import xtx27_vtp15_projection_ref as projection

ROOT = exact.ROOT
SCRATCH = Path(os.environ.get("BC_WORK_ROOT", ROOT.parent)) / "xtx27-prefill" / "vtp15-gdn-core"
SCHEMA = "vtp15-qwen38-27b-original-bf16-GDN-core-CPU-global-shard-v1"
ROLES = ("xtx0", "xtx1", "r9700")


def _projection_values(receipt: dict, role: str, op: str, rows: int, count: int) -> list[list[float]]:
    entry = receipt["rank_results"][role][op]
    if entry["shape"] != [rows, count]:
        raise ValueError("wrong BF16 projected-activation geometry")
    source = Path(entry["bf16"])
    blob = source.read_bytes()
    if len(blob) != rows * count * 2 or hashlib.sha256(blob).hexdigest() != entry["bf16_sha256"]:
        raise ValueError("original BF16 projection reference changed")
    floats = exact.bf16_values(blob)
    if not all(math.isfinite(x) for x in floats):
        raise ValueError("nonfinite original BF16 projection")
    return [floats[r * count:(r + 1) * count] for r in range(rows)]


def _global_from_rank(local: dict, rows: int) -> tuple[list, list, list, list]:
    qkv = [[None] * 10240 for _ in range(rows)]
    a = [[None] * 48 for _ in range(rows)]
    b = [[None] * 48 for _ in range(rows)]
    z = [[None] * 6144 for _ in range(rows)]
    vhead=0
    for role, nval in zip(ROLES, (12, 12, 24), strict=True):
        segments = gdn.layout()[role]
        for key, n in (("qkv", sum(width for _, width in segments)),
                       ("a", nval), ("b", nval), ("z", nval * 128)):
            block = local[role][key]
            if len(block) != rows or any(len(row) != n for row in block):
                raise ValueError(f"{role}.{key}: truncated or malformed projection rows")
        for t in range(rows):
            rank_q = local[role]["qkv"][t]
            cursor = 0
            for start, width in segments:
                if any(value is not None for value in qkv[t][start:start+width]):
                    raise ValueError("GDN QKV global projection gap/overlap")
                qkv[t][start:start+width] = rank_q[cursor:cursor+width]
                cursor += width
            a[t][vhead:vhead+nval] = local[role]["a"][t]
            b[t][vhead:vhead+nval] = local[role]["b"][t]
            z[t][vhead*128:(vhead+nval)*128] = local[role]["z"][t]
        vhead+=nval
    if vhead != 48 or any(any(v is None for v in row) for block in (qkv,a,b,z) for row in block):
        raise ValueError("incomplete original-projection global coverage")
    return qkv,a,b,z


def verify(weights: Path, projected: Path, pack_receipt: Path,
           *, output_dir: Path | None = None) -> dict:
    w = projection.validate(weights)
    refs = json.loads(projected.read_text())
    if (refs.get("status") != "ORIGINAL_BF16_QWEN_GDN_INPUT_PROJECTIONS_CPU_PASS_NOT_GPU"
            or refs.get("weight_pack_sha256") != projection.sha(weights)
            or refs.get("writer_sha256") != projection.sha(Path(projection.__file__))
            or refs.get("oracle_cpp_sha256") != projection.sha(projection.SOURCE)):
        raise ValueError("projection reference provenance mismatch")
    pack = json.loads(pack_receipt.read_text())
    if (pack.get("status") != "EXACT_ORIGINAL_BF16_GDN_QKV_CONV_PACK_PASS_NOT_RUNTIME"
            or w["qkv_pack_receipt"] != str(pack_receipt.resolve())
            or pack["layer"] != w["layer"]):
        raise ValueError("GDN convolution pack does not match original checkpoint")
    rows=w["rows"]
    if type(rows) is not int or not 1<=rows<=8:
        raise ValueError("bounded to M1-M8")
    local={}
    for role, keys, values in zip(ROLES,(4,4,8),(12,12,24),strict=True):
        local[role] = {
            "qkv": _projection_values(refs,role,"qkv",rows,(2*keys+values)*128),
            "a": _projection_values(refs,role,"a",rows,values),
            "b": _projection_values(refs,role,"b",rows,values),
            "z": _projection_values(refs,role,"z",rows,values*128),
        }
    qkv,glob_a,glob_b,glob_z = _global_from_rank(local,rows)
    origin=gdn.sources(Path(w["checkpoint"]),w["layer"])
    source=origin["sources"]["conv1d"]
    with Path(source["path"]).open("rb") as f:
        f.seek(source["offset"])
        blob=f.read(source["bytes"])
    if hashlib.sha256(blob).hexdigest()!=pack["outputs"]["conv1d"]["source_payload_sha256"]:
        raise ValueError("original BF16 convolution payload changed")
    weights_conv=exact.bf16_values(blob)
    conv_global = [
        [exact._bf16_conv(weights_conv,c,t,
                          synthetic=lambda tt,cc:qkv[tt][cc]) for c in range(10240)]
        for t in range(rows)
    ]
    conv_rank={}
    max_conv=0.0
    for role in ROLES:
        source_file=pack["outputs"]["conv1d"]["ranks"][role]
        rank_weights_path=Path(source_file["path"])
        raw=rank_weights_path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=source_file["sha256"]:
            raise ValueError(f"{role} original BF16 convolution pack changed")
        conv_weights=exact.bf16_values(raw)
        channels=[j for start,width in gdn.layout()[role] for j in range(start,start+width)]
        output=[]
        for t in range(rows):
            row=[]
            for j,global_c in enumerate(channels):
                acc=0.0
                for tap in range(4):
                    prev=t+tap-3
                    if prev>=0:
                        acc+=local[role]["qkv"][prev][j]*conv_weights[4*j+tap]
                value=exact.round_bf16(exact.silu(acc))
                max_conv=max(max_conv,abs(conv_global[t][global_c]-value))
                row.append(value)
            output.append(row)
        conv_rank[role]=output
    if max_conv:
        raise ValueError("global original-BF16 versus physical-rank convolution mismatch")

    param,param_hash=exact.original_parameters(Path(w["checkpoint"]),w["layer"])
    maxout,maxstate=0.0,0.0
    result={}
    global_h=0
    for role,nkeys,nvalues in zip(ROLES,(4,4,8),(12,12,24),strict=True):
        projected_values=[]
        for h in range(nvalues):
            key=h//3
            ql,kl,vl=exact.head_local_slice(conv_rank[role],nkeys,nvalues,key,h)
            qg,kg,vg=exact.head_slice(conv_global,(global_h+h)//3,global_h+h)
            state_global=[[0.0]*128 for _ in range(128)]
            state_local=[[0.0]*128 for _ in range(128)]
            for t in range(rows):
                coeff_a=local[role]["a"][t][h]
                coeff_b=local[role]["b"][t][h]
                gate_z=local[role]["z"][t][h*128:(h+1)*128]
                if coeff_a != glob_a[t][global_h+h] or coeff_b != glob_b[t][global_h+h] or (
                    gate_z != glob_z[t][(global_h+h)*128:(global_h+h+1)*128]
                ):
                    raise ValueError("global and rank-local projected gates diverged")
                alog=param["A_log"][global_h+h]
                dt=param["dt_bias"][global_h+h]
                out_global=exact.step_global(
                    state_global,qg[t],kg[t],vg[t],coeff_a,coeff_b,gate_z,alog,dt,param["norm"])
                out_local=exact.step_local(
                    state_local,ql[t],kl[t],vl[t],coeff_a,coeff_b,gate_z,alog,dt,param["norm"])
                maxout=max(maxout,max(abs(x-y) for x,y in zip(out_global,out_local,strict=True)))
                projected_values.extend(out_local)
            maxstate=max(maxstate,max(
                abs(state_global[e][d]-state_local[d][e])
                for e in range(128) for d in range(128)))
        data=struct.pack("<"+str(len(projected_values))+"f",*projected_values)
        result[role]={"heads":nvalues,"shape":[nvalues,rows,128],
                      "bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}
        if output_dir is not None:
            path=output_dir/f"gdn-core-{role}.f32"
            with path.open("xb") as f:
                f.write(data)
            result[role]["output"]=str(path)
        global_h+=nvalues
    if global_h != 48 or maxout>5e-5 or maxstate>5e-5:
        raise ValueError(f"rank-local GDN diverges from global: {maxout}, {maxstate}")
    return {
        "schema": SCHEMA,
        "status": "ORIGINAL_BF16_QWEN_GDN_CORE_CPU_GLOBAL_SHARD_PASS_NOT_GPU",
        "rows":rows,"layer":w["layer"],
        "weight_pack_sha256":projection.sha(weights),
        "projected_input_ref_sha256":projection.sha(projected),
        "conv_pack_sha256":projection.sha(pack_receipt),
        "parameter_hashes":param_hash,
        "global_rank_bf16_conv_max_abs":max_conv,
        "global_rank_recurrent_output_max_abs":maxout,
        "global_rank_recurrent_state_max_abs":maxstate,
        "original_checkpoint_projection_weights":True,
        "original_checkpoint_conv_decay_norm":True,
        "rank_outputs":result,
        "limitations":["synthetic initial hidden", "no GPU core execution with real projections yet",
                       "no output projection", "no multi-step cache continuation",
                       "no long-context prefill KKT", "no full-model serving"],
    }


def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights-receipt",type=Path,required=True)
    p.add_argument("--projected-receipt",type=Path,required=True)
    p.add_argument("--qkv-pack-receipt",type=Path,required=True)
    p.add_argument("--phase",choices=("plan","verify"),default="plan")
    p.add_argument("--run",action="store_true")
    p.add_argument("--output",type=Path)
    a=p.parse_args(argv)
    if a.phase=="verify" and not a.run:
        p.error("original checkpoint processing requires --run")
    if a.phase=="plan":
        print(json.dumps({
            "status":"CARD_FREE_PLAN",
            "weights_receipt":str(a.weights_receipt),
            "projected_receipt":str(a.projected_receipt),
            "qkv_pack_receipt":str(a.qkv_pack_receipt),
            "gpu_required":False,
        },indent=2))
        return 0
    output=(a.output.expanduser().resolve() if a.output else
            SCRATCH/datetime.now(UTC).strftime("layer0-%Y%m%dT%H%M%S%fZ"))
    if output.exists() or output==ROOT or ROOT in output.parents:
        p.error("must use unique outside-repo receipt directory")
    output.mkdir(parents=True,exist_ok=False)
    receipt={"status":"INCOMPLETE",
             "writer_sha256":projection.sha(Path(__file__))}
    result=output/"summary.json"
    result.write_text(json.dumps(receipt,indent=2)+"\n")
    try:
        receipt.update(verify(a.weights_receipt,a.projected_receipt,
                              a.qkv_pack_receipt,output_dir=output))
    except Exception as exc:
        receipt.update({"status":"FAILED_INCOMPLETE","error":str(exc)})
        result.write_text(json.dumps(receipt,indent=2)+"\n")
        raise
    result.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({"status":receipt["status"],"receipt":str(result),
                      "max_abs":receipt["global_rank_recurrent_output_max_abs"]},indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
