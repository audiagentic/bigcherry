"""VTP14: original BF16 Qwen3.8 GDN conv + exact recurrent math oracle (CPU).

This is an intentionally card-free operator proof. Ingress Q|K|V is a
deterministic BF16-representable *preprojected* activation, not the result
of multiplying the checkpoint's in_proj_qkv weights. Actual checkpoint
conv1d, A_log, dt_bias, and shared norm gains ARE consumed and hashed.
Scalar a, b, z projection outputs are deterministic BF16 inputs. The
Qwen decode equations are faithfully implemented: causal depthwise
convolution -> SiLU -> BF16 store, per-head L2 Q/K normalization,
g=-exp(A_log)*softplus(a+dt_bias), beta=sigmoid(b), decayed delta
rule, and gated per-value-head RMSNorm with SiLU(z). No speculative
rollback, no chunk-KKT algorithm, no out_proj, no GPU.
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

import xtx27_checkpoint_audit as audit
import xtx27_vtp13_gdn_pack as gdn
import xtx27_vtp13_ownership as ownership

ROOT = Path(__file__).resolve().parents[3]
SCRATCH = Path(os.environ.get("BC_WORK_ROOT", ROOT.parent)) / "xtx27-prefill" / "vtp14-gdn-reference"
HEAD_DIM = 128
HEADS = 48
KEY_HEADS = 16
GROUP = 3
EPS = 1e-6
SCALE = 1.0 / math.sqrt(HEAD_DIM)
PARAMS = {
    "A_log": ("A_log", [48]),
    "dt_bias": ("dt_bias", [48]),
    "norm": ("norm.weight", [128]),
}
ROLES = ("xtx0", "xtx1", "r9700")


def bf16_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def round_bf16(value: float) -> float:
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    # Round to nearest even, not truncate. NaN/Inf are rejected by caller.
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFFFFFF
    return bf16_float(rounded >> 16)


def bf16_values(raw: bytes) -> list[float]:
    if len(raw) % 2:
        raise ValueError("malformed BF16 payload length")
    return [bf16_float(bits) for (bits,) in struct.iter_unpack("<H", raw)]


def original_parameters(checkpoint: Path, layer: int) -> tuple[dict, dict]:
    root = checkpoint.expanduser().resolve(strict=True)
    index, index_hash = audit.load_metadata(root / "model.safetensors.index.json")
    meta, config_hash = audit.load_metadata(root / "config.json")
    ownership.expected_config(meta)
    if type(layer) is not int or not 0 <= layer < 64 or layer % 4 == 3:
        raise ValueError("not a Qwen3.8 GDN layer")
    out = {}
    hashes = {"index_sha256": index_hash, "config_sha256": config_hash}
    weight_map = index["weight_map"]
    cache = {}
    for key, (name_suffix, expected_shape) in PARAMS.items():
        name = f"model.language_model.layers.{layer}.linear_attn.{name_suffix}"
        filename = weight_map.get(name)
        if not audit.safe_filename(filename):
            raise ValueError(f"missing unsafe parameter source: {name}")
        path = (root / filename).resolve(strict=True)
        if path.parent != root:
            raise ValueError("parameter source escapes checkpoint")
        if path not in cache:
            info = audit.shard_metadata(path)
            with path.open("rb") as stream:
                length = struct.unpack("<Q", stream.read(8))[0]
                head = json.loads(stream.read(length), object_pairs_hook=audit.unique_keys)
            cache[path] = info, head, 8 + length
        info, header, base = cache[path]
        val = header.get(name)
        if not isinstance(val, dict) or val["dtype"] != "BF16" or val["shape"] != expected_shape:
            raise ValueError(f"unexpected original BF16 {name} geometry")
        start, stop = val["data_offsets"]
        if stop - start != math.prod(expected_shape) * 2:
            raise ValueError("wrong original BF16 byte extent")
        with path.open("rb") as stream:
            stream.seek(base + start)
            raw = stream.read(stop - start)
        if len(raw) != stop - start:
            raise ValueError(f"truncated original BF16 parameter: {name}")
        result = bf16_values(raw)
        if not all(math.isfinite(v) for v in result):
            raise ValueError(f"nonfinite original BF16 parameter: {name}")
        out[key] = result
        hashes[key] = {
            "source": name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "header_sha256": info["header_sha256"],
            "shape": expected_shape,
        }
    return out, hashes


def fake_projection(t: int, global_c: int) -> float:
    # x in the checkpoint's original [Q(2048)|K(2048)|V(6144)] channel order.
    # Exactly BF16-representable, no floating ambiguity before convolution.
    return ((t * 7 + global_c * 11) % 23 - 11) * (1.0 / 64.0)


def scalar_projection(t: int, h: int, kind: str) -> float:
    offsets = {"a": 5, "b": 11, "z": 19}
    return ((t * 13 + h * 17 + offsets[kind]) % 31 - 15) / 64.0


def sigmoid(x: float) -> float:
    return (1 / (1 + math.exp(-x))) if x >= 0 else (math.exp(x) / (1 + math.exp(x)))


def softplus(x: float) -> float:
    if x > 20:
        return x
    return math.log1p(math.exp(-abs(x))) + max(x, 0.0)


def silu(x: float) -> float:
    return x * sigmoid(x)


def _bf16_conv(weights: list[float], channel: int, t: int,
               synthetic=fake_projection) -> float:
    # Causal depthwise convolution width 4, initial history zero.
    base = channel * 4
    acc = 0.0
    for j in range(4):
        src = t + j - 3
        if src >= 0:
            acc += synthetic(src, channel) * weights[base + j]
    return round_bf16(silu(acc))


def conv_global(weights: list[float], rows: int) -> list[list[float]]:
    if len(weights) != 10240 * 4 or not 1 <= rows <= 8:
        raise ValueError("expected original 10240x4 depthwise convolution")
    # The global path reads the original unsharded channel order.
    return [[_bf16_conv(weights, c, t) for c in range(10240)] for t in range(rows)]


def conv_packed_rank(weights: list[float], role: str, rows: int) -> list[list[float]]:
    segments = gdn.layout()[role]
    count = sum(n for _, n in segments)
    if len(weights) != count * 4:
        raise ValueError("rank-local original conv tensor shape mismatch")
    global_channels = [idx for begin, n in segments for idx in range(begin, begin + n)]
    result = []
    # A separate local-channel-first expression acts as the packed oracle.
    for t in range(rows):
        output = []
        for local_c, global_c in enumerate(global_channels):
            accum = 0.0
            for tap in range(4):
                source_t = t - (3 - tap)
                if source_t >= 0:
                    accum += fake_projection(source_t, global_c) * weights[4 * local_c + tap]
            output.append(round_bf16(silu(accum)))
        result.append(output)
    return result


def head_slice(conv_rows: list[list[float]], key_head: int, value_head: int) -> tuple[list, list, list]:
    # Global reference q:0-2048, k:2048-4096, v:4096-10240.
    q, k, v = [], [], []
    for row in conv_rows:
        q.append(row[key_head * 128:(key_head + 1) * 128])
        k.append(row[2048 + key_head * 128:2048 + (key_head + 1) * 128])
        v.append(row[4096 + value_head * 128:4096 + (value_head + 1) * 128])
    return q, k, v


def head_local_slice(conv_rows: list[list[float]], nk: int, nv: int,
                     key_head: int, value_head: int) -> tuple[list, list, list]:
    # Sharded reference uses re-packed [Q(local)|K(local)|V(local)].
    q, k, v = [], [], []
    for row in conv_rows:
        q.append(row[128 * key_head:128 * (key_head + 1)])
        k.append(row[128 * (nk + key_head):128 * (nk + key_head + 1)])
        v.append(row[128 * (2 * nk + value_head):128 * (2 * nk + value_head + 1)])
    return q, k, v


def step_global(state: list[list[float]], q: list[float], k: list[float],
                v: list[float], a: float, b: float, z: list[float],
                alog: float, dt: float, gain: list[float]) -> list[float]:
    """Independent global-state reference, state indexed [value_dim][key_dim]."""
    invq = SCALE / math.sqrt(sum(x * x for x in q) + EPS)
    invk = 1 / math.sqrt(sum(x * x for x in k) + EPS)
    nq = [u * invq for u in q]
    nk = [u * invk for u in k]
    decay = math.exp(-math.exp(alog) * softplus(a + dt))
    beta = sigmoid(b)
    projected = []
    for e in range(128):
        src = state[e]
        for d in range(128):
            src[d] *= decay
        pred = sum(src[d] * nk[d] for d in range(128))
        delta = beta * (v[e] - pred)
        for d in range(128):
            src[d] += delta * nk[d]
        projected.append(sum(src[d] * nq[d] for d in range(128)))
    rms = 1.0 / math.sqrt(sum(o * o for o in projected) / 128 + EPS)
    return [o * rms * gain[e] * silu(z[e]) for e, o in enumerate(projected)]


def step_local(state: list[list[float]], q: list[float], k: list[float],
               v: list[float], a: float, b: float, z: list[float],
               alog: float, dt: float, gain: list[float]) -> list[float]:
    """Rank-local independent orientation, state indexed [key_dim][value_dim]."""
    qq = [f * SCALE / math.sqrt(EPS + sum(g * g for g in q)) for f in q]
    kk = [f / math.sqrt(EPS + sum(g * g for g in k)) for f in k]
    delta_decay = math.exp(-math.exp(alog) * softplus(a + dt))
    beta = sigmoid(b)
    predicted = [0.0] * 128
    for d, kd in enumerate(kk):
        for e in range(128):
            state[d][e] *= delta_decay
            predicted[e] += state[d][e] * kd
    betas = [beta * (v[e] - predicted[e]) for e in range(128)]
    unnormalised = [0.0] * 128
    for d in range(128):
        kd, qd = kk[d], qq[d]
        for e in range(128):
            state[d][e] += betas[e] * kd
            unnormalised[e] += state[d][e] * qd
    norm = math.sqrt(EPS + sum(x * x for x in unnormalised) / 128)
    return [unnormalised[e] * gain[e] * silu(z[e]) / norm for e in range(128)]


def verify(checkpoint: Path, pack_receipt: Path, rows: int,
           *, output_root: Path | None = None) -> dict:
    if not 1 <= rows <= 8:
        raise ValueError("bounded to 1..8 synthetic input rows")
    pack = json.loads(pack_receipt.read_text())
    if pack.get("status") != "EXACT_ORIGINAL_BF16_GDN_QKV_CONV_PACK_PASS_NOT_RUNTIME":
        raise ValueError("original packed GDN receipt not successful")
    if pack.get("layer") != 0:
        raise ValueError("first bounded oracle only supports layer 0")
    meta = gdn.sources(checkpoint, 0)
    if meta["sources"] != pack["sources"] or meta["index_sha256"] != pack["index_sha256"]:
        raise ValueError("checkpoint/pack provenance mismatch")
    conv_entry = pack["outputs"]["conv1d"]
    conv_source = meta["sources"]["conv1d"]
    source_path = Path(conv_source["path"])
    with source_path.open("rb") as stream:
        stream.seek(conv_source["offset"])
        raw = stream.read(conv_source["bytes"])
    if hashlib.sha256(raw).hexdigest() != conv_entry["source_payload_sha256"]:
        raise ValueError("original conv source payload hash differs from receipt")
    full_weights = bf16_values(raw)
    global_rows = conv_global(full_weights, rows)
    rank_outputs = {}
    rank_rows = {}
    max_conv_abs = 0.0
    for role in ROLES:
        entry = conv_entry["ranks"][role]
        path = Path(entry["path"])
        packed_raw = path.read_bytes()
        if hashlib.sha256(packed_raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"{role}: packed conv weight SHA-256 mismatch")
        local_rows = conv_packed_rank(bf16_values(packed_raw), role, rows)
        rank_rows[role] = local_rows
        channels = [idx for begin, n in gdn.layout()[role] for idx in range(begin, begin + n)]
        for t in range(rows):
            for j, global_c in enumerate(channels):
                max_conv_abs = max(max_conv_abs, abs(local_rows[t][j] - global_rows[t][global_c]))
    if max_conv_abs:
        raise ValueError("original versus rank-local causal convolution BF16 mismatch")
    parameters, parameter_hashes = original_parameters(checkpoint, 0)
    h_cursor, k_cursor = 0, 0
    max_o, max_s = 0.0, 0.0
    for role, nkeys, nvalues in zip(ROLES, ownership.GDN_K_HEADS,
                                    ownership.GDN_V_HEADS, strict=True):
        local = rank_rows[role]
        rank_result = hashlib.sha256()
        rank_binary = bytearray()
        for local_h in range(nvalues):
            global_h = h_cursor + local_h
            hk_global = global_h // GROUP
            hk_local = local_h // GROUP
            glob_q, glob_k, glob_v = head_slice(global_rows, hk_global, global_h)
            loc_q, loc_k, loc_v = head_local_slice(local, nkeys, nvalues, hk_local, local_h)
            original_state = [[0.0] * 128 for _ in range(128)]
            local_state = [[0.0] * 128 for _ in range(128)]
            for t in range(rows):
                a = scalar_projection(t, global_h, "a")
                b = scalar_projection(t, global_h, "b")
                z = [scalar_projection(t, global_h * 128 + e, "z") for e in range(128)]
                alog = parameters["A_log"][global_h]
                dt = parameters["dt_bias"][global_h]
                first = step_global(original_state, glob_q[t], glob_k[t], glob_v[t],
                                    a, b, z, alog, dt, parameters["norm"])
                second = step_local(local_state, loc_q[t], loc_k[t], loc_v[t],
                                    a, b, z, alog, dt, parameters["norm"])
                max_o = max(max_o, max(abs(a - b) for a, b in zip(first, second, strict=True)))
                encoded = struct.pack("<128f", *second)
                rank_result.update(encoded)
                rank_binary.extend(encoded)
            for e in range(128):
                for d in range(128):
                    max_s = max(max_s, abs(original_state[e][d] - local_state[d][e]))
        rank_outputs[role] = {
            "heads": nvalues, "state_elements": nvalues * 128 * 128,
            "output_sha256": rank_result.hexdigest(),
            "layout": "[value_head, token, 128] FP32 little-endian",
        }
        if output_root:
            output_file = output_root / f"oracle-{role}.f32"
            with output_file.open("xb") as target:
                target.write(rank_binary)
            rank_outputs[role]["oracle_file"] = str(output_file)
        h_cursor += nvalues
        k_cursor += nkeys
    if h_cursor != HEADS or k_cursor != KEY_HEADS or max_o > 5e-5 or max_s > 5e-5:
        raise ValueError(f"global/sharded Qwen recurrent oracle diverged: {max_o} {max_s}")
    coefficients = None
    if output_root:
        packed = struct.pack("<224f", *parameters["A_log"], *parameters["dt_bias"],
                             *parameters["norm"])
        coefficients_path = output_root / "original-gdn-parameters.f32"
        with coefficients_path.open("xb") as dst:
            dst.write(packed)
        coefficients = {
            "file": str(coefficients_path),
            "sha256": hashlib.sha256(packed).hexdigest(),
            "format": "[48 A_log,48 dt_bias,128 norm_gain] FP32 little-endian",
        }
    return {
        "schema": "vtp14-gdn-partial-real-bf16-math-v1",
        "reference_coefficients": coefficients,
        "status": "CARD_FREE_QWEN_GDN_MATH_AND_SHARD_CONTINUITY_PASS_NOT_GPU",
        "layer": 0, "rows": rows,
        "source": "original BF16 conv1d, A_log, dt_bias, norm; synthetic projected QKV/a/b/z",
        "checkpoint_source_sha256": conv_entry["source_payload_sha256"],
        "parameter_hashes": parameter_hashes,
        "global_vs_shard_bf16_conv_max_abs": max_conv_abs,
        "global_vs_shard_output_max_abs": max_o,
        "global_vs_shard_state_max_abs": max_s,
        "rank_results": rank_outputs,
        "not_run": ["GPU", "actual QKV/a/b/z GEMMs", "prefill KKT", "speculative rollback",
                    "out projection", "full model generation"],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--pack-receipt", type=Path, required=True)
    p.add_argument("--rows", type=int, default=2)
    p.add_argument("--phase", choices=("plan", "verify"), default="plan")
    p.add_argument("--run", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    if not 1 <= args.rows <= 8:
        p.error("rows must be 1..8")
    if args.phase == "plan":
        print(json.dumps({
            "status": "CARD_FREE_PLAN",
            "rows": args.rows,
            "checkpoint": str(args.checkpoint),
            "pack_receipt": str(args.pack_receipt),
            "gpu_required": False,
            "limitations": __doc__,
        }, indent=2))
        return 0
    if not args.run:
        p.error("original BF16 payload processing requires explicit --run")
    output = (args.output.expanduser().resolve() if args.output
              else SCRATCH / datetime.now(UTC).strftime("layer0-%Y%m%dT%H%M%S%fZ"))
    if output.exists() or ROOT == output or ROOT in output.parents:
        p.error("must use new outside-Git output directory")
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "INCOMPLETE", "source_sha256": hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()}
    record = output / "summary.json"
    record.write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        data = verify(args.checkpoint, args.pack_receipt, args.rows, output_root=output)
        receipt.update(data)
    except Exception as e:
        receipt.update({"status": "FAILED_INCOMPLETE", "error": str(e)})
        record.write_text(json.dumps(receipt, indent=2) + "\n")
        raise
    record.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"receipt": str(record), "status": receipt["status"],
                      "max_output_abs": receipt["global_vs_shard_output_max_abs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
