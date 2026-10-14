#!/usr/bin/env python3
"""CPU-only numerical proxy for Radiance's 64-value rotated wire; NOT a HIP/byte-layout oracle.

Reproduces: f32 Walsh-Hadamard butterflies, bf16-rounded amax scale,
round-to-nearest-even signed fixed-width codes, clamp, reconstructed rank sum,
inverse transform, and BF16 result. Packed bytes use a logical little-endian
bitstream, NOT libr4d's lane/chunk-interleaved physical memory layout.

No accuracy claims for models, GPU race/synchronization, HIP rounding, or throughput.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import random
import struct

GROUP = 64


def f32(x: float) -> float:
    return struct.unpack('<f', struct.pack('<f', float(x)))[0]


def bf16_bits(x: float) -> int:
    """RNE to BF16, matching upstream's bf16_round for finite positive scales."""
    u = struct.unpack('<I', struct.pack('<f', f32(x)))[0]
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF


def bf16(x: float) -> float:
    return struct.unpack('<f', struct.pack('<I', bf16_bits(x) << 16))[0]


def transform64(values: list[float]) -> list[float]:
    if len(values) != GROUP:
        raise ValueError('Hadamard block requires 64 elements')
    v = [f32(x) for x in values]
    # First 3 stages are within a lane's contiguous group of eight values;
    # last 3 stages exchange corresponding registers across the eight lanes.
    for stride in (1, 2, 4, 8, 16, 32):
        for base in range(0, GROUP, 2 * stride):
            for j in range(stride):
                a, b = v[base + j], v[base + j + stride]
                v[base + j] = f32(a + b)
                v[base + j + stride] = f32(a - b)
    return [f32(x * 0.125) for x in v]


def pack_codes(codes: list[int], bits: int) -> bytes:
    if bits not in (4, 6, 8):
        raise ValueError('only 4, 6, 8 bits are supported')
    out = bytearray((len(codes) * bits + 7) // 8)
    mask = (1 << bits) - 1
    for i, code in enumerate(codes):
        if not 0 <= code <= mask:
            raise ValueError('invalid code')
        at = i * bits
        out[at // 8] |= (code << (at % 8)) & 0xFF
        if at % 8 + bits > 8:
            out[at // 8 + 1] |= code >> (8 - at % 8)
    return bytes(out)


def unpack_codes(payload: bytes, bits: int, count: int) -> list[int]:
    if len(payload) != (bits * count + 7) // 8:
        raise ValueError('incorrect packet length')
    out = []
    mask = (1 << bits) - 1
    for i in range(count):
        at = i * bits
        word = payload[at // 8]
        if at % 8 + bits > 8:
            word |= payload[at // 8 + 1] << 8
        out.append((word >> (at % 8)) & mask)
    return out


@dataclass(frozen=True)
class Packet:
    bits: int
    count: int
    packed: bytes
    scale_bits: tuple[int, ...]

    @property
    def wire_bytes(self) -> int:
        return len(self.packed) + 2 * len(self.scale_bits)


def encode(src: list[float], bits: int) -> Packet:
    if bits not in (4, 6, 8) or not src or len(src) % GROUP:
        raise ValueError('invalid bit width or shape')
    codes: list[int] = []
    scales: list[int] = []
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    for start in range(0, len(src), GROUP):
        # Inputs to Radiance AR are half/bf16; mock uses bf16 only.
        rotated = transform64([bf16(x) for x in src[start:start + GROUP]])
        amax = max(abs(v) for v in rotated)
        scale = bf16(f32(amax / qmax)) if amax else 1.0
        if not (scale > 0.0 and math.isfinite(scale)):
            raise ValueError('invalid scale; proxy supports only finite normal input')
        scales.append(bf16_bits(scale))
        inv = f32(1.0 / scale)
        for value in rotated:
            q = round(f32(value * inv))
            codes.append(max(qmin, min(qmax, q)) - qmin)
    return Packet(bits, len(src), pack_codes(codes, bits), tuple(scales))


def reduce_packets(a: Packet, b: Packet) -> list[float]:
    if (a.bits, a.count) != (b.bits, b.count):
        raise ValueError('incompatible packets')
    qmin = -(1 << (a.bits - 1))
    ac = unpack_codes(a.packed, a.bits, a.count)
    bc = unpack_codes(b.packed, b.bits, b.count)
    out: list[float] = []
    for start in range(0, a.count, GROUP):
        block = start // GROUP
        sa = struct.unpack('<f', struct.pack('<I', a.scale_bits[block] << 16))[0]
        sb = struct.unpack('<f', struct.pack('<I', b.scale_bits[block] << 16))[0]
        rotated_sum = [f32(f32((ac[i] + qmin) * sa) + f32((bc[i] + qmin) * sb))
                       for i in range(start, start + GROUP)]
        out.extend(bf16(x) for x in transform64(rotated_sum))
    return out


def exact_sum(a: list[float], b: list[float]) -> list[float]:
    return [bf16(f32(bf16(x) + bf16(y))) for x, y in zip(a, b, strict=True)]


def make_inputs(seed: int, case: str, groups: int = 8) -> tuple[list[float], list[float]]:
    rng = random.Random(seed)

    def values() -> list[float]:
        v = [rng.gauss(0, 1) for _ in range(groups * GROUP)]
        if case == 'heavy_tail':
            for i in range(0, len(v), GROUP):
                v[i + rng.randrange(GROUP)] = rng.choice([-1.0, 1.0]) * 55.0
        elif case == 'sparse':
            v = [x if rng.random() < 0.08 else 0.0 for x in v]
        elif case == 'scale_mixed':
            for i in range(0, len(v), GROUP):
                factor = 0.01 if (i // GROUP) % 2 else 15.0
                v[i:i + GROUP] = [x * factor for x in v[i:i + GROUP]]
        elif case != 'gaussian':
            raise ValueError(f'unknown pattern: {case}')
        return v
    return values(), values()


def metrics(reference: list[float], actual: list[float]) -> dict[str, float]:
    diff = [x - y for x, y in zip(reference, actual, strict=True)]
    return {
        'rel_l2': math.sqrt(sum(x*x for x in diff) / max(sum(x*x for x in reference), 1e-30)),
        'rmse': math.sqrt(sum(x*x for x in diff) / len(diff)),
        'max_abs': max(abs(x) for x in diff),
    }


def run_mock() -> dict:
    out = {'schema': 1, 'kind': 'synthetic-cpu-only', 'seed': 7107,
           'gpu_tested': False, 'model_kl_tested': False, 'groups_per_rank': 8,
           'wire_layout': 'logical-bitstream-not-gpu-interleaved', 'results': {}}
    for case in ('gaussian', 'heavy_tail', 'sparse', 'scale_mixed'):
        a, b = make_inputs(7107, case)
        ref = exact_sum(a, b)
        rows = {}
        for bits in (4, 6, 8):
            pa, pb = encode(a, bits), encode(b, bits)
            result = reduce_packets(pa, pb)
            rows[f'wht{bits}'] = {**metrics(ref, result),
                                   'wire_bytes_per_rank': pa.wire_bytes,
                                   'payload_ratio_to_bf16': pa.wire_bytes / (2 * len(a))}
        out['results'][case] = rows
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true', help='emit structured mock results')
    args = parser.parse_args()
    result = run_mock()
    if args.json:
        print(json.dumps(result, sort_keys=True, indent=2))
    else:
        for case, rows in result['results'].items():
            print(case, ' '.join(f'{mode}: rel_l2={m["rel_l2"]:.6f}, bytes={m["wire_bytes_per_rank"]}'
                                 for mode, m in rows.items()))
        print('SYNTHETIC ONLY: no model KLD, no GPU validation, no performance evidence')


if __name__ == '__main__':
    main()
