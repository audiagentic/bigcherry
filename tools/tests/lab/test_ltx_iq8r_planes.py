"""Read-only, card-free structural IQ8R plane check for the opt-in local LTX model.

This deliberately checks only the small RAD2 header, encoding, directory and
plane tables, not weight payload or GPU numerical equivalence. Python is test
tooling; this file is NOT part of any installed Radiance runtime.
"""
import math
import os
from pathlib import Path
import struct

HEADER = struct.Struct("<IIQ" + "Q" * 5 + "Q" * 16 + "Q" * 8)
ENTRY = struct.Struct("<QQQII6qiiiIQQQ2Q")
PLANE = struct.Struct("<QQ")
ENC_HEADER = struct.Struct("<24s24sii")
ENC_PLANE = struct.Struct("<24sII4q")
ENC_BYTES = ENC_HEADER.size + 8 * ENC_PLANE.size
MAX_TABLE_BYTES = 16 * 1024 * 1024

assert ENTRY.size == 136
assert ENC_BYTES == 568


def _cstr(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("ascii")


def inspect_iq8r_planes(path: Path) -> dict[str, object]:
    """Check *all* declared IQ8R plane sizes/alignment, not their payload values."""
    size = path.stat().st_size
    with path.open("rb") as f:
        h = HEADER.unpack(f.read(HEADER.size))
        if (h[0], h[1], h[2]) != (0x31444152, 2, size):
            raise ValueError("not an intact RAD2 header")

        def table(off: int, count: int, record: struct.Struct) -> list[tuple]:
            total = count * record.size
            if off < HEADER.size or total > MAX_TABLE_BYTES or off > size or total > size-off:
                raise ValueError("out-of-range RAD table")
            f.seek(off)
            return list(record.iter_unpack(f.read(total)))

        str_off, str_bytes = h[8:10]
        if str_off < HEADER.size or str_bytes > MAX_TABLE_BYTES or str_off > size or str_bytes > size-str_off:
            raise ValueError("invalid RAD string section")
        f.seek(str_off)
        strings = f.read(str_bytes)

        def name(off: int) -> str:
            if off >= len(strings):
                raise ValueError("out-of-range RAD name")
            end = strings.find(b"\0", off)
            if end < 0:
                raise ValueError("unterminated RAD name")
            return strings[off:end].decode("utf-8")

        if name(h[3]) != "ltx2":
            raise ValueError("not an LTX2 model")

        dirs = table(h[12], h[13], ENTRY)
        planes = table(h[14], h[15], PLANE)
        if h[17] == 0 or h[17] * ENC_BYTES > MAX_TABLE_BYTES:
            raise ValueError("missing encoding table")
        if h[16] > size or h[17] * ENC_BYTES > size - h[16]:
            raise ValueError("encoding table outside file")
        f.seek(h[16])
        encodings = [f.read(ENC_BYTES) for _ in range(h[17])]
        iq_index = None

        for idx, blob in enumerate(encodings):
            scheme, transform, count, reserved = ENC_HEADER.unpack_from(blob)
            if (_cstr(scheme), _cstr(transform)) != ("affine", "fwht128"):
                continue
            expected = (
                ("codes", 8, 0, 1, 1),
                ("scale", 9, 0, 1, 256),
                ("scale.1", 3, 0, 1, 0),
            )
            if count != 3 or iq_index is not None:
                raise ValueError("invalid/ambiguous IQ8R encoding")
            for j, want in enumerate(expected):
                role, dtype, kind, br, bc, er, ec = ENC_PLANE.unpack_from(
                    blob, ENC_HEADER.size + j * ENC_PLANE.size)
                if (_cstr(role), dtype, kind, br, bc) != want:
                    raise ValueError("unexpected IQ8R encoding plane")
            iq_index = idx
        if iq_index is None:
            raise ValueError("IQ8R encoding absent")

        found = {}
        count = 0
        for e in dirs:
            name_off, _quant, _opts, enc, rank = e[:5]
            if enc != iq_index:
                continue
            shape = e[5:11][:rank]
            group, expert, slot, nplanes = e[11:15]
            first, off, span = e[15:18]
            if rank < 2 or nplanes != 3 or any(d <= 0 for d in shape):
                raise ValueError("unexpected IQ8R weight shape")
            if first > len(planes) or nplanes > len(planes) - first:
                raise ValueError("out-of-range IQ8R planes")
            if off > size or span > size - off:
                raise ValueError("IQ8R weight entry outside file")
            rows, k = math.prod(shape[:-1]), shape[-1]
            expected = (rows * k, rows * ((k + 255)//256), rows * 2)
            for i, want in enumerate(expected):
                po, pb = planes[first + i]
                if pb != want or po % 256 or po < off or po-off > span or pb > span-(po-off):
                    raise ValueError("IQ8R code/scale plane extent mismatch")
            count += 1
            filename = name(name_off)
            if filename in (
                "dit.blk.0.attn1.qkv.weight",
                "dit.blk.0.audio_attn1.qkv.weight",
                "conn.video.blk.0.qkv.weight",
                "te.blk.0.attn_q.weight",
            ):
                found[filename] = {"shape": list(shape), "plane_bytes": list(expected)}
        if count == 0:
            raise ValueError("no IQ8R weights")
        return {"count": count, "samples": found}


def _fixture(path: Path) -> tuple[int, int]:
    strings = b"\0ltx2\0dit.blk.0.attn1.qkv.weight\0"
    name_at = strings.index(b"dit.")
    data_off = 2 * 1024 * 1024
    size = data_off + 4096
    header = [0x31444152, 2, size,
              1, 1, 0, 0, 0,
              HEADER.size, len(strings),
              256, 0, 512, 1,
              768, 3, 1024, 1,
              0, 0, 0, 0,
              data_off, 4096] + [0] * 8
    row = ENTRY.pack(name_at, 0, 0, 0, 2,
                     2, 256, 0, 0, 0, 0,
                     0, -1, 0, 3, 0, data_off, 1024, 0, 0)
    encoding = ENC_HEADER.pack(b"affine", b"fwht128", 3, 0)
    encoding += ENC_PLANE.pack(b"codes", 8, 0, 1, 1, 0, 0)
    encoding += ENC_PLANE.pack(b"scale", 9, 0, 1, 256, 0, 0)
    encoding += ENC_PLANE.pack(b"scale.1", 3, 0, 1, 0, 0, 0)
    encoding += bytes(5 * ENC_PLANE.size)
    with path.open("wb") as f:
        f.write(HEADER.pack(*header))
        f.seek(HEADER.size)
        f.write(strings)
        f.seek(512)
        f.write(row)
        f.seek(768)
        f.write(PLANE.pack(data_off, 512))
        f.write(PLANE.pack(data_off + 512, 2))
        f.write(PLANE.pack(data_off + 768, 4))
        f.seek(1024)
        f.write(encoding)
        f.truncate(size)
    return 768, size


def test_small_iq8r_plane_fixture(tmp_path: Path) -> None:
    import pytest
    p = tmp_path / "small.rad"
    plane_off, size = _fixture(p)
    result = inspect_iq8r_planes(p)
    assert result["count"] == 1
    assert result["samples"]["dit.blk.0.attn1.qkv.weight"] == {
        "shape": [2, 256], "plane_bytes": [512, 2, 4],
    }

    with p.open("r+b") as f:
        f.seek(plane_off + PLANE.size + 8)
        f.write(struct.pack("<Q", 3))  # actual scale plane is 2 B
    with pytest.raises(ValueError, match="plane extent"):
        inspect_iq8r_planes(p)


def test_real_iq8r_planes_when_supplied() -> None:
    path = os.environ.get("BC_LTX_IQ8R_MODEL")
    if not path:
        import pytest
        pytest.skip("no optional BC_LTX_IQ8R_MODEL")
    result = inspect_iq8r_planes(Path(path))
    assert result["count"] > 0
    assert result["samples"]["dit.blk.0.attn1.qkv.weight"]["shape"] == [12288, 4096]
    assert result["samples"]["dit.blk.0.audio_attn1.qkv.weight"]["shape"] == [6144, 2048]


if __name__ == "__main__":
    import json
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 test_ltx_iq8r_planes.py /path/to/model.rad")
    print(json.dumps(inspect_iq8r_planes(Path(sys.argv[1])), sort_keys=True, indent=2))
