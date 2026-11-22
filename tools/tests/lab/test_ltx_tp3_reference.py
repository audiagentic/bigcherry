"""Card-free native LTX TP3 oracles and opt-in read-only RAD header inspection.

Only test/build tooling uses Python. No engine code imports this module.
"""

import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / "tools/lab/radiance/ltx-tp3"

# Frozen candidate abi/rad_format.h: magic, version, file bytes,
# five string offsets, eight section pairs, eight reserved words.
RAD_HEADER = struct.Struct("<IIQ" + "Q" * 5 + "Q" * 16 + "Q" * 8)
RAD_MAGIC = int.from_bytes(b"RAD1", "little")
RAD_FORMAT_VERSION = 2


def inspect_ltx_container_header(path: Path) -> dict[str, int | str]:
    """Bounded structural preflight, not a full container/quantization validation."""
    with path.open("rb") as source:
        size = path.stat().st_size
        header = source.read(RAD_HEADER.size)
        if len(header) != RAD_HEADER.size:
            raise ValueError("truncated RAD header")
        words = RAD_HEADER.unpack(header)
        if words[0] != RAD_MAGIC or words[1] != RAD_FORMAT_VERSION:
            raise ValueError("unexpected RAD magic or format version")
        if words[2] != size:
            raise ValueError("RAD file size mismatch")
        strings_off, strings_bytes = words[8:10]
        data_off, data_bytes = words[22:24]
        if not (RAD_HEADER.size <= strings_off < size
                and strings_bytes > 0
                and strings_bytes <= size - strings_off):
            raise ValueError("invalid RAD string section")
        if data_off > size or data_bytes > size - data_off:
            raise ValueError("invalid RAD data extent")
        # Bounded reads: the string table may be huge, but we only need
        # the declared architecture name and model identifier.
        def string_at(offset: int) -> str:
            if offset >= strings_bytes:
                raise ValueError("out-of-range string reference")
            source.seek(strings_off + offset)
            text = source.read(min(128, strings_bytes - offset))
            if b"\0" not in text:
                raise ValueError("unterminated RAD string")
            return text.split(b"\0", 1)[0].decode("ascii")

        arch, model = string_at(words[3]), string_at(words[4])
        if arch != "ltx2" or model != "ltx2":
            raise ValueError("container is not LTX-2.5")
        # Offsets/counts of sections must be inside the declared file.
        for at in (10, 12, 14, 16, 18, 20):
            if words[at] > size:
                raise ValueError("out-of-range RAD section")
        if words[13] < 1 or words[15] < 1 or words[17] < 1:
            raise ValueError("missing weight directory, planes or encodings")
        return {
            "format": int(words[1]), "bytes": size,
            "architecture": arch, "model": model,
            "directory_entries": int(words[13]),
            "planes": int(words[15]),
            "encodings": int(words[17]),
        }


def run_checked(args: list[str]) -> None:
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, check=False)
    assert result.returncode == 0, result.stdout


def test_native_ltx_tp3_reference() -> None:
    if not shutil.which("cmake") or not (shutil.which("g++") or shutil.which("clang++")):
        pytest.skip("C++20/CMake toolchain unavailable")

    work_root = Path(os.environ.get("BC_WORK_ROOT", str(ROOT / "work")))
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ltx-tp3-test-", dir=work_root) as tmp:
        build = Path(tmp) / "build"
        run_checked(["cmake", "-S", str(REFERENCE), "-B", str(build),
                     "-DBUILD_TESTING=ON"])
        run_checked(["cmake", "--build", str(build), "--parallel", "2"])
        run_checked(["cmake", "--build", str(build), "--target", "test"])


def test_rad_header_fixture(tmp_path: Path) -> None:
    # Small sparse model-shaped file: structural preflight does not allocate
    # an actual container or parse/checkpoint weight data.
    path = tmp_path / "fixture.rad"
    size = (2 * 1024 * 1024) + 256
    strings = b"\0ltx2\0"
    words = [RAD_MAGIC, RAD_FORMAT_VERSION, size,
             1, 1, 0, 0, 0,
             RAD_HEADER.size, len(strings),
             256, 0, 256, 1, 512, 1,
             768, 1, 1024, 0, 1024, 0,
             2 * 1024 * 1024, 256] + [0] * 8
    with path.open("wb") as out:
        out.write(RAD_HEADER.pack(*words))
        out.write(strings)
        out.truncate(size)
    result = inspect_ltx_container_header(path)
    assert result["architecture"] == "ltx2"
    assert result["bytes"] == size
    assert result["planes"] == 1

    with path.open("r+b") as out:
        out.seek(8)
        out.write(struct.pack("<Q", size - 1))
    with pytest.raises(ValueError, match="size mismatch"):
        inspect_ltx_container_header(path)

    with path.open("r+b") as out:
        out.seek(8)
        out.write(struct.pack("<Q", size))
        out.seek(24)
        out.write(struct.pack("<Q", 999999999))
    with pytest.raises(ValueError, match="string reference"):
        inspect_ltx_container_header(path)


def test_local_ltx_iq8r_header_when_supplied() -> None:
    value = os.environ.get("BC_LTX_IQ8R_MODEL")
    if not value:
        pytest.skip("opt in with BC_LTX_IQ8R_MODEL; model is not a CI dependency")
    result = inspect_ltx_container_header(Path(value))
    assert result["directory_entries"] > 0
    assert result["planes"] >= result["directory_entries"]
