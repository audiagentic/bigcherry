"""Validate device code images before a GPU-assisted IQ6R conversion.

A serving kernel plugin may contain gfx1100 code even when the separately
compiled libquant GPU quantizer has only gfx1201 code. A layout plan does not
test this. Fail before GPU launch if the quantizer lacks the selected device.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path


def parse_offload_arches(output: str) -> list[str]:
    return sorted(set(re.findall(r"hipv4-amdgcn-amd-amdhsa--(gfx[0-9]+)", output)))


def quantizer_offload_arches(binary: Path, required: str) -> list[str]:
    result = subprocess.run(["llvm-objdump", "--offloading", str(binary)],
                            capture_output=True, text=True, check=True)
    arches = parse_offload_arches(result.stdout)
    if not arches or required not in arches:
        raise RuntimeError(
            f"quantizer {binary.name} offload arches={arches}; "
            f"cannot run on required {required}"
        )
    return arches
