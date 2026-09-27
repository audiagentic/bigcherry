"""PNRO20: apply the test-backend-ops patch stack in composition order to a
scratch copy of the vendored file, to find which edit breaks 1253's anchor.

Usage: python tools/lab/pnro20/repro_compose.py [patch_id ...]
"""

import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo / "tools"))
from bigcherry.patch import apply as A  # noqa: E402

order = sys.argv[1:] or [
    "1222_hi67_deterministic_test_backend_ops_seed",
    "1223_hi67_machine_readable_correctness_metrics",
    "1253_nro04_gfx1100_bf16_chunked_gdn",
    "1258_rd12_paired_mul_mat_test_case",
]
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    (root / "tests").mkdir()
    shutil.copy(repo / "vendor/llama.cpp/tests/test-backend-ops.cpp", root / "tests/test-backend-ops.cpp")
    for pid in order:
        spec = importlib.util.spec_from_file_location(pid, repo / "patches" / pid / "patch.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        for p in m.PATCHES:
            if p.path != "tests/test-backend-ops.cpp":
                continue
            try:
                A.apply_patch(p, root)
                print(pid, "OK")
            except Exception as e:  # noqa: BLE001
                print(pid, "FAIL", str(e)[:600])
