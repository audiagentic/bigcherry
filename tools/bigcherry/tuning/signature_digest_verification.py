"""HI121 close-out step 4 (RV84 P0-4): C++-authoritative canonical signature
digest verification.

``inventory.load_measurements()``'s ``signature_digest_verifier`` seam exists
because this module deliberately does NOT reimplement HIP's canonical
serializer or its digest hash in Python -- the only trustworthy proof that a
given canonical signature dict really does hash to a given digest is to run
the REAL compiled ``test-backend-ops`` binary in record mode and read back
what its own C++ signature-hashing code (``hip-autotune-signature.cpp`` +
the BLAKE2 implementation) actually computed. Two hex strings from the same
real code are compared directly; nothing here re-derives either one.

This generalizes ``hi80_generate_correctness_evidence.py``'s own
``_observed_signature_hex()`` (HI119, previously hardcoded to the MoE-GLU
fused-dispatch case) to every canonical-signature shape HI121's audited
domain currently covers: ``hip_required_capabilities()`` is consulted FIRST
so an op/fusion combination outside that audited domain never reaches a real
GPU launch at all (``UnsupportedSignatureDomain``), and only a shape the
existing test-file-line mappers can actually reproduce is attempted
(``SignatureMappingError`` otherwise -- e.g. GATE_BIAS GLU is HI121-audited
but the harness has no biased-GLU mapper yet, and must fail closed rather
than silently substitute the simple-GATE case).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import correctness_evidence as ce
from . import signature_capabilities as sc
from . import signature_mapping as scm


def observed_test_backend_ops_signature_hex(
    binary: Path, *,
    test_file: Path | None = None,
    moe_glu_file: Path | None = None,
    seed: int = 1,
    runner=subprocess.run,
) -> str:
    """Run ``test-backend-ops`` in record mode against one real test-file
    or moe-glu-file line and return the single distinct signature hex its
    own dispatch-recording code observed.

    Exactly HI80's proven rule (generalized, not reimplemented twice):
    repeated observation rows for the SAME signature are fine (a graph can
    legitimately dispatch the same op more than once); zero or more than
    one DISTINCT observed signature means this run cannot unambiguously
    identify which observation corresponds to the requested case, and
    fails closed rather than guessing.
    """
    with tempfile.TemporaryDirectory() as tmp:
        record_db = Path(tmp) / "observed.jsonl"
        result = ce.run_test_backend_ops(
            binary, test_file=test_file, moe_glu_file=moe_glu_file,
            seed=seed, dispatch_mode="record", forced_candidate=None,
            env={"GGML_HIP_DISPATCH_DB": str(record_db)}, runner=runner,
        )
        if result.returncode != 0 or not record_db.is_file():
            raise ce.EvidenceError(
                f"signature-verification record-mode run failed (exit "
                f"{result.returncode}) or produced no dispatch_db -- cannot "
                f"independently observe the real signature hex:\n"
                f"{result.stdout}\n{result.stderr}"
            )
        observed: set[str] = set()
        for line in record_db.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("kind") != "observation":
                continue
            signature = row.get("signature")
            if isinstance(signature, str):
                observed.add(signature.lower())
        if len(observed) != 1:
            raise ce.EvidenceError(
                f"signature-verification record-mode run observed "
                f"{len(observed)} distinct signature(s) (expected exactly "
                f"1) -- cannot unambiguously identify the real hex for "
                f"this case: {sorted(observed)!r}"
            )
        value = next(iter(observed))
        if len(value) != 32 or any(c not in "0123456789abcdef" for c in value):
            raise ce.EvidenceError(
                f"signature-verification record-mode run observed a "
                f"malformed signature hex: {value!r}"
            )
        return value


def observed_signature_digest_hex(
    canonical: dict[str, Any], *,
    binary: Path, vendor_root: Path, seed: int = 1, runner=subprocess.run,
) -> str:
    """Route ``canonical`` to the real test-backend-ops case that reproduces
    it, run it, and return the digest hex the real C++ signature-hashing
    code computed for that real dispatch.

    ``hip_required_capabilities()`` is the audited-domain gate: it raises
    ``UnsupportedSignatureDomain`` for anything outside HI121's currently
    audited op/fusion combinations before this function ever launches a
    GPU process. A signature inside that audited domain that the existing
    test-file-line mappers still cannot represent (e.g. biased/scaled GLU)
    raises ``SignatureMappingError`` from the mapper itself -- this
    function does not invent a substitute case for it.
    """
    sc.hip_required_capabilities(canonical, vendor_root=vendor_root)

    op_names = scm.load_ggml_op_names(vendor_root)
    op_name = op_names.get(int(canonical["op"]))

    with tempfile.TemporaryDirectory() as tmp:
        case_path = Path(tmp) / "signature-case.txt"
        if op_name == "GLU":
            line, _target, _digest = scm.signature_to_moe_glu_file_line(
                canonical, vendor_root=vendor_root,
            )
            case_path.write_text(line + "\n", encoding="utf-8")
            return observed_test_backend_ops_signature_hex(
                binary, moe_glu_file=case_path, seed=seed, runner=runner,
            )

        line, _target, _digest = scm.signature_to_any_test_file_line(
            canonical, vendor_root=vendor_root,
        )
        case_path.write_text(line + "\n", encoding="utf-8")
        return observed_test_backend_ops_signature_hex(
            binary, test_file=case_path, seed=seed, runner=runner,
        )


def make_signature_digest_verifier(
    *, binary: Path, vendor_root: Path, seed: int = 1, runner=subprocess.run,
) -> Callable[[dict[str, Any]], str]:
    """Build a ``signature_digest_verifier`` for
    ``inventory.load_measurements()``, memoized per unique canonical
    signature.

    ``load_measurements()`` calls the verifier once per result row before
    its own signature cache short-circuits, so a single ingest can name the
    same canonical signature many times over -- without memoization here,
    each occurrence would launch a real GPU process for a result already
    proven. The cache key is the exact canonical JSON (sorted, separators
    normalized) so two structurally-identical canonicals always share one
    real verification regardless of dict key order.
    """
    cache: dict[str, str] = {}

    def verify(canonical: dict[str, Any]) -> str:
        key = json.dumps(
            canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        )
        if key not in cache:
            cache[key] = observed_signature_digest_hex(
                canonical, binary=binary, vendor_root=vendor_root,
                seed=seed, runner=runner,
            )
        return cache[key]

    return verify
