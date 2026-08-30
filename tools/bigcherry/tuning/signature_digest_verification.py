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
) -> tuple[str, dict[str, Any]]:
    """Run ``test-backend-ops`` in record mode against one real test-file
    or moe-glu-file line and return (a) the single distinct signature hex
    its own dispatch-recording code observed, and (b) the canonical
    content it recorded alongside that hex.

    Exactly HI80's proven rule (generalized, not reimplemented twice):
    repeated observation rows for the SAME signature are fine (a graph can
    legitimately dispatch the same op more than once); zero or more than
    one DISTINCT observed signature means this run cannot unambiguously
    identify which observation corresponds to the requested case, and
    fails closed rather than guessing.

    The observed canonical (adversarial-review follow-up, 2026-08-27):
    ``hip-autotune-record.cpp`` writes the real C++-computed
    ``canonical_json`` alongside every observation's signature hex --
    returning it lets a caller require it to equal the canonical it
    SUPPLIED before trusting the digest, which this function alone cannot
    do (it only knows what the mapper chose to reconstruct, not what the
    caller originally claimed).
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
        observed: dict[str, dict[str, Any]] = {}
        for line in record_db.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("kind") != "observation":
                continue
            signature = row.get("signature")
            canonical = row.get("canonical")
            if isinstance(signature, str) and isinstance(canonical, dict):
                observed[signature.lower()] = canonical
        if len(observed) != 1:
            raise ce.EvidenceError(
                f"signature-verification record-mode run observed "
                f"{len(observed)} distinct signature(s) (expected exactly "
                f"1) -- cannot unambiguously identify the real hex for "
                f"this case: {sorted(observed)!r}"
            )
        value, observed_canonical = next(iter(observed.items()))
        if len(value) != 32 or any(c not in "0123456789abcdef" for c in value):
            raise ce.EvidenceError(
                f"signature-verification record-mode run observed a "
                f"malformed signature hex: {value!r}"
            )
        return value, observed_canonical


def _normalized(canonical: dict[str, Any]) -> str:
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _require_canonical_match(supplied: dict[str, Any], observed: dict[str, Any]) -> None:
    """Adversarial-review follow-up (2026-08-27): matching digests alone
    does not prove ``supplied`` is what the real hash was computed from --
    the test-file-line mappers only use a SUBSET of a canonical's fields
    (e.g. signature_to_test_file_line ignores nb0/nb1/nbd/prec/flags
    entirely, always reconstructing the natural contiguous strides for the
    given shape/type). A canonical with those fields tampered but the same
    ne/type fields would make the mapper reconstruct an IDENTICAL real
    dispatch and observe the SAME digest, letting a poisoned canonical
    pass digest verification even though its true hash (which the real
    signature hasher DOES cover those fields for) would differ -- exactly
    the poisoned-first-canonical attack HI125 exists to close. Comparing
    the REAL observed canonical (recorded by the same real C++ code that
    computed the digest) against the supplied one closes it: for honest
    data the mapper's reconstruction is a deterministic function of
    ne/type, so the natural contiguous layout it produces always equals
    a legitimately-recorded production canonical's own fields; only a
    tampered field creates a mismatch."""
    if _normalized(supplied) != _normalized(observed):
        raise ce.EvidenceError(
            "signature-verification observed canonical content does not "
            "match the supplied canonical -- the real runtime's own "
            "recorded content disagrees with what was claimed, even though "
            "the resulting digest matched; refusing to trust this pairing "
            f"(supplied={supplied!r}, observed={observed!r})"
        )


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

    Digest equality ALONE does not prove ``canonical`` is what the real
    hash was computed from -- see ``_require_canonical_match``'s docstring
    for the concrete poisoned-canonical attack this additionally closes by
    comparing the real observed canonical (recorded by the same C++ code
    that computed the digest) against the one supplied here.
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
            observed_hex, observed_canonical = observed_test_backend_ops_signature_hex(
                binary, moe_glu_file=case_path, seed=seed, runner=runner,
            )
        else:
            line, _target, _digest = scm.signature_to_any_test_file_line(
                canonical, vendor_root=vendor_root,
            )
            case_path.write_text(line + "\n", encoding="utf-8")
            observed_hex, observed_canonical = observed_test_backend_ops_signature_hex(
                binary, test_file=case_path, seed=seed, runner=runner,
            )

    _require_canonical_match(canonical, observed_canonical)
    return observed_hex


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
