"""HI80: correctness-evidence generation CLI.

Loops over a measurements JSONL's promotable-but-correctness-blocked rows
(``provisional_winner`` set and not equal to ``native``), derives the
op_filter/target_tensor for each row's real dispatch signature via
signature_correctness_mapping.signature_to_op_filter(), runs
correctness_evidence.generate_correctness_evidence() against a patched
test-backend-ops binary, and writes the result into dispatch_db via
correctness_evidence.write_correctness_evidence() -- exactly the input
tune_promotion.promote()'s RV49 gate (promotion_correctness_gate.py) checks
for, closing the loop this item's own notes describe as "not done this
session" at HI80's original filing.

Deliberately MUL_MAT-only this slice, same as signature_correctness_mapping
itself (see that module's own docstring on why this is not generic): any
non-MUL_MAT row is reported and skipped, not treated as a hard CLI failure,
since HI80 never claimed op coverage beyond MUL_MAT.

Fails closed per row, not globally: a genuine evidence-generation failure
(EvidenceError -- e.g. a digest mismatch, a nonzero exit) or an unsupported/
malformed signature (SignatureMappingError) is reported for that row and
does not stop the loop from attempting the remaining rows -- but the process
exits nonzero if any row failed, so a caller scripting this cannot silently
treat a partial run as a clean one. A row whose exact (build, hardware,
signature, candidate, contract_version) identity already has evidence is
skipped without re-running anything (idempotent re-invocation).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from . import correctness_evidence as ce
from . import paths
from . import promotion_correctness_gate as gate
from . import signature_correctness_mapping as scm
from . import tune_promotion


class CliError(RuntimeError):
    pass


def _read_measurements(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        return tune_promotion._read(path)
    except tune_promotion.PromotionError as exc:
        raise CliError(str(exc)) from exc


def find_candidate_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows with a real nominated, non-native provisional winner -- the same
    "is this row even a candidate for promotion" test promote() itself uses
    (tune_promotion.promote()'s own ``provisional == row.get("native")``
    skip), independent of promotion_status so this can run BEFORE promote()
    ever sees the file."""
    return [
        row
        for row in results
        if isinstance(row.get("provisional_winner"), str)
        and row["provisional_winner"] != row.get("native")
    ]


def _existing_evidence_id(
    conn: sqlite3.Connection, identity: gate.PromotionIdentity, *, contract_version: str,
) -> int | None:
    row = conn.execute(
        "SELECT correctness_evidence_id FROM correctness_evidence "
        "WHERE build_id = ? AND hardware_id = ? AND signature_id = ? "
        "AND candidate_id = ? AND contract_version = ?",
        (identity.build_id, identity.hardware_id, identity.signature_id,
         identity.candidate_id, contract_version),
    ).fetchone()
    return row[0] if row is not None else None


def _load_signature_dict(conn: sqlite3.Connection, signature_id: int) -> dict[str, Any]:
    row = conn.execute(
        "SELECT canonical_json FROM signature WHERE signature_id = ?", (signature_id,)
    ).fetchone()
    if row is None:
        raise CliError(f"signature_id={signature_id} vanished from dispatch_db mid-run")
    return json.loads(row[0])


def generate_for_row(
    conn: sqlite3.Connection, row: dict[str, Any], *,
    binary: Path, vendor_root: Path, seeds: tuple[int, ...],
    headroom_fraction: float, contract_version: str, tool_version: str,
    runner=subprocess.run,
) -> str:
    """Returns a short human-readable outcome string. Raises CliError,
    scm.SignatureMappingError or ce.EvidenceError on failure -- the caller
    decides whether that is fatal to the whole run."""
    dispatch_hex = row.get("dispatch")
    signature_hex = row.get("signature")
    hardware_hex = row.get("hardware")
    native_name = row.get("native")
    candidate_name = row["provisional_winner"]
    if not all(isinstance(v, str) and v for v in
               (dispatch_hex, signature_hex, hardware_hex, native_name)):
        raise CliError(
            f"dispatch={dispatch_hex!r}: missing dispatch/signature/hardware/native "
            "identity fields, cannot resolve against dispatch_db"
        )

    identity = gate.resolve_promotion_identity(
        conn, dispatch_hex=dispatch_hex, signature_hex=signature_hex,
        hardware_hex=hardware_hex, native_name=native_name, candidate_name=candidate_name,
    )

    existing = _existing_evidence_id(conn, identity, contract_version=contract_version)
    if existing is not None:
        return f"skip (already has evidence_id={existing})"

    signature_dict = _load_signature_dict(conn, identity.signature_id)
    # HI80 (2026-08-23 real-hardware finding): signature_to_op_filter()'s
    # -p regex only matches a real signature by coincidence, when it
    # happens to equal one of test-backend-ops' own fixed synthetic test
    # shapes -- confirmed on Brutus that a real production MUL_MAT
    # (m=32/n=21/k=2880) matches none of them. signature_to_test_file_line()
    # drives test-backend-ops' --test-file escape hatch instead, which
    # builds the exact requested shape directly via test_generic_op, so it
    # works for any signature within its (narrower) type scope regardless
    # of whether the fixed corpus happens to contain it.
    test_file_line, target_tensor, digest_tensor = scm.signature_to_test_file_line(
        signature_dict, vendor_root=vendor_root
    )
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    ) as handle:
        handle.write(test_file_line + "\n")
        test_file_path = Path(handle.name)
    try:
        aggregate = ce.generate_correctness_evidence(
            binary, test_file=test_file_path, target_tensor=target_tensor,
            digest_tensor=digest_tensor,
            candidate_stable_name=candidate_name, seeds=seeds,
            headroom_fraction=headroom_fraction, contract_version=contract_version,
            runner=runner,
        )
    finally:
        test_file_path.unlink(missing_ok=True)
    evidence_id = ce.write_correctness_evidence(
        conn, build_id=identity.build_id, hardware_id=identity.hardware_id,
        signature_id=identity.signature_id, candidate_id=identity.candidate_id,
        native_candidate_id=identity.native_candidate_id, aggregate=aggregate,
        tool_version=tool_version,
    )
    verdict = "dispatchable" if aggregate.dispatchable else "NOT dispatchable"
    return f"wrote evidence_id={evidence_id} ({verdict}, e_c={aggregate.e_c_nmse:.6g})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path, help="measurements JSONL to scan")
    parser.add_argument("--dispatch-db", type=Path, required=True)
    parser.add_argument(
        "--binary", type=Path, required=True,
        help="patched test-backend-ops (patches 1222+1223 applied)",
    )
    parser.add_argument("--llama-root", default=None, help="vendor checkout (default: vendor/llama.cpp)")
    parser.add_argument(
        "--seeds", default="1,2,3",
        help="comma-separated deterministic seed set (versioned per contract_version, RV77 Q2)",
    )
    parser.add_argument("--headroom-fraction", type=float, default=ce.DEFAULT_HEADROOM_FRACTION)
    parser.add_argument("--contract-version", default=ce.CONTRACT_VERSION)
    parser.add_argument("--tool-version", default="hi80-cli-v1")
    args = parser.parse_args(argv)

    seeds = tuple(int(s) for s in args.seeds.split(","))
    vendor_root = paths.llama_root(args.llama_root)

    try:
        _header, results = _read_measurements(args.measurements)
    except CliError as exc:
        print(f"hi80-generate-correctness-evidence: {exc}", file=sys.stderr)
        return 2

    rows = find_candidate_rows(results)
    if not rows:
        print("no promotable-but-non-native rows found; nothing to do", file=sys.stderr)
        return 0

    conn = sqlite3.connect(str(args.dispatch_db))
    failed = 0
    try:
        for row in rows:
            dispatch = row.get("dispatch", "?")
            try:
                outcome = generate_for_row(
                    conn, row, binary=args.binary, vendor_root=vendor_root,
                    seeds=seeds, headroom_fraction=args.headroom_fraction,
                    contract_version=args.contract_version, tool_version=args.tool_version,
                )
                print(f"{dispatch}: {outcome}", file=sys.stderr)
            except scm.SignatureMappingError as exc:
                print(f"{dispatch}: SKIPPED (unsupported signature): {exc}", file=sys.stderr)
            except (gate.CorrectnessGateError, ce.EvidenceError, CliError) as exc:
                failed += 1
                print(f"{dispatch}: FAILED: {exc}", file=sys.stderr)
    finally:
        conn.close()

    if failed:
        print(f"hi80-generate-correctness-evidence: {failed}/{len(rows)} row(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
