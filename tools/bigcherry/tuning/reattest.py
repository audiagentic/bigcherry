"""HI121 close-out step 7 (HI128): positive re-attestation of existing
schema-8 winner rows against their ORIGINAL measurement artifacts.

HI127 already closed the NEGATIVE half of HI128's original scope for free:
migration 0008 deliberately creates zero historical winner_verification
rows, and replay_projection.py already treats an unattested winner as an
ordinary counted omission rather than trusted evidence -- "do not silently
grandfather old schema-7/8 data" is already true today, with no code in
this module. What remains is the POSITIVE half: opportunistically prove
that an EXISTING winner row -- ingested before HI127 existed, or ingested
without a signature_digest_verifier -- deserves the SAME attestation a
strengthened load_measurements() call would have given it, and write that
attestation without disturbing anything else.

This module never calls load_measurements(). Replaying an original
artifact through that function would INSERT OR REPLACE the very
measurement/winner rows this module is trying to prove something about --
mutating the evidence under examination. Instead it reuses
replay_projection.py's own read-only verification primitives
(_load_source_manifest, _require_row_belongs_to_build) that were already
built to prove artifact/DB/manifest agreement without writing anything.

Re-attestation is NOT offline. A row's own recorded ``canonical`` (whether
inline in the artifact or recovered from a --signature-source file) is
still just a CLAIM; HI125 exists precisely because a canonical cannot be
trusted to correspond to its own digest without live confirmation from a
real, compiled test-backend-ops binary via
signature_digest_verification.make_signature_digest_verifier(). A caller
that cannot supply a genuine verifier has nothing this module can attest
with -- see reattest_winners()'s docstring for what dry_run does and does
not prove.

Two-phase design, matching the TOCTOU concern GPT and this session
converged on: Phase A performs every check against a READ-ONLY connection
(and may launch the real, possibly slow GPU verifier subprocess) without
ever writing to the database. Phase B re-opens a short read-write
transaction, re-binds each Phase-A-passed row to its CURRENT winner
identity (in case something else replaced it during Phase A's GPU calls),
and only then writes winner_verification -- so a concurrent write from an
unrelated process can never cause this module to attest a row different
from the one it actually verified.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from . import inventory
from . import replay as replay_module
from . import replay_projection as rp
from . import verification_state


class ReattestationError(RuntimeError):
    """The whole remediation pass cannot be trusted (never a per-row failure)."""


@dataclass(frozen=True)
class RowOutcome:
    dispatch: str
    status: str
    detail: str = ""


@dataclass(frozen=True)
class ReattestationReport:
    examined: int
    attested: int
    already_attested: int
    outcomes: tuple[RowOutcome, ...]


# Every non-"attested"/"already_attested" status leaves winner_verification
# untouched for that row -- a failed or skipped row is simply left at
# whatever state it was already in (usually: absent, i.e. UNKNOWN).
_STATUS_MISSING_SIGNATURE = "missing_signature"
_STATUS_MISSING_CANONICAL = "missing_canonical"
_STATUS_ROW_NO_LONGER_MATCHES_DB = "row_no_longer_matches_db"
_STATUS_CANONICAL_DISAGREES_WITH_DB = "canonical_disagrees_with_db"
_STATUS_SIGNATURE_VERIFICATION_FAILED = "signature_verification_failed"
_STATUS_UNSUPPORTED_SIGNATURE_DOMAIN = "unsupported_signature_domain"
_STATUS_ALREADY_ATTESTED = "already_attested"
_STATUS_CHANGED_DURING_REMEDIATION = "changed_during_remediation"
_STATUS_ATTESTED = "attested"
_STATUS_WOULD_ATTEST = "would_attest"


def _load_signature_shapes(signature_source_paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    """Recover canonical shapes from record/replay diagnostics files, using
    the exact same fallback convention load_measurements() uses for
    artifacts predating inline `canonical` metadata."""
    shapes: dict[str, dict[str, Any]] = {}
    for source_path in signature_source_paths:
        if not source_path.is_file():
            continue
        with source_path.open(encoding="utf-8") as source:
            for line in source:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                signature_hex = row.get("signature", "")
                canonical = row.get("canonical")
                if len(signature_hex) == 32 and isinstance(canonical, dict):
                    shapes.setdefault(signature_hex, canonical)
    return shapes


@dataclass(frozen=True)
class _ProvenRow:
    dispatch: str
    signature_hex: str
    hardware_hex: str
    winner_name: str
    native_name: str
    canonical: dict[str, Any]


def reattest_winners(
    database_path: Path,
    *,
    source_build_id: int,
    measurements_path: Path,
    manifest_path: Path,
    signature_digest_verifier: Callable[[dict[str, Any]], str],
    signature_source_paths: Sequence[Path] = (),
    dry_run: bool = False,
) -> ReattestationReport:
    """Re-verify ``measurements_path``'s winning rows against
    ``source_build_id`` in ``database_path`` and attest the ones that
    genuinely pass every check HI127's strengthened-ingest profile
    requires.

    ``dry_run=True`` runs every check (including the real hardware
    verifier -- there is no cheaper honest check) but never writes
    winner_verification; use it to see what WOULD be attested. It does
    NOT mean "skip the hardware verifier" -- a canonical's correspondence
    to its own digest cannot be proven any other way (see module
    docstring).

    Raises ReattestationError for anything that makes the WHOLE pass
    untrustworthy (unresolvable build/manifest provenance, a projected
    artifact, wrong schema version) -- an individual row's own
    verification failure is instead counted and reported, never raised.
    """
    measurements_path = Path(measurements_path)
    manifest_path = Path(manifest_path)
    database_path = Path(database_path)

    header, results = replay_module.read_results(measurements_path, require_header=True)
    if "hi121_source_provenance" in header:
        raise ReattestationError(
            f"{measurements_path} is a HI121 replay-projection artifact (carries "
            f"hi121_source_provenance), not an original producer measurements file -- "
            f"refusing to re-attest a projection as if it were the original evidence"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    signature_shapes = _load_signature_shapes(signature_source_paths)

    proven: list[_ProvenRow] = []
    outcomes: list[RowOutcome] = []

    ro_conn = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    try:
        try:
            inventory._require_current_schema(ro_conn)
        except inventory.RecordError as exc:
            raise ReattestationError(str(exc)) from exc

        # Read-only: proves manifest hash/descriptor/producer_capabilities
        # and header<->DB<->manifest agreement for source_build_id, without
        # writing anything -- the same primitive M4 uses to trust a source
        # build's capability attestation.
        try:
            rp._load_source_manifest(
                ro_conn, source_manifest_path=manifest_path,
                source_build_id=source_build_id, header=header,
            )
        except rp.ProjectionError as exc:
            raise ReattestationError(str(exc)) from exc

        for row in results:
            dispatch_hex = row.get("dispatch")
            if not isinstance(dispatch_hex, str):
                continue  # not a real result row; nothing to re-attest

            signature_hex = row.get("signature")
            if not isinstance(signature_hex, str):
                outcomes.append(RowOutcome(dispatch_hex, _STATUS_MISSING_SIGNATURE))
                continue

            try:
                winner_id = rp._require_row_belongs_to_build(
                    ro_conn, source_build_id=source_build_id, row=row, signature_hex=signature_hex,
                )
            except rp.ProjectionError as exc:
                outcomes.append(RowOutcome(dispatch_hex, _STATUS_ROW_NO_LONGER_MATCHES_DB, str(exc)))
                continue

            if verification_state.is_winner_verified(ro_conn, winner_id=winner_id):
                outcomes.append(RowOutcome(dispatch_hex, _STATUS_ALREADY_ATTESTED))
                continue

            canonical = row.get("canonical")
            if not isinstance(canonical, dict):
                canonical = signature_shapes.get(signature_hex)
            if not isinstance(canonical, dict):
                outcomes.append(RowOutcome(dispatch_hex, _STATUS_MISSING_CANONICAL))
                continue

            try:
                db_canonical = rp._load_canonical_signature(ro_conn, signature_hex)
            except rp.ProjectionError as exc:
                outcomes.append(RowOutcome(dispatch_hex, _STATUS_ROW_NO_LONGER_MATCHES_DB, str(exc)))
                continue
            if canonical != db_canonical:
                outcomes.append(RowOutcome(dispatch_hex, _STATUS_CANONICAL_DISAGREES_WITH_DB))
                continue

            try:
                observed_hex = signature_digest_verifier(canonical)
            except Exception as exc:  # noqa: BLE001 -- a verifier failure is a per-row outcome
                outcomes.append(RowOutcome(dispatch_hex, _STATUS_UNSUPPORTED_SIGNATURE_DOMAIN, str(exc)))
                continue
            if observed_hex != signature_hex:
                outcomes.append(RowOutcome(dispatch_hex, _STATUS_SIGNATURE_VERIFICATION_FAILED))
                continue

            hardware_hex = row.get("hardware")
            winner_name = row.get("winner")
            native_name = row.get("native")
            proven.append(
                _ProvenRow(
                    dispatch=dispatch_hex, signature_hex=signature_hex,
                    hardware_hex=hardware_hex, winner_name=winner_name, native_name=native_name,
                    canonical=db_canonical,
                )
            )
    finally:
        ro_conn.close()

    attested = 0
    if not dry_run and proven:
        rw_conn = sqlite3.connect(str(database_path))
        try:
            rw_conn.execute("PRAGMA foreign_keys = ON")
            rw_conn.execute("BEGIN IMMEDIATE")
            try:
                inventory._require_current_schema(rw_conn)
                for proven_row in proven:
                    row = {
                        "dispatch": proven_row.dispatch,
                        "hardware": proven_row.hardware_hex,
                        "winner": proven_row.winner_name,
                        "native": proven_row.native_name,
                    }
                    try:
                        current_winner_id = rp._require_row_belongs_to_build(
                            rw_conn, source_build_id=source_build_id, row=row,
                            signature_hex=proven_row.signature_hex,
                        )
                    except rp.ProjectionError:
                        outcomes.append(
                            RowOutcome(proven_row.dispatch, _STATUS_CHANGED_DURING_REMEDIATION)
                        )
                        continue
                    current_canonical = rp._load_canonical_signature(rw_conn, proven_row.signature_hex)
                    if current_canonical != proven_row.canonical:
                        outcomes.append(
                            RowOutcome(proven_row.dispatch, _STATUS_CHANGED_DURING_REMEDIATION)
                        )
                        continue
                    verification_state.record_winner_verification(
                        rw_conn, winner_id=current_winner_id,
                    )
                    attested += 1
                    outcomes.append(RowOutcome(proven_row.dispatch, _STATUS_ATTESTED))
                rw_conn.commit()
            except BaseException:
                rw_conn.rollback()
                raise
        finally:
            rw_conn.close()
    elif dry_run:
        for proven_row in proven:
            outcomes.append(RowOutcome(proven_row.dispatch, _STATUS_WOULD_ATTEST, "dry-run: not written"))

    already_attested = sum(1 for o in outcomes if o.status == _STATUS_ALREADY_ATTESTED)
    return ReattestationReport(
        examined=len(outcomes),
        attested=attested,
        already_attested=already_attested,
        outcomes=tuple(outcomes),
    )
