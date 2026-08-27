"""HI121 close-out step 7 (HI128): positive re-attestation of existing
schema-8 winner rows against their ORIGINAL measurement artifacts.

HI127 already closed the NEGATIVE half of HI128's original scope for free:
migration 0008 deliberately creates zero historical winner_verification
rows, and replay_projection.py already treats an unattested winner as an
ordinary counted omission rather than trusted evidence -- "do not silently
grandfather old schema-7/8 data" is already true today, with no code in
this module. What remains is the POSITIVE half: opportunistically prove
that an EXISTING winner row -- including one belonging to a build that
predates HI121 entirely, with NO build_capability row and a NULL
build_descriptor_hash on `build` -- deserves the SAME attestation a
strengthened load_measurements() call would have given it, and write that
attestation without disturbing anything else.

This module never calls load_measurements(). Replaying an original
artifact through that function would INSERT OR REPLACE the very
measurement/winner rows this module is trying to prove something about --
mutating the evidence under examination. It reuses
inventory.verify_hip_build_artifacts() (the pure, DB-independent half of
HI121 M2's capability proof -- deliberately NOT replay_projection.py's
_load_source_manifest(), which asserts "this build is already
HI121-projectable" and therefore REQUIRES an existing build_capability
row; that is the wrong question for a historical build this module exists
specifically to recover from a MISSING build_capability row) together
with replay_projection.py's own read-only row-binding primitives
(_require_row_belongs_to_build, _load_canonical_signature) that were
already built to prove artifact/DB agreement without writing anything.

Re-attestation is NOT offline. A row's own recorded ``canonical`` (whether
inline in the artifact or recovered from a --signature-source file) is
still just a CLAIM; HI125 exists precisely because a canonical cannot be
trusted to correspond to its own digest without live confirmation from a
real, compiled test-backend-ops binary via
signature_digest_verification.make_signature_digest_verifier(). A caller
that cannot supply a genuine verifier has nothing this module can attest
with -- see reattest_winners()'s docstring for what dry_run does and does
not prove.

Two-phase design, matching the TOCTOU concern this session converged on
with dev-gpt-agent across two review rounds: Phase A performs every check
against a READ-ONLY connection (and may launch the real, possibly slow GPU
verifier subprocess) without ever writing to the database. Phase B
re-opens a short read-write transaction and re-checks EVERYTHING Phase A's
attestation depended on -- not just per-winner row identity, but the
build-level provenance (source_revision, manifest_hash,
build_descriptor_hash, and any existing build_capability row) that every
proven row in the batch shares -- before writing anything, so a concurrent
change during Phase A's GPU calls can never cause this module to attest a
row (or backfill a build-level claim) inconsistent with what it actually
verified. A build-level disagreement aborts the WHOLE pass (every proven
row shared that provenance, so one bad build-level fact poisons all of
them); a single row's own identity changing is instead that one row's
ordinary, non-fatal `changed_during_remediation` outcome.
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
from . import signature_capabilities as sc
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
    backfilled_build_descriptor: bool
    backfilled_build_capability: bool
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

_HEX_DIGITS = set("0123456789abcdefABCDEF")


def _valid_digest_hex(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 32 and all(c in _HEX_DIGITS for c in value)


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


def _require_build_provenance_matches(
    connection: sqlite3.Connection, *, source_build_id: int, proof: inventory.HipBuildProof,
) -> tuple[bool, bool]:
    """Prove that ``source_build_id``'s CURRENT build/build_capability rows
    are consistent with ``proof`` (never conflicting; NULL/absent means
    not-yet-attested, safely backfillable). Returns
    (build_descriptor_needs_backfill, build_capability_needs_backfill).

    Raises ReattestationError on any actual disagreement -- this is a
    whole-pass abort, not a per-row outcome, because every proven row in
    this pass shares the SAME build-level provenance; one bad build-level
    fact means none of them can be trusted.
    """
    build_row = connection.execute(
        "SELECT source_revision, manifest_hash, build_descriptor_hash FROM build WHERE build_id = ?",
        (source_build_id,),
    ).fetchone()
    if build_row is None:
        raise ReattestationError(f"source_build_id={source_build_id} does not exist in this dispatch_db")
    db_source_revision, db_manifest_hash, db_descriptor_hash = build_row
    if db_source_revision != proof.source_revision or db_manifest_hash != proof.manifest_hash:
        raise ReattestationError(
            f"source_build_id={source_build_id}'s own source_revision/manifest_hash does not "
            f"match the original artifact's proven values -- this measurements/manifest pair "
            f"does not belong to this build"
        )
    needs_descriptor_backfill = False
    if db_descriptor_hash is None:
        needs_descriptor_backfill = True
    elif db_descriptor_hash != proof.build_descriptor_hash:
        raise ReattestationError(
            f"source_build_id={source_build_id} already has a DIFFERENT build_descriptor_hash "
            f"on file than the original artifact proves -- refusing to silently overwrite an "
            f"existing build-level claim"
        )

    cap_row = connection.execute(
        "SELECT producer_capabilities FROM build_capability WHERE build_id = ? AND backend = 'hip'",
        (source_build_id,),
    ).fetchone()
    needs_capability_backfill = False
    if cap_row is None:
        needs_capability_backfill = True
    elif cap_row[0] != proof.producer_capabilities.to_bytes():
        raise ReattestationError(
            f"source_build_id={source_build_id} already has a DIFFERENT hip producer_capabilities "
            f"row on file than the original artifact proves -- capability claims are immutable "
            f"once persisted, never silently overwritten"
        )
    return needs_descriptor_backfill, needs_capability_backfill


@dataclass(frozen=True)
class _ProvenRow:
    dispatch: str
    signature_hex: str
    hardware_hex: str
    winner_name: str
    native_name: str
    canonical: dict[str, Any]


def _verify_rows(
    connection: sqlite3.Connection, *, source_build_id: int, results: list[dict[str, Any]],
    signature_shapes: dict[str, dict[str, Any]],
    signature_digest_verifier: Callable[[dict[str, Any]], str],
) -> tuple[list[_ProvenRow], list[RowOutcome]]:
    proven: list[_ProvenRow] = []
    outcomes: list[RowOutcome] = []
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
                connection, source_build_id=source_build_id, row=row, signature_hex=signature_hex,
            )
        except rp.ProjectionError as exc:
            outcomes.append(RowOutcome(dispatch_hex, _STATUS_ROW_NO_LONGER_MATCHES_DB, str(exc)))
            continue

        if verification_state.is_winner_verified(connection, winner_id=winner_id):
            outcomes.append(RowOutcome(dispatch_hex, _STATUS_ALREADY_ATTESTED))
            continue

        canonical = row.get("canonical")
        if not isinstance(canonical, dict):
            canonical = signature_shapes.get(signature_hex)
        if not isinstance(canonical, dict):
            outcomes.append(RowOutcome(dispatch_hex, _STATUS_MISSING_CANONICAL))
            continue

        try:
            db_canonical = rp._load_canonical_signature(connection, signature_hex)
        except rp.ProjectionError as exc:
            outcomes.append(RowOutcome(dispatch_hex, _STATUS_ROW_NO_LONGER_MATCHES_DB, str(exc)))
            continue
        if canonical != db_canonical:
            outcomes.append(RowOutcome(dispatch_hex, _STATUS_CANONICAL_DISAGREES_WITH_DB))
            continue

        try:
            observed_hex = signature_digest_verifier(canonical)
        except sc.UnsupportedSignatureDomain as exc:
            outcomes.append(RowOutcome(dispatch_hex, _STATUS_UNSUPPORTED_SIGNATURE_DOMAIN, str(exc)))
            continue
        except Exception as exc:  # noqa: BLE001 -- any other verifier failure is a per-row outcome
            outcomes.append(RowOutcome(dispatch_hex, _STATUS_SIGNATURE_VERIFICATION_FAILED, str(exc)))
            continue
        if not _valid_digest_hex(observed_hex):
            outcomes.append(
                RowOutcome(
                    dispatch_hex, _STATUS_SIGNATURE_VERIFICATION_FAILED,
                    f"verifier returned an invalid digest {observed_hex!r}",
                )
            )
            continue
        if observed_hex.lower() != signature_hex.lower():
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
    return proven, outcomes


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
    requires -- including a build that has NO existing build_capability
    row or build_descriptor_hash yet, which Phase B backfills once the
    original artifact proves it (never overwriting a conflicting existing
    value).

    ``dry_run=True`` runs every check (including the real hardware
    verifier -- there is no cheaper honest check) but never writes
    anything; outcomes that would have been attested are reported as
    "would_attest" instead of "attested", and a second, read-only
    final-state check re-verifies build-level provenance and each row's
    binding/canonical exactly as Phase B would, so a dry run's report
    reflects what a real run would ACTUALLY do at commit time, not just
    what Phase A alone proved.

    Raises ReattestationError for anything that makes the WHOLE pass
    untrustworthy (unresolvable build/manifest provenance, a projected
    artifact, wrong schema version, or a build-level provenance conflict)
    -- an individual row's own verification failure is instead counted and
    reported, never raised.
    """
    measurements_path = Path(measurements_path)
    manifest_path = Path(manifest_path)
    database_path = Path(database_path)

    # Every failure mode below makes the WHOLE pass untrustworthy (a
    # malformed/unreadable artifact, a projection mistaken for original
    # evidence, or a header/manifest disagreement) -- normalize all of them
    # to ReattestationError so a caller catching only that type (as
    # cli/tuning.py's cmd_reattest does) cannot be bypassed by, say, a
    # corrupt manifest raising a bare RecordError/SystemExit/OSError
    # instead.
    try:
        header, results = replay_module.read_results(measurements_path, require_header=True)
    except SystemExit as exc:
        raise ReattestationError(f"{measurements_path}: {exc}") from exc
    if "hi121_source_provenance" in header:
        raise ReattestationError(
            f"{measurements_path} is a HI121 replay-projection artifact (carries "
            f"hi121_source_provenance), not an original producer measurements file -- "
            f"refusing to re-attest a projection as if it were the original evidence"
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ReattestationError(f"{manifest_path}: not readable/valid JSON: {exc}") from exc
    try:
        proof = inventory.verify_hip_build_artifacts(header=header, manifest=manifest)
    except inventory.RecordError as exc:
        raise ReattestationError(str(exc)) from exc
    if proof is None:
        raise ReattestationError(
            f"{measurements_path}/{manifest_path} do not establish a HIP build capability "
            f"proof (no producer_capabilities on the header/manifest) -- nothing for this "
            f"module to attest"
        )

    signature_shapes = _load_signature_shapes(signature_source_paths)

    ro_conn = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    try:
        try:
            inventory._require_current_schema(ro_conn)
        except inventory.RecordError as exc:
            raise ReattestationError(str(exc)) from exc

        _require_build_provenance_matches(ro_conn, source_build_id=source_build_id, proof=proof)
        proven, outcomes = _verify_rows(
            ro_conn, source_build_id=source_build_id, results=results,
            signature_shapes=signature_shapes, signature_digest_verifier=signature_digest_verifier,
        )
    finally:
        ro_conn.close()

    if dry_run:
        # A second, independent read-only pass so a dry run's report
        # reflects Phase B's real final-state checks (build-level
        # provenance re-verified, each row re-bound/re-compared) rather
        # than only what Phase A alone proved -- "runs every check"
        # includes this one.
        final_ro_conn = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
        try:
            try:
                inventory._require_current_schema(final_ro_conn)
            except inventory.RecordError as exc:
                raise ReattestationError(str(exc)) from exc
            _require_build_provenance_matches(final_ro_conn, source_build_id=source_build_id, proof=proof)
            for proven_row in proven:
                row = {
                    "dispatch": proven_row.dispatch, "hardware": proven_row.hardware_hex,
                    "winner": proven_row.winner_name, "native": proven_row.native_name,
                }
                try:
                    winner_id = rp._require_row_belongs_to_build(
                        final_ro_conn, source_build_id=source_build_id, row=row,
                        signature_hex=proven_row.signature_hex,
                    )
                    current_canonical = rp._load_canonical_signature(final_ro_conn, proven_row.signature_hex)
                except rp.ProjectionError:
                    outcomes.append(RowOutcome(proven_row.dispatch, _STATUS_CHANGED_DURING_REMEDIATION))
                    continue
                if current_canonical != proven_row.canonical:
                    outcomes.append(RowOutcome(proven_row.dispatch, _STATUS_CHANGED_DURING_REMEDIATION))
                    continue
                outcomes.append(RowOutcome(proven_row.dispatch, _STATUS_WOULD_ATTEST, "dry-run: not written"))
        finally:
            final_ro_conn.close()
        already_attested = sum(1 for o in outcomes if o.status == _STATUS_ALREADY_ATTESTED)
        return ReattestationReport(
            examined=len(outcomes), attested=0, already_attested=already_attested,
            backfilled_build_descriptor=False, backfilled_build_capability=False,
            outcomes=tuple(outcomes),
        )

    attested = 0
    backfilled_descriptor = False
    backfilled_capability = False
    if proven:
        rw_conn = sqlite3.connect(str(database_path))
        try:
            rw_conn.execute("PRAGMA foreign_keys = ON")
            rw_conn.execute("BEGIN IMMEDIATE")
            try:
                try:
                    inventory._require_current_schema(rw_conn)
                except inventory.RecordError as exc:
                    raise ReattestationError(str(exc)) from exc
                needs_descriptor, needs_capability = _require_build_provenance_matches(
                    rw_conn, source_build_id=source_build_id, proof=proof,
                )
                if needs_descriptor:
                    try:
                        rw_conn.execute(
                            "UPDATE build SET build_descriptor_hash = ? WHERE build_id = ?",
                            (proof.build_descriptor_hash, source_build_id),
                        )
                    except sqlite3.IntegrityError as exc:
                        # e.g. build_legacy_identity_uq: backfilling this
                        # build's descriptor would collide with a different
                        # existing build that already shares its legacy
                        # identity plus this descriptor -- a real conflict,
                        # not something this module can safely resolve.
                        raise ReattestationError(
                            f"backfilling build_descriptor_hash for source_build_id="
                            f"{source_build_id} would violate a database constraint: {exc}"
                        ) from exc
                    backfilled_descriptor = True
                if needs_capability:
                    rw_conn.execute(
                        "INSERT INTO build_capability (build_id, backend, producer_capabilities) "
                        "VALUES (?, 'hip', ?)",
                        (source_build_id, proof.producer_capabilities.to_bytes()),
                    )
                    backfilled_capability = True

                for proven_row in proven:
                    row = {
                        "dispatch": proven_row.dispatch, "hardware": proven_row.hardware_hex,
                        "winner": proven_row.winner_name, "native": proven_row.native_name,
                    }
                    try:
                        current_winner_id = rp._require_row_belongs_to_build(
                            rw_conn, source_build_id=source_build_id, row=row,
                            signature_hex=proven_row.signature_hex,
                        )
                        current_canonical = rp._load_canonical_signature(rw_conn, proven_row.signature_hex)
                    except rp.ProjectionError:
                        outcomes.append(
                            RowOutcome(proven_row.dispatch, _STATUS_CHANGED_DURING_REMEDIATION)
                        )
                        continue
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

    already_attested = sum(1 for o in outcomes if o.status == _STATUS_ALREADY_ATTESTED)
    return ReattestationReport(
        examined=len(outcomes), attested=attested, already_attested=already_attested,
        backfilled_build_descriptor=backfilled_descriptor,
        backfilled_build_capability=backfilled_capability,
        outcomes=tuple(outcomes),
    )
