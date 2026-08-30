"""HI121 close-out step 6 (HI127): per-winner verification-state tracking.

Schema 8's ``winner_verification`` table distinguishes a winner row
ingested through the HI125-strengthened path (build capability/descriptor
proof passed AND this row's canonical/digest passed real C++ verification)
from one that wasn't -- absence means unknown/not-strengthened, never an
implicit pass, mirroring ``build_capability``'s own established policy.

This module is the one place both the writer (inventory.py's
``load_measurements()``) and the readers (replay_projection.py, and
eventually the promotion gate) agree on what "verified" means, so neither
side can drift into its own interpretation of the same persisted fact.
"""

from __future__ import annotations

import sqlite3

#: The only value ``load_measurements()`` is ever allowed to write --
#: schema 8's own CHECK constraint enforces this at the database level too,
#: but asserting it here means a caller gets a clear Python-level error
#: instead of a raw sqlite3.IntegrityError.
CURRENT_PROFILE = "hi121-strengthened-ingest-v1"


class WinnerRowIdentityError(ValueError):
    """The supplied result row is not exactly one authoritative winner."""


def require_winner_row(
    connection: sqlite3.Connection, *, row: dict, signature_hex: str,
    source_build_id: int | None = None,
) -> int:
    """Resolve a result row to exactly one authoritative winner row.

    ``source_build_id`` scopes projection to its source build.  ``None`` is
    used by same-generation replay and therefore requires one unambiguous
    match across the whole database.  Identity mismatches and ambiguity are
    always hard failures; verification state is checked separately.
    """
    dispatch_hex = row.get("dispatch")
    hardware_hex = row.get("hardware")
    winner_name = row.get("winner")
    native_name = row.get("native")
    if not all(isinstance(value, str) for value in (
        dispatch_hex, hardware_hex, winner_name, native_name, signature_hex,
    )):
        raise WinnerRowIdentityError(f"result row is missing a valid winner identity: {row!r}")
    try:
        dispatch_bytes = bytes.fromhex(dispatch_hex)
        hardware_bytes = bytes.fromhex(hardware_hex)
        signature_bytes = bytes.fromhex(signature_hex)
    except ValueError as exc:
        raise WinnerRowIdentityError(
            f"result row contains a malformed identity digest: {row!r}"
        ) from exc
    if any(len(value) != 16 for value in (dispatch_bytes, hardware_bytes, signature_bytes)):
        raise WinnerRowIdentityError(f"result row identity digests must be 16 bytes: {row!r}")
    if source_build_id is not None and (
        isinstance(source_build_id, bool)
        or not isinstance(source_build_id, int)
        or source_build_id < 1
    ):
        raise WinnerRowIdentityError(f"source_build_id must be a positive integer: {source_build_id!r}")

    query = (
        "SELECT DISTINCT w.winner_id "
        "FROM measurement m "
        "JOIN winner w ON w.build_id = m.build_id "
        " AND w.hardware_id = m.hardware_id "
        " AND w.signature_id = m.signature_id "
        " AND w.dispatch_digest = m.dispatch_digest "
        "JOIN hardware h ON h.hardware_id = w.hardware_id AND h.hardware_digest = ? "
        "JOIN signature s ON s.signature_id = w.signature_id AND s.signature_digest = ? "
        "JOIN candidate c ON c.candidate_id = w.candidate_id "
        " AND c.build_id = w.build_id AND c.stable_name = w.stable_name "
        "WHERE w.dispatch_digest = ? AND w.stable_name = ? AND w.native_stable_name = ?"
    )
    params: list[object] = [
        hardware_bytes, signature_bytes, dispatch_bytes, winner_name, native_name,
    ]
    if source_build_id is not None:
        query += " AND w.build_id = ?"
        params.append(source_build_id)
    matches = connection.execute(query, params).fetchall()
    if len(matches) != 1:
        scope = f"build_id={source_build_id}" if source_build_id is not None else "any build"
        if not matches:
            detail = "does not resolve to an authoritative winner"
        else:
            detail = f"resolves ambiguously to {len(matches)} winner rows"
        raise WinnerRowIdentityError(
            f"result row identity (dispatch={dispatch_hex!r}, signature={signature_hex!r}, "
            f"hardware={hardware_hex!r}, winner={winner_name!r}, native={native_name!r}) "
            f"{detail} in {scope}"
        )
    return int(matches[0][0])


def record_winner_verification(connection: sqlite3.Connection, *, winner_id: int) -> None:
    """Attest that ``winner_id`` was ingested through the current
    strengthened path. Idempotent (INSERT OR REPLACE) so re-attesting the
    same still-current winner_id within one load is harmless; a genuinely
    different winner_id (after a replace) gets its own fresh row."""
    connection.execute(
        "INSERT OR REPLACE INTO winner_verification (winner_id, verification_profile) "
        "VALUES (?, ?)",
        (winner_id, CURRENT_PROFILE),
    )


def is_winner_verified(connection: sqlite3.Connection, *, winner_id: int) -> bool:
    """True if ``winner_id`` carries a CURRENT-profile attestation.

    Deliberately does not accept an older profile string as equivalent --
    if a future profile ever supersedes this one, a caller wanting to
    treat old attestations as still valid must say so explicitly rather
    than have this function silently widen what "verified" means.
    """
    row = connection.execute(
        "SELECT 1 FROM winner_verification WHERE winner_id = ? AND verification_profile = ?",
        (winner_id, CURRENT_PROFILE),
    ).fetchone()
    return row is not None
