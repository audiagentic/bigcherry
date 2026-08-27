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
