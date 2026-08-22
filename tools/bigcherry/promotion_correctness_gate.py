"""HI67 slice 3: correctness-evidence gating for tune_promotion.promote().

RV49's contract (already implemented in correctness_evidence.py for slice 2)
must be a HARD AND with the existing BH/bootstrap statistical promotion
criteria -- correctness never rescues a statistical failure, and statistical
significance never bypasses missing or failed correctness evidence.

Design corrected by GPT adjudication (RV77 follow-up, request
req_5e04cb1d9c254fe5, 2026-08-21) after an initial proposal that would have
regressed real identity invariants this codebase already enforces:

- build_id is NOT resolved via (source_revision, manifest_hash) -- that is
  the LEGACY key; schema 4 deliberately allows distinct campaign builds to
  share it (see sql/dispatch-db.sql's identity_scope CHECK and the two
  partial UNIQUE indexes). This module resolves identity purely from rows
  the dispatch_db ALREADY has -- the measurements artifact was ingested into
  this exact database before promotion ever runs (lifecycle.py's tune
  stage), so no upsert/legacy-key guessing is needed, only SELECTs.
- signature_id is NOT resolved via dispatch_digest -- that mapping is not
  1:1 (measurement.signature_id can be NULL). It is resolved directly from
  the JSONL row's own "signature" field against signature.signature_digest,
  exactly as inventory.load_measurements() and tune_promotion.
  validate_schedule() already treat that field.
- The stored correctness_evidence parent row is NOT trusted as authority.
  Its worst-of-seeds aggregates are re-derived from the child
  correctness_evidence_seed rows (the real evidence) and cross-checked
  against the stored aggregate; a mismatch fails closed.
- native_candidate_id is verified, not just candidate_id -- otherwise
  evidence proving candidate C beats native N1 could wrongly authorize
  promoting C against a different native N2 the row actually claims.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from . import correctness_evidence as ce


class CorrectnessGateError(RuntimeError):
    """Identity could not be resolved, or evidence is missing/invalid --
    always a REJECTION, never a promotion decided by absence of information."""


@dataclass(frozen=True)
class PromotionIdentity:
    build_id: int
    hardware_id: int
    signature_id: int
    native_candidate_id: int
    candidate_id: int


def resolve_promotion_identity(
    conn: sqlite3.Connection, *, dispatch_hex: str, signature_hex: str,
    hardware_hex: str, native_name: str, candidate_name: str,
) -> PromotionIdentity:
    """Resolve one promotable row's real DB identity, read-only.

    Fails closed (CorrectnessGateError) on anything ambiguous or missing --
    never falls back to a legacy key or an inferred mapping."""
    signature_row = conn.execute(
        "SELECT signature_id FROM signature WHERE signature_digest = ?",
        (bytes.fromhex(signature_hex),),
    ).fetchone()
    if signature_row is None:
        raise CorrectnessGateError(
            f"no signature row for signature digest {signature_hex!r} -- "
            f"this dispatch_db was not ingested from the same measurements"
        )
    signature_id = signature_row[0]

    hardware_row = conn.execute(
        "SELECT hardware_id FROM hardware WHERE hardware_digest = ?",
        (bytes.fromhex(hardware_hex),),
    ).fetchone()
    if hardware_row is None:
        raise CorrectnessGateError(
            f"no hardware row for hardware digest {hardware_hex!r}"
        )
    hardware_id = hardware_row[0]

    # build_id: resolved from the measurement table's OWN already-ingested
    # row for this exact (signature, hardware) pair -- never from the
    # legacy (source_revision, manifest_hash) key, which schema 4's
    # identity_scope split deliberately allows two distinct campaign builds
    # to share. Ambiguous (more than one build_id) or absent (zero) both
    # fail closed rather than guessing.
    build_rows = conn.execute(
        "SELECT DISTINCT build_id FROM measurement "
        "WHERE signature_id = ? AND hardware_id = ? AND dispatch_digest = ?",
        (signature_id, hardware_id, bytes.fromhex(dispatch_hex)),
    ).fetchall()
    if len(build_rows) != 1:
        raise CorrectnessGateError(
            f"expected exactly one build for dispatch {dispatch_hex!r} "
            f"signature/hardware pair in this dispatch_db, found {len(build_rows)} "
            f"-- this dispatch_db does not unambiguously pair with these measurements"
        )
    build_id = build_rows[0][0]

    def _candidate_id(stable_name: str) -> int:
        row = conn.execute(
            "SELECT candidate_id FROM candidate WHERE build_id = ? AND stable_name = ?",
            (build_id, stable_name),
        ).fetchone()
        if row is None:
            raise CorrectnessGateError(
                f"no candidate row for build_id={build_id} stable_name={stable_name!r}"
            )
        return row[0]

    return PromotionIdentity(
        build_id=build_id,
        hardware_id=hardware_id,
        signature_id=signature_id,
        native_candidate_id=_candidate_id(native_name),
        candidate_id=_candidate_id(candidate_name),
    )


@dataclass(frozen=True)
class CorrectnessBinding:
    """The exact production unit RV49 evidence is keyed at: one dispatch key
    resolving to one candidate, on one piece of hardware, against one native
    baseline. Deliberately the SAME granularity resolve_promotion_identity()
    already takes as keyword arguments -- a named wrapper, not a new identity
    model, so a future caller (the replay-cache exporter) has one canonical
    binding type to import rather than five loose positional hex strings."""
    dispatch_hex: str
    signature_hex: str
    hardware_hex: str
    native_name: str
    candidate_name: str


def resolve_correctness_binding(
    conn: sqlite3.Connection, binding: CorrectnessBinding,
) -> PromotionIdentity:
    """CorrectnessBinding-typed entry point for resolve_promotion_identity().
    Identical resolution, same fail-closed behavior -- see that function's
    own docstring for the identity rules (never a legacy source_revision/
    manifest_hash key, never an inferred signature_id, ambiguous or absent
    rows always raise)."""
    return resolve_promotion_identity(
        conn, dispatch_hex=binding.dispatch_hex, signature_hex=binding.signature_hex,
        hardware_hex=binding.hardware_hex, native_name=binding.native_name,
        candidate_name=binding.candidate_name,
    )


def require_correctness_binding(
    conn: sqlite3.Connection, binding: CorrectnessBinding, *,
    required_contract_version: str = ce.CONTRACT_VERSION,
    required_headroom_fraction: float = ce.DEFAULT_HEADROOM_FRACTION,
) -> PromotionIdentity:
    """Resolve identity for one binding and require it to pass the RV49
    correctness gate, or raise. Intended for a caller (e.g. the replay-cache
    exporter) that needs "prove this exact binding is production-safe or
    stop" as a single call, rather than tune_promotion.py's own two-step
    resolve-then-evaluate flow, which needs the intermediate identity for
    other purposes too."""
    identity = resolve_correctness_binding(conn, binding)
    passed, status = evaluate_correctness_gate(
        conn, identity, required_contract_version=required_contract_version,
        required_headroom_fraction=required_headroom_fraction,
    )
    if not passed:
        raise CorrectnessGateError(
            f"{binding.dispatch_hex}: candidate {binding.candidate_name!r} failed the "
            f"RV49 correctness gate ({status})"
        )
    return identity


def evaluate_correctness_gate(
    conn: sqlite3.Connection, identity: PromotionIdentity, *,
    required_contract_version: str = ce.CONTRACT_VERSION,
    required_headroom_fraction: float = ce.DEFAULT_HEADROOM_FRACTION,
) -> tuple[bool, str]:
    """Returns (passed, rejection_status_if_not_passed).

    T (threshold_t) is RE-DERIVED from the child correctness_evidence_seed
    rows' own threshold_t columns (HI67 threshold-authority fix -- each seed
    row carries the upstream threshold as ACTUALLY EMITTED by test-backend-
    ops for that run, not a caller-supplied Python float), the same way
    e_n_nmse/e_c_nmse/max_abs_* are re-derived rather than trusted off the
    parent. The parent row's own threshold_t is then cross-checked against
    that re-derived value below, exactly like the other aggregates.
    contract_version and headroom_fraction ARE promotion-owned policy and
    are independently required to match, never trusted off the row."""
    parent = conn.execute(
        "SELECT correctness_evidence_id, native_candidate_id, contract_version, "
        "threshold_t, headroom_fraction, seed_count, "
        "e_n_nmse, e_c_nmse, max_abs_native, max_abs_candidate "
        "FROM correctness_evidence "
        "WHERE build_id = ? AND hardware_id = ? AND signature_id = ? AND candidate_id = ?",
        (identity.build_id, identity.hardware_id, identity.signature_id, identity.candidate_id),
    ).fetchone()
    if parent is None:
        return False, "rejected_no_correctness_evidence"

    (evidence_id, native_candidate_id, contract_version, threshold_t,
     headroom_fraction, seed_count, parent_e_n, parent_e_c,
     parent_max_abs_n, parent_max_abs_c) = parent

    if native_candidate_id != identity.native_candidate_id:
        # Evidence for candidate C vs a DIFFERENT native baseline must not
        # authorize promoting C against THIS row's native.
        return False, "rejected_correctness_contract"
    if contract_version != required_contract_version:
        return False, "rejected_correctness_contract"
    if headroom_fraction != required_headroom_fraction:
        return False, "rejected_correctness_contract"

    seed_rows = conn.execute(
        "SELECT seed, reference_digest, e_n_nmse, e_c_nmse, max_abs_native, "
        "max_abs_candidate, native_execution_status, candidate_execution_status, "
        "threshold_t "
        "FROM correctness_evidence_seed WHERE correctness_evidence_id = ? ORDER BY seed",
        (evidence_id,),
    ).fetchall()
    if len(seed_rows) != seed_count:
        # The parent's own seed_count claim disagrees with what is actually
        # stored -- do not trust either number blindly.
        return False, "rejected_correctness_contract"

    seed_evidence = [
        ce.SeedEvidence(
            seed=row[0], reference_digest=row[1], e_n_nmse=row[2], e_c_nmse=row[3],
            max_abs_native=row[4], max_abs_candidate=row[5],
            native_execution_status=row[6], candidate_execution_status=row[7],
            threshold_t=row[8],
        )
        for row in seed_rows
    ]
    try:
        aggregate = ce.aggregate_seed_evidence(
            seed_evidence, headroom_fraction=headroom_fraction, contract_version=contract_version,
        )
    except ce.EvidenceError:
        # Too few seeds, a seed that did not execute cleanly, or the seeds
        # disagree on threshold_t -- the stored parent row claims evidence
        # that does not actually hold up.
        return False, "rejected_correctness_contract"

    # Cross-check: the parent row's own claimed aggregates -- INCLUDING
    # threshold_t -- must agree with what the child seed rows actually
    # recompute to (RV77 Q2: "the DB is evidence, not authority" -- the
    # parent is an index, the seeds are the real evidence). A mismatch means
    # the parent row is not a faithful summary of its own children -- do not
    # trust either number then.
    _TOLERANCE = 1e-12
    if (
        abs(aggregate.e_n_nmse - parent_e_n) > _TOLERANCE
        or abs(aggregate.e_c_nmse - parent_e_c) > _TOLERANCE
        or abs(aggregate.max_abs_native - parent_max_abs_n) > _TOLERANCE
        or abs(aggregate.max_abs_candidate - parent_max_abs_c) > _TOLERANCE
        or abs(aggregate.threshold_t - threshold_t) > _TOLERANCE
    ):
        return False, "rejected_correctness_contract"

    if not aggregate.dispatchable:
        return False, "rejected_correctness"

    return True, ""
