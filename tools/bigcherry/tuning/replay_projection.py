"""HI121 M4: offline, capability-gated compatibility projection of a
measurements JSONL for a specific target build.

Framing (see docs/planning/active/hip-autotune/HI121.md): the dispatch_db
measurement history is a durable, multi-generation knowledge store; a
replay cache is a cheap, TARGET-SPECIFIC PROJECTION of only the
measurements known to be safe for that one binary. This module produces
that projection -- it never touches the production C++ resolver, the
replay wire format, or replay.py's own reader/writer, which stay exactly
as they are today. The projected file is an ordinary measurements JSONL
that `replay.build()` consumes completely unchanged.

Filtering rule per result row, all of which must hold to retain a row:
  * the row's own signature digest resolves to a real dispatch_db signature
    row with parseable canonical content (an unresolvable signature is a
    data problem, not a capability problem -- fails the whole projection,
    not just that row, since it means this measurements/dispatch_db pairing
    itself is suspect);
  * hip_required_capabilities() does not raise UnsupportedSignatureDomain
    for that signature's canonical content;
  * the SOURCE build's verified, DB-attested producer_capabilities mask
    (build_capability, persisted once at ingest time by inventory.py's
    _verify_and_persist_hip_capabilities -- never re-derived here) is a
    superset of what's required;
  * the TARGET build's producer_capabilities (read from its manifest,
    itself verified against its own materialized source root) is ALSO a
    superset of what's required -- a target that cannot itself distinguish
    the relevant semantics must not receive the row either, even if the
    source could (see HI121 round 9's own self-review finding).

Retained rows are copied byte-for-byte from the source JSON object --
this function never mutates a result row's own content.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import catalog
from . import hip_capabilities as hc
from . import replay as replay_module
from . import signature_capabilities as sc
from .capabilities import CapabilityMask128, CapabilityMaskError
from ..source.audit import git_revision


class ProjectionError(RuntimeError):
    """The requested compatibility projection cannot be proven safe."""


@dataclass(frozen=True)
class ProjectionSummary:
    examined: int
    retained: int
    omitted_missing_producer_capability: int
    omitted_missing_target_capability: int
    omitted_unsupported_domain: int


def _load_source_capabilities(
    conn: sqlite3.Connection, *, source_build_id: int, header: dict[str, Any],
) -> CapabilityMask128:
    """Resolve and verify the source build's DB-attested capability mask.

    The DB attestation alone is not proof that THIS measurements artifact is
    the one it was attested for: ingest verifies header<->manifest<->DB
    agreement once, at load time, but a later artifact (a hand-edited copy, a
    forged header, an artifact from a different build that happens to share
    this build's source_revision/manifest_hash) is a different question this
    function must independently answer -- never just trust the DB row because
    SOME artifact once satisfied it. Requires the CURRENT header's own
    producer_capabilities and build_descriptor_hash (when present) to agree
    with the DB, matching inventory.py's ingest-time strength rather than a
    weaker subset of it.
    """
    build_row = conn.execute(
        "SELECT source_revision, manifest_hash, build_descriptor_hash FROM build WHERE build_id = ?",
        (source_build_id,),
    ).fetchone()
    if build_row is None:
        raise ProjectionError(f"source_build_id={source_build_id} does not exist in this dispatch_db")
    db_source_revision, db_manifest_hash, db_descriptor_hash = build_row

    if header.get("source_revision") != db_source_revision or header.get("manifest_hash") != db_manifest_hash:
        raise ProjectionError(
            f"measurements header does not match build_id={source_build_id}'s own "
            f"source_revision/manifest_hash -- refusing to project against the wrong build"
        )
    header_descriptor_hash = header.get("build_descriptor_hash")
    if (
        db_descriptor_hash is not None
        and header_descriptor_hash is not None
        and header_descriptor_hash != db_descriptor_hash
    ):
        raise ProjectionError(
            f"measurements header build_descriptor_hash={header_descriptor_hash!r} does not "
            f"match build_id={source_build_id}'s own build_descriptor_hash={db_descriptor_hash!r}"
        )

    cap_row = conn.execute(
        "SELECT producer_capabilities FROM build_capability WHERE build_id = ? AND backend = 'hip'",
        (source_build_id,),
    ).fetchone()
    if cap_row is None:
        raise ProjectionError(
            f"source_build_id={source_build_id} has no verified hip producer_capabilities "
            f"attestation (see inventory.load_measurements' manifest-verified ingest) -- "
            f"cannot determine what this build's producer actually knew how to evaluate"
        )
    db_mask = CapabilityMask128.from_bytes(cap_row[0])

    header_caps_hex = header.get("producer_capabilities")
    if isinstance(header_caps_hex, str):
        # Older measurements artifacts predate this header field -- fall
        # back to trusting the DB attestation alone (matches inventory.py's
        # own "missing field is not an error" stance). When the field IS
        # present, it must agree: this is what stops a DIFFERENT artifact
        # that happens to share this build's identity fields from silently
        # inheriting an attestation it was never actually checked against.
        try:
            header_mask = CapabilityMask128.from_hex(header_caps_hex)
        except CapabilityMaskError as exc:
            raise ProjectionError(f"measurements header producer_capabilities is malformed: {exc}") from exc
        if header_mask != db_mask:
            raise ProjectionError(
                f"measurements header producer_capabilities={header_caps_hex!r} does not match "
                f"build_id={source_build_id}'s DB-attested mask {db_mask.to_hex()!r} -- refusing "
                f"to trust an attestation this specific artifact was never verified against"
            )
    return db_mask


def _load_target_capabilities(target_manifest: dict[str, Any], *, vendor_root: Path) -> CapabilityMask128:
    caps_hex = target_manifest.get("producer_capabilities")
    if not isinstance(caps_hex, str):
        raise ProjectionError("target manifest has no producer_capabilities field")
    if catalog.manifest_hash(target_manifest) != target_manifest.get("manifest_hash"):
        raise ProjectionError(
            "target manifest's recomputed manifest_hash does not match its own manifest_hash "
            "field -- the manifest file may be corrupted or hand-edited"
        )
    # manifest_hash() deliberately excludes source_revision (it is scoped to
    # variant_set/candidate set, per replay.py's own comment on the same
    # point) -- so a correct manifest_hash alone does not prove vendor_root
    # is actually the revision the manifest claims. A different checkout
    # that happens to declare the same producer_capabilities would otherwise
    # pass silently. Verify the real git identity independently.
    manifest_revision = target_manifest.get("source_revision")
    if not isinstance(manifest_revision, str) or not manifest_revision:
        raise ProjectionError("target manifest has no source_revision field")
    actual_revision, _dirty = git_revision(vendor_root, check_dirty=False)
    if actual_revision != manifest_revision:
        raise ProjectionError(
            f"target manifest claims source_revision={manifest_revision!r}, but vendor_root "
            f"{vendor_root} is actually at {actual_revision!r} -- this manifest was not "
            f"generated from the exact materialized root it claims"
        )
    declared = hc.load_declared_producer_capabilities(vendor_root)
    try:
        claimed = CapabilityMask128.from_hex(caps_hex)
    except CapabilityMaskError as exc:
        raise ProjectionError(f"target manifest's producer_capabilities is malformed: {exc}") from exc
    if declared != claimed:
        raise ProjectionError(
            f"target manifest claims producer_capabilities={caps_hex!r}, but the target "
            f"vendor_root's OWN source declaration is {declared.to_hex()!r} -- this manifest "
            f"was not generated from the exact source root it claims"
        )
    return claimed


def _load_canonical_signature(conn: sqlite3.Connection, signature_hex: str) -> dict[str, Any]:
    row = conn.execute(
        "SELECT canonical_json FROM signature WHERE signature_digest = ?",
        (bytes.fromhex(signature_hex),),
    ).fetchone()
    if row is None:
        raise ProjectionError(
            f"no signature row for signature digest {signature_hex!r} -- this measurements "
            f"file was not ingested into the supplied dispatch_db"
        )
    return json.loads(row[0])


def _raw_result_lines_by_dispatch(measurements_path: Path) -> dict[str, str]:
    """Map each result row's normalized (lowercase) dispatch digest to its
    ORIGINAL raw JSONL line text -- so a retained row can be written back
    byte-for-byte rather than re-serialized through json.dumps(), which can
    silently change whitespace/escaping/key-order even for semantically
    identical content (round 9's own explicit byte-for-byte requirement)."""
    raw_by_dispatch: dict[str, str] = {}
    with measurements_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or record.get("kind") != "result" or not record.get("winner"):
                continue
            dispatch = record.get("dispatch")
            if isinstance(dispatch, str) and re.fullmatch(r"[0-9a-fA-F]{32}", dispatch):
                raw_by_dispatch[dispatch.lower()] = stripped
    return raw_by_dispatch


def project_measurements(
    measurements_path: Path, output_path: Path, *,
    dispatch_db: Path, source_build_id: int,
    target_manifest_path: Path, vendor_root: Path,
) -> ProjectionSummary:
    """Filter one measurements artifact to rows safe for a target HIP build.

    Raises ProjectionError for anything that makes the WHOLE projection
    untrustworthy (unresolvable build/capability provenance, a malformed or
    misattributed target manifest) -- an individual row's own applicability
    failure is instead counted and omitted, never raised, since "this one
    signature needs a rerun" is an expected, ordinary outcome.

    The OUTPUT header is target-bound (its source_revision/manifest_hash/
    build_descriptor_hash are rewritten to the TARGET's own, under an
    explicit `hi121_source_provenance` block recording the true origin) so
    the projected file satisfies replay.build()'s own real requirement that
    the measurements producer's source_revision match the target manifest's
    -- without this, a genuine cross-generation projection (the central
    multi-generation reuse case HI121 exists for) could never actually reach
    replay.build() at all, since the unmodified source header's revision
    would almost never equal a different target build's revision. Retained
    RESULT rows are copied byte-for-byte from the source file and are never
    rewritten -- only the header identity changes, since the header is what
    replay.build() actually gates on, not individual result rows.
    """
    header, results = replay_module.read_results(measurements_path, require_header=True)
    raw_lines = _raw_result_lines_by_dispatch(measurements_path)

    target_manifest = json.loads(Path(target_manifest_path).read_text(encoding="utf-8"))
    target_caps = _load_target_capabilities(target_manifest, vendor_root=vendor_root)

    conn = sqlite3.connect(str(dispatch_db))
    try:
        source_caps = _load_source_capabilities(conn, source_build_id=source_build_id, header=header)

        examined = 0
        retained_dispatches: list[str] = []
        omitted_missing_producer = 0
        omitted_missing_target = 0
        omitted_unsupported = 0

        for row in results:
            examined += 1
            signature_hex = row.get("signature")
            if not isinstance(signature_hex, str):
                raise ProjectionError(f"result row missing a valid signature digest: {row!r}")
            canonical = _load_canonical_signature(conn, signature_hex)
            try:
                required = sc.hip_required_capabilities(canonical, vendor_root=vendor_root)
            except sc.UnsupportedSignatureDomain:
                omitted_unsupported += 1
                continue
            if not source_caps.contains(required):
                omitted_missing_producer += 1
                continue
            if not target_caps.contains(required):
                omitted_missing_target += 1
                continue
            retained_dispatches.append(row["dispatch"])
    finally:
        conn.close()

    output_header = dict(header)
    output_header["hi121_source_provenance"] = {
        "source_revision": header.get("source_revision"),
        "manifest_hash": header.get("manifest_hash"),
        "build_descriptor_hash": header.get("build_descriptor_hash"),
        "source_build_id": source_build_id,
    }
    output_header["source_revision"] = target_manifest.get("source_revision")
    output_header["manifest_hash"] = target_manifest.get("manifest_hash")
    target_descriptor_hash = (target_manifest.get("build_descriptor") or {}).get("descriptor_hash")
    if target_descriptor_hash is not None:
        output_header["build_descriptor_hash"] = target_descriptor_hash
    elif "build_descriptor_hash" in output_header:
        del output_header["build_descriptor_hash"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(output_header, separators=(",", ":")) + "\n")
        for dispatch in retained_dispatches:
            handle.write(raw_lines[dispatch] + "\n")

    return ProjectionSummary(
        examined=examined,
        retained=len(retained_dispatches),
        omitted_missing_producer_capability=omitted_missing_producer,
        omitted_missing_target_capability=omitted_missing_target,
        omitted_unsupported_domain=omitted_unsupported,
    )
