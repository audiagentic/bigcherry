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
from . import inventory
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
    omitted_candidate_mismatch: int


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
    producer_capabilities and build_descriptor_hash to agree
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
    # HI121 review follow-up: these fields are now REQUIRED, not "checked
    # only if present". A header missing them could otherwise silently fall
    # back to trusting the bare DB attestation with no per-artifact proof at
    # all -- exactly the gap a forged/edited/mismatched artifact could
    # exploit. The one legitimate "missing field" case (a genuinely older
    # measurements artifact predating this header field) is exactly the
    # case that must NOT be treated as HI121-projectable -- it has no way to
    # prove it is the artifact the DB attestation was ever checked against.
    header_descriptor_hash = header.get("build_descriptor_hash")
    if not isinstance(db_descriptor_hash, str) or not db_descriptor_hash:
        raise ProjectionError(
            f"source_build_id={source_build_id} has no build_descriptor_hash in the DB -- "
            "a descriptor-less build is capability-unknown and is not HI121-projectable"
        )
    if not isinstance(header_descriptor_hash, str) or not header_descriptor_hash:
        raise ProjectionError(
            "measurements header has no build_descriptor_hash -- an artifact predating this "
            "field cannot be proven to be the one its build's capability attestation was "
            "verified against, and is not HI121-projectable"
        )
    if header_descriptor_hash != db_descriptor_hash:
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
    # Source parsing must be exactly as fail-closed as target parsing
    # (load_declared_producer_capabilities() already rejects an unknown bit
    # in a target declaration) -- an older M4 must not silently ignore a
    # future capability bit a newer producer legitimately claimed, since
    # HI121's frozen identity epoch means a brand new semantic axis can
    # arrive without any schema bump to force this code to be updated too.
    known_mask = hc.known_hip_capability_mask()
    if (db_mask.value & ~known_mask.value) != 0:
        raise ProjectionError(
            f"build_id={source_build_id}'s DB-attested capability mask {db_mask.to_hex()!r} sets "
            f"bit(s) this tooling's HipCapability registry does not recognize -- refusing to "
            f"silently ignore a capability newer than this code understands"
        )

    header_caps_hex = header.get("producer_capabilities")
    if not isinstance(header_caps_hex, str):
        raise ProjectionError(
            "measurements header has no producer_capabilities -- an artifact predating this "
            "field is not HI121-projectable (see build_descriptor_hash requirement above)"
        )
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


def _recompute_manifest_descriptor(
    manifest: dict[str, Any], *, label: str,
) -> dict[str, Any]:
    """Prove that a manifest's embedded descriptor is derived from its content."""
    embedded = manifest.get("build_descriptor")
    if not isinstance(embedded, dict):
        raise ProjectionError(
            f"{label} manifest has no build_descriptor -- a descriptor-less manifest "
            "is not HI121-projectable"
        )
    try:
        recomputed = catalog.build_descriptor(manifest)
    except (KeyError, TypeError, ValueError, catalog.CatalogError) as exc:
        raise ProjectionError(
            f"{label} manifest build_descriptor cannot be recomputed from its content"
        ) from exc
    if recomputed != embedded:
        raise ProjectionError(
            f"{label} manifest's embedded build_descriptor does not exactly match "
            "catalog.build_descriptor() recomputed from the manifest content"
        )
    descriptor_hash = recomputed.get("descriptor_hash")
    if not isinstance(descriptor_hash, str) or not descriptor_hash:
        raise ProjectionError(
            f"{label} manifest has no non-empty recomputed build_descriptor hash"
        )
    return recomputed


def _load_source_manifest(
    conn: sqlite3.Connection, *, source_manifest_path: Path,
    source_build_id: int, header: dict[str, Any],
) -> tuple[dict[str, Any], CapabilityMask128]:
    """Load only a source manifest proven to describe ``source_build_id``.

    Candidate descriptors are intentionally not inspected by the caller until
    every manifest, header, DB, and capability attestation check below passes.
    """
    try:
        manifest = json.loads(Path(source_manifest_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProjectionError(f"source manifest {source_manifest_path} is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise ProjectionError(f"source manifest {source_manifest_path} is not a JSON object")

    build_row = conn.execute(
        "SELECT source_revision, manifest_hash FROM build WHERE build_id = ?",
        (source_build_id,),
    ).fetchone()
    if build_row is None:
        raise ProjectionError(f"source_build_id={source_build_id} does not exist in this dispatch_db")
    db_source_revision, db_manifest_hash = build_row

    try:
        recomputed_manifest_hash = catalog.manifest_hash(manifest)
    except (KeyError, TypeError, ValueError) as exc:
        raise ProjectionError(
            "source manifest hash cannot be recomputed from its content"
        ) from exc
    if recomputed_manifest_hash != manifest.get("manifest_hash"):
        raise ProjectionError(
            "source manifest's recomputed manifest_hash does not match its own "
            "manifest_hash field -- the manifest file may be corrupted or hand-edited"
        )
    if manifest.get("manifest_hash") != db_manifest_hash:
        raise ProjectionError(
            f"source manifest manifest_hash={manifest.get('manifest_hash')!r} does not "
            f"match source_build_id={source_build_id}'s DB manifest_hash={db_manifest_hash!r}"
        )

    manifest_revision = manifest.get("source_revision")
    if not isinstance(manifest_revision, str) or not manifest_revision:
        raise ProjectionError("source manifest has no source_revision field")
    if manifest_revision != db_source_revision or manifest_revision != header.get("source_revision"):
        raise ProjectionError(
            "source manifest source_revision does not match both the source build DB row "
            "and the current measurements header"
        )
    if manifest.get("manifest_hash") != header.get("manifest_hash"):
        raise ProjectionError(
            "source manifest manifest_hash does not match the current measurements header"
        )

    descriptor = _recompute_manifest_descriptor(manifest, label="source")
    source_caps = _load_source_capabilities(
        conn, source_build_id=source_build_id, header=header,
    )
    manifest_caps_hex = manifest.get("producer_capabilities")
    if not isinstance(manifest_caps_hex, str):
        raise ProjectionError("source manifest has no producer_capabilities field")
    try:
        manifest_caps = CapabilityMask128.from_hex(manifest_caps_hex)
    except CapabilityMaskError as exc:
        raise ProjectionError(f"source manifest's producer_capabilities is malformed: {exc}") from exc
    if manifest_caps != source_caps:
        raise ProjectionError(
            f"source manifest producer_capabilities={manifest_caps_hex!r} does not match "
            f"source_build_id={source_build_id}'s DB/header attestation {source_caps.to_hex()!r}"
        )
    if descriptor["descriptor_hash"] != header.get("build_descriptor_hash"):
        raise ProjectionError(
            "source manifest's recomputed build_descriptor hash does not match the "
            "current measurements header"
        )
    return manifest, source_caps


def _require_row_belongs_to_build(
    conn: sqlite3.Connection, *, source_build_id: int, row: dict[str, Any], signature_hex: str,
) -> None:
    """Prove that the artifact row is the exact DB-recorded winner identity.

    A verified build-level capability attestation is not proof that any GIVEN
    result row in the CURRENT artifact actually belongs to that build -- rows
    could be edited, mixed in from another artifact, or otherwise not be the
    ones the attestation covers.  Bind every identity-bearing field available
    in the artifact to the authoritative winner row, including the winner's
    candidate and native names.  The candidate join also makes sure the
    winner's foreign key resolves to the candidate with that stable name in
    this same build; a dispatch/signature membership check alone cannot prove
    any of those claims.
    """
    dispatch_hex = row.get("dispatch")
    if not isinstance(dispatch_hex, str):
        raise ProjectionError(f"result row missing a valid dispatch digest: {row!r}")
    hardware_hex = row.get("hardware")
    if not isinstance(hardware_hex, str):
        raise ProjectionError(f"result row missing a valid hardware digest: {row!r}")
    winner_name = row.get("winner")
    if not isinstance(winner_name, str):
        raise ProjectionError(f"result row missing a valid winner: {row!r}")
    native_name = row.get("native")
    if not isinstance(native_name, str):
        raise ProjectionError(f"result row missing a valid native candidate: {row!r}")
    try:
        dispatch_bytes = bytes.fromhex(dispatch_hex)
        hardware_bytes = bytes.fromhex(hardware_hex)
        signature_bytes = bytes.fromhex(signature_hex)
    except ValueError as exc:
        raise ProjectionError(f"result row contains a malformed identity digest: {row!r}") from exc
    match = conn.execute(
        "SELECT 1 "
        "FROM measurement m "
        "JOIN winner w ON w.build_id = m.build_id "
        " AND w.hardware_id = m.hardware_id "
        " AND w.signature_id = m.signature_id "
        " AND w.dispatch_digest = m.dispatch_digest "
        "JOIN hardware h ON h.hardware_id = w.hardware_id "
        " AND h.hardware_digest = ? "
        "JOIN signature s ON s.signature_id = w.signature_id "
        " AND s.signature_digest = ? "
        "JOIN candidate c ON c.candidate_id = w.candidate_id "
        " AND c.build_id = w.build_id "
        " AND c.stable_name = w.stable_name "
        "WHERE w.build_id = ? "
        " AND w.dispatch_digest = ? "
        " AND w.stable_name = ? "
        " AND w.native_stable_name = ?",
        (hardware_bytes, signature_bytes, source_build_id, dispatch_bytes, winner_name, native_name),
    ).fetchone()
    if match is None:
        raise ProjectionError(
            f"result row identity (dispatch={dispatch_hex!r}, signature={signature_hex!r}, "
            f"hardware={hardware_hex!r}, winner={winner_name!r}, native={native_name!r}) "
            f"does not resolve to the authoritative winner recorded against "
            f"build_id={source_build_id} -- this row cannot be proven to belong to the "
            f"build whose capabilities were verified"
        )


def _candidate_implementation_is_equivalent(
    *, winner_name: str, source_candidates_by_name: dict[str, dict[str, Any]],
    target_candidates_by_name: dict[str, dict[str, Any]],
) -> bool:
    """Signature-semantic capability compatibility proves A and B AGREE on
    what a dispatch signature MEANS -- it says nothing about whether the
    CANDIDATE that won on the source is the same implementation that will
    execute on the target. A stable_name surviving unchanged across a real
    kernel/implementation change (config knobs, or the manually-maintained
    IMPLEMENTATION_VERSION never bumped) would otherwise let a completely
    different implementation silently inherit a reuse decision based only
    on signature compatibility. Requires the winner's full candidate
    descriptor to be IDENTICAL (not just present) in both manifests' own
    candidate lists -- a per-row applicability question like any other
    (an implementation that genuinely changed just needs a rerun, the same
    as an unsupported signature domain), not a whole-projection abort."""
    source_candidate = source_candidates_by_name.get(winner_name)
    target_candidate = target_candidates_by_name.get(winner_name)
    if source_candidate is None or target_candidate is None:
        return False
    # Descriptor equality remains required.  It is not sufficient because the
    # descriptor's implementation_version is manually maintained.  Catalog
    # generation now persists a source-derived digest for the explicit
    # family dispatch/kernel slice; missing or malformed identities fail
    # closed.  This proves only equality of that file slice, not compiler
    # flags, unlisted transitive headers, vendor libraries, or GPU behavior.
    # There is no candidate->patch/composition reference in the current
    # catalog, so patch effects are covered only when they change this slice.
    if source_candidate != target_candidate:
        return False
    source_digest = source_candidate.get("implementation_digest")
    target_digest = target_candidate.get("implementation_digest")
    return (
        isinstance(source_digest, str)
        and bool(source_digest)
        and isinstance(target_digest, str)
        and bool(target_digest)
        and source_digest == target_digest
    )


def _load_target_capabilities(target_manifest: dict[str, Any], *, vendor_root: Path) -> CapabilityMask128:
    if not isinstance(target_manifest, dict):
        raise ProjectionError("target manifest is not a JSON object")
    caps_hex = target_manifest.get("producer_capabilities")
    if not isinstance(caps_hex, str):
        raise ProjectionError("target manifest has no producer_capabilities field")
    try:
        manifest_hash = catalog.manifest_hash(target_manifest)
    except (KeyError, TypeError, ValueError) as exc:
        raise ProjectionError("target manifest hash cannot be recomputed from its content") from exc
    if manifest_hash != target_manifest.get("manifest_hash"):
        raise ProjectionError(
            "target manifest's recomputed manifest_hash does not match its own manifest_hash "
            "field -- the manifest file may be corrupted or hand-edited"
        )
    _recompute_manifest_descriptor(target_manifest, label="target")
    # manifest_hash() deliberately excludes source_revision (it is scoped to
    # variant_set/candidate set, per replay.py's own comment on the same
    # point) -- so a correct manifest_hash alone does not prove vendor_root
    # is actually the revision the manifest claims. A different checkout
    # that happens to declare the same producer_capabilities would otherwise
    # pass silently. Verify the real git identity independently.
    manifest_revision = target_manifest.get("source_revision")
    if not isinstance(manifest_revision, str) or not manifest_revision:
        raise ProjectionError("target manifest has no source_revision field")
    actual_revision, dirty = git_revision(vendor_root, check_dirty=True)
    if actual_revision != manifest_revision:
        raise ProjectionError(
            f"target manifest claims source_revision={manifest_revision!r}, but vendor_root "
            f"{vendor_root} is actually at {actual_revision!r} -- this manifest was not "
            f"generated from the exact materialized root it claims"
        )
    if dirty:
        # A dirty tree with the SAME manifest_hash/build_descriptor_hash and
        # the SAME declared producer_capabilities could still have different
        # actual kernel/dispatch source than what was verified -- the digest
        # checks above only cover fields explicitly hashed into those
        # identities, not arbitrary uncommitted source edits. Fail closed
        # rather than trust an uncommitted checkout's identity claims.
        raise ProjectionError(
            f"vendor_root {vendor_root} has uncommitted changes -- refusing to trust its "
            f"declared source_revision/producer_capabilities identity against a dirty tree"
        )
    # Adversarial-review follow-up (HI124 composition gap): everything above
    # proves vendor_root IS the exact, clean revision the target manifest
    # claims -- it does NOT prove any given candidate's embedded
    # implementation_digest was actually computed from THIS root's real
    # files. Without this, a self-consistent target manifest could copy a
    # stale (e.g. source-side) implementation_digest into its own candidate
    # entries: manifest_hash/descriptor recompute cleanly (they hash
    # whatever is in the dict, forged value included), the git revision
    # check passes (a real, correctly-identified checkout), and
    # _candidate_implementation_is_equivalent() would then compare two
    # digests that were never independently verified against live target
    # bytes. Recompute every candidate's digest fresh from vendor_root and
    # require it to match the manifest's embedded value exactly.
    for candidate in target_manifest.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        embedded_digest = candidate.get("implementation_digest")
        if not isinstance(embedded_digest, str) or not embedded_digest:
            continue
        try:
            recomputed_digest = catalog.candidate_implementation_digest(candidate, vendor_root)
        except catalog.CatalogError as exc:
            raise ProjectionError(
                f"target candidate {candidate.get('stable_name')!r} implementation digest "
                f"cannot be recomputed from vendor_root {vendor_root}: {exc}"
            ) from exc
        if recomputed_digest != embedded_digest:
            raise ProjectionError(
                f"target candidate {candidate.get('stable_name')!r} embedded "
                f"implementation_digest={embedded_digest!r} does not match the digest "
                f"independently recomputed from vendor_root {vendor_root} "
                f"({recomputed_digest!r}) -- the manifest was not generated from this "
                f"root's real implementation source"
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


def _raw_result_lines_by_dispatch(measurements_path: Path) -> dict[str, bytes]:
    """Map each result row's normalized (lowercase) dispatch digest to its
    ORIGINAL raw JSONL line bytes -- so a retained row can be written back
    byte-for-byte rather than re-serialized through json.dumps(), which can
    silently change whitespace/escaping/key-order even for semantically
    identical content (round 9's own explicit byte-for-byte requirement).
    Operates on raw bytes, not decoded text: a text-mode read with
    errors="replace" would itself silently mutate any invalid byte before
    this function ever gets a chance to preserve it."""
    raw_by_dispatch: dict[str, bytes] = {}
    data = Path(measurements_path).read_bytes()
    for line in data.split(b"\n"):
        if not line.strip():
            continue
        try:
            record = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict) or record.get("kind") != "result" or not record.get("winner"):
            continue
        dispatch = record.get("dispatch")
        if isinstance(dispatch, str) and re.fullmatch(r"[0-9a-fA-F]{32}", dispatch):
            raw_by_dispatch[dispatch.lower()] = line
    return raw_by_dispatch


def project_measurements(
    measurements_path: Path, output_path: Path, *,
    dispatch_db: Path, source_build_id: int, source_manifest_path: Path,
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

    Two additional real proofs beyond signature-capability compatibility,
    both required per row: (1) the row must resolve to an actual measurement
    recorded against source_build_id -- a verified BUILD-level capability
    attestation is not proof any GIVEN row in THIS artifact belongs to that
    build; (2) the winner's full candidate descriptor must be identical in
    both the source and target manifests -- signature-semantic compatibility
    proves A and B agree on what a dispatch MEANS, not that the candidate
    measured on A is still the same implementation that will execute on B.
    """
    output_path = Path(output_path)
    input_paths = {
        "measurements": Path(measurements_path).resolve(),
        "dispatch_db": Path(dispatch_db).resolve(),
        "source_manifest": Path(source_manifest_path).resolve(),
        "target_manifest": Path(target_manifest_path).resolve(),
    }
    resolved_output = output_path.resolve()
    for label, path in input_paths.items():
        if resolved_output == path:
            raise ProjectionError(
                f"output_path {output_path} resolves to the same file as the {label} input "
                f"-- refusing to overwrite an input this projection reads from"
            )

    header, results = replay_module.read_results(measurements_path, require_header=True)
    raw_lines = _raw_result_lines_by_dispatch(measurements_path)

    target_manifest = json.loads(Path(target_manifest_path).read_text(encoding="utf-8"))
    target_caps = _load_target_capabilities(target_manifest, vendor_root=vendor_root)

    conn = sqlite3.connect(f"file:{Path(dispatch_db)}?mode=ro", uri=True)
    try:
        try:
            inventory._require_current_schema(conn)
        except inventory.RecordError as exc:
            raise ProjectionError(str(exc)) from exc
        source_manifest, source_caps = _load_source_manifest(
            conn, source_manifest_path=source_manifest_path,
            source_build_id=source_build_id, header=header,
        )
        source_candidates_by_name = {
            c["stable_name"]: c for c in source_manifest.get("candidates", [])
        }
        target_candidates_by_name = {
            c["stable_name"]: c for c in target_manifest.get("candidates", [])
        }

        examined = 0
        retained_dispatches: list[str] = []
        omitted_missing_producer = 0
        omitted_missing_target = 0
        omitted_unsupported = 0
        omitted_candidate = 0

        for row in results:
            examined += 1
            signature_hex = row.get("signature")
            if not isinstance(signature_hex, str):
                raise ProjectionError(f"result row missing a valid signature digest: {row!r}")
            _require_row_belongs_to_build(
                conn, source_build_id=source_build_id, row=row, signature_hex=signature_hex,
            )
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
            winner_name = row.get("winner")
            if not isinstance(winner_name, str):
                raise ProjectionError(f"result row missing a valid winner: {row!r}")
            if not _candidate_implementation_is_equivalent(
                winner_name=winner_name,
                source_candidates_by_name=source_candidates_by_name,
                target_candidates_by_name=target_candidates_by_name,
            ):
                omitted_candidate += 1
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    with tmp_path.open("wb") as handle:
        handle.write(json.dumps(output_header, separators=(",", ":")).encode("utf-8") + b"\n")
        for dispatch in retained_dispatches:
            handle.write(raw_lines[dispatch] + b"\n")
    tmp_path.replace(output_path)

    return ProjectionSummary(
        examined=examined,
        retained=len(retained_dispatches),
        omitted_missing_producer_capability=omitted_missing_producer,
        omitted_missing_target_capability=omitted_missing_target,
        omitted_unsupported_domain=omitted_unsupported,
        omitted_candidate_mismatch=omitted_candidate,
    )
