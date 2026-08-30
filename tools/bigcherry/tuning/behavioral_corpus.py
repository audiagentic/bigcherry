"""HTR03: versioned, configurable behavioral-regression corpus.

Before this module, HI143's detection corpus was exactly ONE hardcoded,
frozen prompt (``behavioral_gate.load_hi141_regression_vector()``),
selected via a hardcoded ``_default_behavioral_corpus()`` function in
workflow.py -- adding a newly-discovered regression scenario required a
code change. Applicability was a single boolean ``require_mtp`` inferred
by string-matching ``'--spec-type'`` in a runtime profile's server_args --
not a general "decision-sensitive workload class" concept.

Adversarially designed with GPT (session ses_330ae3c055084f38, 2026-08-29
and 2026-08-30). Three independent identities, deliberately never
conflated (GPT explicit correction):

    behavioral_gate_contract_version -- comparison SEMANTICS (the
        hard_fail/exact_pass/behavior_changed three-state contract
        itself; see behavioral_gate.py's CONTRACT_VERSION-equivalent).
    corpus_schema_version -- the manifest FILE FORMAT.
    corpus edition + content_digest -- the exact curated CONTENTS (which
        vectors, with what parameters). The (edition, content_digest)
        PAIR is the actual immutable identity -- editing an edition's
        vectors in place is a policy convention this module documents and
        expects, NOT something load_corpus_edition() enforces or detects
        (GPT review round 2, 2026-08-30, correcting an earlier overclaim):
        nothing stops someone from changing this file's vectors while
        keeping the same `edition` name, and the loader will accept it
        with a newly-computed digest. What this DOES guarantee: a past
        receipt's own recorded content_digest lets a reviewer detect after
        the fact whether "edition X" meant the same thing then as it does
        now, by comparing digests -- reproducibility is verifiable, not
        structurally prevented from drifting.

Applicability: a vector applies to a campaign iff its ``applies_to`` list
intersects the runtime profile's ``behavioral_classes``
(core.config.RuntimeProfile.behavioral_classes). Classes are a
DATA-DEFINED closed vocabulary declared by the manifest itself (the
``classes`` list) -- not a Python enum (a new class would need a code
deployment) and not unconstrained free-form tags (a typo would silently
drop required coverage with no error). Both runtime profiles' and
vectors' class tags are validated against this declared registry.

Per-vector ``requirements`` (e.g. ``mtp-telemetry``) is a SEPARATE concept
from ``applies_to`` (GPT explicit, 2026-08-30): applicability says WHEN a
vector is relevant; requirements says WHAT EXECUTION TELEMETRY the
evaluator must be able to read to judge it. A new behavioral class reusing
EXISTING telemetry stays pure data (no code change); a genuinely new
telemetry requirement needs new evaluator code -- which is correct and
expected, not something this module should paper over.

Explicitly, deliberately NOT built here (GPT's premature-abstraction
boundary, unchanged): a trigger-plugin API, live/production-traffic
monitoring, automatic corpus rotation, raw-log archival, or a
hierarchical/tag-inheritance policy language.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path

CORPUS_SCHEMA_VERSION = 1
KNOWN_REQUIREMENTS = frozenset({"mtp-telemetry"})


class BehavioralCorpusError(RuntimeError):
    """A corpus manifest or applicability resolution failed closed."""


@dataclass(frozen=True)
class CorpusVectorSpec:
    id: str
    prompt_file: str
    prompt_sha256: str
    n_predict: int
    seed: int
    applies_to: tuple[str, ...]
    requirements: tuple[str, ...]
    scenario: str
    provenance: str
    content_digest: str  # sha256 over this spec's own canonical fields


@dataclass(frozen=True)
class CorpusEdition:
    schema_version: int
    edition: str
    classes: tuple[str, ...]
    vectors: tuple[CorpusVectorSpec, ...]
    content_digest: str  # sha256 over the whole edition's canonical contents


def _canonical_digest(parts: list[str]) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(part.encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


def load_corpus_edition(manifest_path: Path, fixtures_dir: Path) -> CorpusEdition:
    """Parse and validate a corpus manifest TOML file. Fails closed on:
    an unsupported schema_version, an unknown class tag anywhere
    (a vector's applies_to, or a requirement outside KNOWN_REQUIREMENTS),
    a prompt file whose actual content does not match its declared
    prompt_sha256 (the immutability guarantee -- an edition's vectors
    must never silently drift), or any vector missing a required field.
    """
    try:
        raw = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise BehavioralCorpusError(f"{manifest_path}: invalid TOML: {exc}") from exc

    schema_version = raw.get("schema_version")
    if schema_version != CORPUS_SCHEMA_VERSION:
        raise BehavioralCorpusError(
            f"{manifest_path}: unsupported corpus schema_version {schema_version!r} "
            f"(expected {CORPUS_SCHEMA_VERSION})"
        )
    edition = raw.get("edition")
    if not isinstance(edition, str) or not edition:
        raise BehavioralCorpusError(f"{manifest_path}: missing/empty 'edition'")
    classes = raw.get("classes")
    if not isinstance(classes, list) or not all(isinstance(c, str) and c for c in classes):
        raise BehavioralCorpusError(f"{manifest_path}: 'classes' must be a list of non-empty strings")
    known_classes = frozenset(classes)

    raw_vectors = raw.get("vectors")
    if not isinstance(raw_vectors, list) or not raw_vectors:
        raise BehavioralCorpusError(f"{manifest_path}: 'vectors' must be a non-empty list")

    vectors: list[CorpusVectorSpec] = []
    seen_ids: set[str] = set()
    for entry in raw_vectors:
        vector_id = entry.get("id")
        if not isinstance(vector_id, str) or not vector_id:
            raise BehavioralCorpusError(f"{manifest_path}: a vector is missing 'id'")
        if vector_id in seen_ids:
            raise BehavioralCorpusError(f"{manifest_path}: duplicate vector id {vector_id!r}")
        seen_ids.add(vector_id)

        applies_to = tuple(entry.get("applies_to") or ())
        unknown = set(applies_to) - known_classes
        if unknown:
            raise BehavioralCorpusError(
                f"{manifest_path}: vector {vector_id!r} applies_to references "
                f"unknown class(es) {sorted(unknown)!r} -- not in this edition's "
                f"declared classes {sorted(known_classes)!r}"
            )
        if not applies_to:
            raise BehavioralCorpusError(
                f"{manifest_path}: vector {vector_id!r} has empty applies_to -- "
                f"a vector that applies to nothing can never be selected"
            )

        requirements = tuple(entry.get("requirements") or ())
        unknown_reqs = set(requirements) - KNOWN_REQUIREMENTS
        if unknown_reqs:
            raise BehavioralCorpusError(
                f"{manifest_path}: vector {vector_id!r} requirements reference "
                f"unknown requirement(s) {sorted(unknown_reqs)!r} -- known: "
                f"{sorted(KNOWN_REQUIREMENTS)!r} (a genuinely new telemetry "
                f"requirement needs evaluator code, not just a manifest entry)"
            )

        prompt_file = entry.get("prompt_file")
        declared_sha256 = entry.get("prompt_sha256")
        if not isinstance(prompt_file, str) or not prompt_file:
            raise BehavioralCorpusError(f"{manifest_path}: vector {vector_id!r} missing prompt_file")
        if not isinstance(declared_sha256, str) or not declared_sha256:
            raise BehavioralCorpusError(f"{manifest_path}: vector {vector_id!r} missing prompt_sha256")
        prompt_path = fixtures_dir / prompt_file
        if not prompt_path.is_file():
            raise BehavioralCorpusError(
                f"{manifest_path}: vector {vector_id!r} prompt_file {prompt_file!r} "
                f"not found under {fixtures_dir}"
            )
        actual_sha256 = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
        if actual_sha256 != declared_sha256:
            raise BehavioralCorpusError(
                f"{manifest_path}: vector {vector_id!r} prompt_file {prompt_file!r} "
                f"content does not match its declared prompt_sha256 -- an edition's "
                f"vectors must be immutable; publish a NEW edition instead of "
                f"editing this file in place (actual={actual_sha256}, "
                f"declared={declared_sha256})"
            )

        n_predict = entry.get("n_predict")
        seed = entry.get("seed")
        scenario = entry.get("scenario")
        provenance = entry.get("provenance")
        if not isinstance(n_predict, int) or isinstance(n_predict, bool) or n_predict <= 0:
            raise BehavioralCorpusError(f"{manifest_path}: vector {vector_id!r} n_predict must be a positive integer")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise BehavioralCorpusError(f"{manifest_path}: vector {vector_id!r} seed must be an integer")
        if not isinstance(scenario, str) or not scenario:
            raise BehavioralCorpusError(f"{manifest_path}: vector {vector_id!r} missing scenario")
        if not isinstance(provenance, str) or not provenance:
            raise BehavioralCorpusError(f"{manifest_path}: vector {vector_id!r} missing provenance")

        content_digest = _canonical_digest([
            vector_id, prompt_file, declared_sha256, str(n_predict), str(seed),
            ",".join(sorted(applies_to)), ",".join(sorted(requirements)), scenario, provenance,
        ])
        vectors.append(CorpusVectorSpec(
            id=vector_id, prompt_file=prompt_file, prompt_sha256=declared_sha256,
            n_predict=n_predict, seed=seed, applies_to=applies_to,
            requirements=requirements, scenario=scenario, provenance=provenance,
            content_digest=content_digest,
        ))

    edition_digest = _canonical_digest(
        [str(schema_version), edition, ",".join(sorted(known_classes))]
        + [v.content_digest for v in vectors]
    )
    return CorpusEdition(
        schema_version=schema_version, edition=edition, classes=tuple(sorted(known_classes)),
        vectors=tuple(vectors), content_digest=edition_digest,
    )


def resolve_manifest_path(edition: str, fixtures_dir: Path) -> Path:
    """GPT review round 2 (2026-08-30): resolve an edition NAME (from
    config, e.g. RuntimeProfile.behavioral_corpus_edition) to its manifest
    file by a fixed naming convention -- ``corpus-<edition>.toml`` under
    ``fixtures_dir``. This is the actual fix for HTR03's primary goal:
    publishing a new edition means dropping a new file at this
    conventional path and pointing config at its name, never editing
    Python code (the previous version hardcoded one specific manifest
    path as a module-level constant, which GPT correctly flagged as not
    actually satisfying that goal)."""
    path = fixtures_dir / f"corpus-{edition}.toml"
    if not path.is_file():
        raise BehavioralCorpusError(
            f"corpus edition {edition!r} not found -- expected {path} to exist"
        )
    return path


def to_behavioral_vector(spec: CorpusVectorSpec, fixtures_dir: Path):
    """Build a real behavioral_gate.BehavioralVector from a manifest spec --
    a thin import-local conversion (behavioral_gate.py has no dependency on
    this module, keeping the corpus/gate layering one-directional)."""
    from . import behavioral_gate as behavioral_gate_mod
    prompt = (fixtures_dir / spec.prompt_file).read_text(encoding="utf-8")
    return behavioral_gate_mod.BehavioralVector(
        name=spec.id, prompt=prompt, n_predict=spec.n_predict, seed=spec.seed,
        requires_mtp="mtp-telemetry" in spec.requirements,
    )


def resolve_applicable_vectors(
    corpus: CorpusEdition, behavioral_classes: tuple[str, ...],
) -> tuple[CorpusVectorSpec, ...]:
    """Vectors whose applies_to intersects the given behavioral_classes.
    Fails LOUD (never a quiet "nothing to check") when behavioral_classes
    is non-empty but resolves to zero vectors -- a campaign declaring a
    real workload class it expects coverage for must not silently pass
    with no comparison ever performed."""
    unknown = set(behavioral_classes) - set(corpus.classes)
    if unknown:
        raise BehavioralCorpusError(
            f"runtime profile declares behavioral_classes {sorted(unknown)!r} "
            f"unknown to corpus edition {corpus.edition!r} (known: "
            f"{sorted(corpus.classes)!r})"
        )
    classes = set(behavioral_classes)
    applicable = tuple(v for v in corpus.vectors if classes & set(v.applies_to))
    if behavioral_classes and not applicable:
        raise BehavioralCorpusError(
            f"runtime profile behavioral_classes {sorted(behavioral_classes)!r} "
            f"matched ZERO vectors in corpus edition {corpus.edition!r} -- refusing "
            f"to silently treat 'nothing applicable' as 'nothing to check'"
        )
    return applicable
