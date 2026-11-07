"""Production patch-evidence gate shared by campaign builds and pin-bump.

G4 is evaluated with patch.gates.evaluate_evidence_gate, the SAME authority
used by patch-gates --intent build. No CLI subprocess or parallel verifier.
Evidence carries across pin bumps and patch edits; where a qualifying record
was taken at another digest or pin, the audit lists a provenance note.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..core import config as recipes_config
from . import catalog, gates, patchset, registry


@dataclass(frozen=True)
class EvidenceAudit:
    engine: str
    source: str
    pin: str
    mode: str
    validated: tuple[str, ...]
    failed: tuple[tuple[str, str], ...]
    notes: tuple[tuple[str, str], ...]
    skipped: tuple[tuple[str, str], ...]

    @property
    def pending_line(self) -> str:
        return (
            f"evidence gate pending for {self.engine}: {len(self.failed)} of "
            f"{len(self.validated)} validated patches do not qualify at "
            f"{self.pin} (MEN28)"
        )

    def record(self) -> dict:
        return {
            "mode": self.mode,
            "result": "pending" if self.mode == "pending-requalification" else "passed",
            "pin": self.pin,
            "validated_count": len(self.validated),
            "pending_count": len(self.failed),
            "message": self.pending_line if self.mode == "pending-requalification" else "",
            "first_five_patch_ids": [patch for patch, _ in self.failed[:5]],
            "failures": [{"patch": patch, "reason": reason} for patch, reason in self.failed],
            "notes": [
                {"patch": patch, "note": note}
                for patch, note in self.notes
            ],
            "skipped_sets": [
                {"set": name, "required_state": state}
                for name, state in self.skipped
            ],
        }


def evaluate_source(
    *, engine: str, cfg: recipes_config.Config, source_name: str,
    patches_dir: Path, pinned_ref: str, resolved_base_revision: str,
    mode: str,
) -> EvidenceAudit:
    """Evaluate G4 on each validated set member, not on exempt sets.

    A set omitted from recipes' required-state uses the validated default.
    The shared evaluator returns a non-PASS for missing or ineligible evidence.
    ``pinned_ref``/``resolved_base_revision`` only produce provenance notes.
    """
    if mode not in ("enforced", "pending-requalification"):
        raise ValueError(f"invalid evidence gate mode {mode!r}")
    source = cfg.sources[source_name]
    validated: list[str] = []
    skipped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name in source.patch_sets:
        set_cfg = cfg.patch_sets[name]
        if set_cfg.required_state != "validated":
            skipped.append((name, set_cfg.required_state))
            continue
        for patch_id in set_cfg.patches:
            if patch_id not in seen:
                validated.append(patch_id)
                seen.add(patch_id)

    if not validated:
        return EvidenceAudit(engine, source_name, pinned_ref, mode, (), (), (), tuple(skipped))

    descriptors = registry.load_registry(engine, patches_dir).by_id
    composition = patchset.resolve_exact(
        engine,
        tuple(dict.fromkeys(p for name in source.patch_sets for p in cfg.patch_sets[name].patches)),
        directory=patches_dir,
        allow_rejected=False,
    )
    failures: list[tuple[str, str]] = []
    notes: list[tuple[str, str]] = []
    for patch_id in validated:
        descriptor = descriptors[patch_id]
        context = gates.GateContext(
            engine=engine,
            descriptor=descriptor,
            composition=composition,
            pinned_ref=pinned_ref,
            intent=gates.GateIntent.BUILD,
            patch_context=catalog.PatchContext(backend=source.backend, source=source_name),
            patches_dir=patches_dir,
            resolved_base_revision=resolved_base_revision,
            catalog_path=patches_dir / "catalog.toml",
        )
        result = gates.evaluate_evidence_gate(context)
        if result.status is not gates.GateStatus.PASS:
            failures.append((patch_id, "; ".join(result.detail) or f"G4 {result.status.value}"))
        else:
            notes.extend(
                (patch_id, detail.removeprefix(gates.EVIDENCE_NOTE_PREFIX))
                for detail in result.detail if detail.startswith(gates.EVIDENCE_NOTE_PREFIX)
            )
    return EvidenceAudit(
        engine, source_name, pinned_ref, mode,
        tuple(validated), tuple(failures), tuple(notes), tuple(skipped),
    )


def report_lines(audit: EvidenceAudit) -> tuple[str, ...]:
    """Human-visible messages at the start of the build and in failures."""
    lines = [
        f"evidence gate skipped for set {name} (required-state {state})"
        for name, state in audit.skipped
    ]
    lines += [
        f"evidence note: {patch}: {note}"
        for patch, note in audit.notes
    ]
    if audit.mode == "pending-requalification":
        lines = [audit.pending_line, "first five patches: " +
                 (", ".join(p for p, _ in audit.failed[:5]) or "(none)"), *lines]
    return tuple(lines)


def failure_text(audit: EvidenceAudit) -> str:
    lines = [
        f"patch evidence G4 does not qualify at {audit.pin} for {audit.engine}:"
    ]
    for patch, reason in audit.failed:
        lines.append(f"  {patch}: {reason}")
        lines.append(
            f"    inspect: python -m bigcherry --engine {audit.engine} "
            f"patch-gates --intent build --source {audit.source} {patch}"
        )
    lines.append("re-qualify using skill bigcherry-patch-qualification (patch-qualification flow)")
    return "\n".join(lines)
