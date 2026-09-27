"""Pure production-coexistence model for timed scientific execution.

This module intentionally performs no HTTP/process inspection. RCD11 host
adapters build a ProductionSnapshot from llama-swap config/runtime/process
facts; these functions make the safety decision deterministically and are
fully mockable without GPUs.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping


class ProductionClaimError(ValueError):
    pass


@dataclass(frozen=True)
class ProductionClaim:
    all_devices: bool = False
    stable_device_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProductionSnapshot:
    config_hash: str
    potential_all: bool
    potential_devices: tuple[str, ...]
    running_devices: tuple[str, ...]
    observed_devices: tuple[str, ...]
    ambiguities: tuple[str, ...] = ()

    @property
    def fingerprint(self) -> str:
        payload = "|".join(
            (
                self.config_hash,
                "all" if self.potential_all else ",".join(self.potential_devices),
                ",".join(self.running_devices),
                ",".join(self.observed_devices),
                ",".join(self.ambiguities),
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CoexistenceMode(str, Enum):
    EXCLUSIVE_WINDOW = "exclusive-window"
    IDLE_ATTESTATION = "idle-attestation"


@dataclass(frozen=True)
class CoexistenceDecision:
    mode: CoexistenceMode
    conflicting_devices: tuple[str, ...]
    reason: str


def parse_gpu_claim(
    raw: str | None,
    *,
    architecture_devices: Mapping[str, Iterable[str]] | None = None,
) -> ProductionClaim:
    """Parse the narrow declarative production GPU claim language.

    Supported forms:
      all
      uuid:<id>[,uuid:<id>...]
      arch:<gfx>,count=<n>

    Architecture/count claims are deliberately conservative: every accepted
    device of that architecture is potential, because which exact ``count``
    cards llama-swap will choose may change between loads. Unknown/malformed
    claims raise; callers must convert that to fail-closed ALL potential.
    """
    if raw is None or not raw.strip():
        raise ProductionClaimError("production GPU claim is missing")
    text = raw.strip()
    if text == "all":
        return ProductionClaim(all_devices=True)
    parts = tuple(part.strip() for part in text.split(",") if part.strip())
    if parts and all(part.startswith("uuid:") for part in parts):
        ids = tuple(sorted({part[5:] for part in parts if part[5:]}))
        if len(ids) != len(parts):
            raise ProductionClaimError("UUID claim contains empty or duplicate IDs")
        return ProductionClaim(stable_device_ids=ids)
    if text.startswith("arch:"):
        fields = text.split(",")
        arch = fields[0][5:].strip()
        count = None
        for field in fields[1:]:
            if field.startswith("count="):
                try:
                    count = int(field[6:])
                except ValueError as exc:
                    raise ProductionClaimError("invalid architecture claim count") from exc
            else:
                raise ProductionClaimError(f"unknown architecture claim field: {field}")
        if not arch or count is None or count < 1:
            raise ProductionClaimError("architecture claim requires arch and positive count")
        if architecture_devices is None or arch not in architecture_devices:
            raise ProductionClaimError(f"no accepted inventory for production architecture {arch}")
        candidates = tuple(sorted(set(str(item) for item in architecture_devices[arch])))
        if len(candidates) < count:
            raise ProductionClaimError(
                f"production claim requests {count} {arch} GPUs but only {len(candidates)} accepted"
            )
        # Potential is all candidates, not an arbitrary selected subset.
        return ProductionClaim(stable_device_ids=candidates)
    raise ProductionClaimError(f"unsupported production GPU claim: {raw!r}")


def build_snapshot(
    *,
    config_hash: str,
    claims: Iterable[str | None],
    architecture_devices: Mapping[str, Iterable[str]],
    running_devices: Iterable[str] = (),
    observed_devices: Iterable[str] = (),
) -> ProductionSnapshot:
    potential: set[str] = set()
    potential_all = False
    ambiguities: list[str] = []
    for index, raw in enumerate(claims):
        try:
            claim = parse_gpu_claim(raw, architecture_devices=architecture_devices)
        except ProductionClaimError as exc:
            potential_all = True
            ambiguities.append(f"claim[{index}]: {exc}")
            continue
        potential_all = potential_all or claim.all_devices
        potential.update(claim.stable_device_ids)
    return ProductionSnapshot(
        config_hash=config_hash,
        potential_all=potential_all,
        potential_devices=tuple(sorted(potential)),
        running_devices=tuple(sorted(set(str(item) for item in running_devices))),
        observed_devices=tuple(sorted(set(str(item) for item in observed_devices))),
        ambiguities=tuple(ambiguities),
    )


def decide_coexistence(
    target_stable_ids: Iterable[str], snapshot: ProductionSnapshot
) -> CoexistenceDecision:
    target = tuple(sorted(set(str(item) for item in target_stable_ids)))
    if not target:
        raise ValueError("target stable device set is empty")
    if snapshot.potential_all or snapshot.ambiguities:
        return CoexistenceDecision(
            CoexistenceMode.EXCLUSIVE_WINDOW,
            target,
            "production device ownership is ambiguous/all-devices",
        )
    conflict = tuple(sorted(set(target) & set(snapshot.potential_devices)))
    if conflict:
        return CoexistenceDecision(
            CoexistenceMode.EXCLUSIVE_WINDOW,
            conflict,
            "target intersects configured production potential",
        )
    return CoexistenceDecision(
        CoexistenceMode.IDLE_ATTESTATION,
        (),
        "target is disjoint from configured production potential",
    )


def snapshot_contaminated(
    baseline: ProductionSnapshot,
    current: ProductionSnapshot,
    *,
    target_stable_ids: Iterable[str],
) -> tuple[bool, str | None]:
    """Continuous timed-run guard.

    Any config hash/ambiguity/potential change is contamination. A newly
    running/observed process on a target card is contamination even when the
    declarative claim did not change.
    """
    target = set(str(item) for item in target_stable_ids)
    if current.config_hash != baseline.config_hash:
        return True, "production configuration changed"
    if current.potential_all != baseline.potential_all:
        return True, "production potential changed"
    if current.potential_devices != baseline.potential_devices:
        return True, "production device claim changed"
    if current.ambiguities:
        return True, "production ownership became ambiguous"
    if target & set(current.running_devices):
        return True, "production backend is running on a target GPU"
    if target & set(current.observed_devices):
        return True, "production process/VRAM usage observed on a target GPU"
    return False, None
