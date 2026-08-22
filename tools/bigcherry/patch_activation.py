"""HI82: activation-evidence gate for patch-validation campaigns.

A green build plus a plausible benchmark is not proof a patch's new code
path actually executed -- confirmed this session by 1221_rd50_gdn_chunked_
recurrence (arch-gated to RDNA3.5; our gfx1100/gfx1201 hardware would
silently fall through to the unchanged existing path and still produce a
normal-looking flat bench "pass"). This module makes "did the patch's own
code path execute" a first-class, explicit gate alongside build/correctness/
bench, rather than an assumption a green pipeline implicitly makes.

Design: GPT (gpt-auto-agent, request req_51838ef1ea5f4086 design + implemented
in req_2c85f337296e4c30), applied per plan item HI82.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

ActivationStatus = Literal[
    "executed",
    "not_executed",
    "not_applicable",
    "unobservable",
]


@dataclass(frozen=True)
class ActivationEvidence:
    status: ActivationStatus
    mechanism: str
    detail: str


def verdict(
    activation: ActivationEvidence,
    *,
    correctness_passed: bool | None,
) -> str:
    """Collapse activation + correctness evidence into the campaign disposition."""
    if activation.status == "not_applicable":
        return "gate-verified-blocked"

    if activation.status == "unobservable":
        return "unobservable"

    if activation.status == "executed":
        if correctness_passed is False:
            return "failed-correctness"

        if correctness_passed is True or correctness_passed is None:
            return "validated"

    return "failed-activation"


def write_activation_json(
    path: Path,
    activation: ActivationEvidence,
    verdict_str: str,
    *,
    extra: dict | None = None,
) -> None:
    """Atomically write activation evidence and its resulting disposition."""
    payload: dict[str, Any] = {
        "activation": asdict(activation),
        "verdict": verdict_str,
    }

    if extra is not None:
        reserved = {"activation", "verdict"}.intersection(extra)
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ValueError(f"extra may not overwrite reserved keys: {names}")
        payload.update(extra)

    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True,
    )
    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def eligibility_gate_evidence(
    *,
    predicate_name: str,
    our_archs: list[str],
    predicate_results: dict[str, bool],
    required_true_for: list[str],
) -> ActivationEvidence:
    """Build fail-closed evidence that an unsafe patch is gated off on our GPUs.

    `our_archs` are the architectures available for validation. Every one must
    evaluate the eligibility predicate to False, otherwise the patch is
    applicable somewhere we can test and this helper must not classify it as
    not_applicable.

    `required_true_for` acts as a positive-control set for the gate. When
    supplied, every listed architecture must be present in predicate_results
    and evaluate True; otherwise the gate itself has not been demonstrated.
    """
    if not predicate_name:
        raise ValueError("predicate_name must not be empty")

    relevant_archs = list(dict.fromkeys([*our_archs, *required_true_for]))

    missing = [arch for arch in relevant_archs if arch not in predicate_results]
    if missing:
        raise ValueError(
            f"missing {predicate_name} result(s) for architecture(s): "
            + ", ".join(missing)
        )

    non_bool = [
        arch for arch in relevant_archs
        if not isinstance(predicate_results[arch], bool)
    ]
    if non_bool:
        raise TypeError(
            f"{predicate_name} results must be bool for architecture(s): "
            + ", ".join(non_bool)
        )

    our_true = [arch for arch in our_archs if predicate_results[arch]]
    our_false = [arch for arch in our_archs if not predicate_results[arch]]

    # If the patch can activate on hardware we actually have, it is not a
    # "not applicable here" case. The caller needs real activation evidence
    # instead of using this helper to manufacture a blocked disposition.
    if our_true:
        raise ValueError(
            f"cannot classify {predicate_name} as not_applicable: "
            "the predicate is true on available architecture(s): "
            + ", ".join(our_true)
        )

    required_false = [
        arch for arch in required_true_for if not predicate_results[arch]
    ]
    # required_true_for is the positive control proving that the gate is not
    # merely always-false. If one of those controls is false, the eligibility
    # mechanism itself has not been demonstrated well enough to claim
    # gate-verified-blocked.
    if required_false:
        raise ValueError(
            f"eligibility gate {predicate_name} failed its required "
            "positive control(s): " + ", ".join(required_false)
        )

    all_true = sorted(arch for arch, result in predicate_results.items() if result)
    all_false = sorted(arch for arch, result in predicate_results.items() if not result)

    detail = (
        f"Eligibility predicate {predicate_name!r} was false for all "
        f"available architectures {our_false!r}, so the patch cannot "
        "activate on the hardware used by this validation. "
        f"Required true architecture(s): {required_true_for!r}. "
        f"Supplied predicate results: true={all_true!r}, false={all_false!r}."
    )

    return ActivationEvidence(
        status="not_applicable",
        mechanism=f"eligibility-gate:{predicate_name}",
        detail=detail,
    )
