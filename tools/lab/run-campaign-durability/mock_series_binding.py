"""Falsify hardware-cohort binding rules for RCD04/RCD12 planning.

Lab-only. A series binds concrete stable device IDs before any session is
submitted; Slurm allocation may contain a superset, never a substitute card.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Device:
    device_id: str
    arch: str
    model: str
    vram_gib: int


@dataclass(frozen=True)
class Requirement:
    arch: str
    count: int = 1
    min_vram_gib: int = 0
    exact_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class CohortPlan:
    selected_ids: tuple[str, ...]
    reserve_count: int


def bind_series_cohort(req: Requirement, devices: tuple[Device, ...]) -> CohortPlan:
    by_id = {d.device_id: d for d in devices}
    arch_devices = tuple(sorted((d for d in devices if d.arch == req.arch), key=lambda d: d.device_id))
    if req.exact_ids:
        try:
            selected = tuple(by_id[item] for item in req.exact_ids)
        except KeyError as exc:
            raise ValueError("missing exact device") from exc
    else:
        selected = tuple(d for d in arch_devices if d.vram_gib >= req.min_vram_gib)[: req.count]
    if len(selected) != req.count:
        raise ValueError("insufficient cohort")
    if any(d.arch != req.arch or d.vram_gib < req.min_vram_gib for d in selected):
        raise ValueError("cohort violates requirement")
    # Architecture-typed Slurm cannot demand one stable ID from an otherwise
    # interchangeable type. Reserve the whole arch pool whenever the pinned
    # cohort is a strict subset, then narrow visibility to the selected IDs.
    reserve_count = len(arch_devices) if len(arch_devices) > len(selected) else len(selected)
    return CohortPlan(tuple(d.device_id for d in selected), reserve_count)


def attest_allocation(plan: CohortPlan, allocated_ids: tuple[str, ...]) -> None:
    if not set(plan.selected_ids) <= set(allocated_ids):
        raise ValueError("allocation does not contain pinned cohort")


def self_test() -> None:
    devices = (
        Device("A", "gfx1100", "XTX", 24),
        Device("B", "gfx1100", "XTX", 24),
        Device("C", "gfx1201", "R9700", 32),
    )
    one = bind_series_cohort(Requirement("gfx1100"), devices)
    assert one.selected_ids == ("A",)
    assert one.reserve_count == 2
    assert bind_series_cohort(Requirement("gfx1100", exact_ids=("B",)), devices).selected_ids == ("B",)
    dual = bind_series_cohort(Requirement("gfx1100", count=2), devices)
    assert dual.selected_ids == ("A", "B") and dual.reserve_count == 2
    attest_allocation(one, ("A", "B"))
    try:
        attest_allocation(one, ("B",))
    except ValueError:
        pass
    else:
        raise AssertionError("substitute card was accepted")
    replacement = (Device("X", "gfx1100", "XTX", 24), devices[1], devices[2])
    replacement_plan = bind_series_cohort(Requirement("gfx1100"), replacement)
    assert replacement_plan.selected_ids != one.selected_ids
    try:
        bind_series_cohort(Requirement("gfx1201", min_vram_gib=64), devices)
    except ValueError:
        pass
    else:
        raise AssertionError("impossible VRAM requirement was accepted")


def main() -> int:
    self_test()
    print(json.dumps({"ok": True, "checks": 8}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
