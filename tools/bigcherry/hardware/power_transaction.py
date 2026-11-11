"""PWR09: apply GPU power settings to the selected cards as one transaction.

Rules (GPT power portability review 2026-11-10, accepted): nothing is written until every requested control on every
selected card has passed preflight against a fresh probe (device-reported range, writable, original value readable) and a
recovery journal is durably on disk; devices are written in BDF order and every setting is read back exactly; any failure
rolls back everything touched, in reverse; restore happens only when the caller says the processes using the cards are
gone, and a restore that cannot be verified keeps the journal and says so. A stale unresolved journal refuses further
writes (observation-only callers are not affected). The lease check is an injected callable: this module never decides
ownership from text.

Writes go through ``FileOps`` so tests inject faults and a device model; paths are the real sysfs paths and ``FileOps``
maps them under a root.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import gpu_power, power_probe
from .power_receipt import read_json, write_json_atomic

_ACTIVE = ("applying", "applied", "rollback_failed", "restore_failed")


class PowerRefused(Exception):
    """Preflight said no; nothing was written."""

    def __init__(self, reasons: list[str]):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


class PowerFailed(Exception):
    """A write or readback failed after writing began; ``rolled_back`` says whether the cards are verified restored."""

    def __init__(self, message: str, rolled_back: bool):
        super().__init__(message)
        self.rolled_back = rolled_back


class FileOps:
    """Reads and writes sysfs files; every path is an absolute real path remapped under ``root``."""

    def __init__(self, root: Path = Path("/")):
        self.root = Path(root)

    def _p(self, path: str) -> Path:
        return self.root / path.lstrip("/")

    def write(self, path: str, value: str) -> None:
        with open(self._p(path), "w", encoding="ascii") as f:
            f.write(value + "\n")


@dataclass(frozen=True)
class _Dev:
    bdf: str
    hwmon: str | None
    setting: gpu_power.Setting
    before: gpu_power.State


def _state_from_probe(g: power_probe.GpuCapability) -> gpu_power.State:
    c = g.controls

    def cur(name):
        v = c[name].current
        return v.value if v.status == "supported" else None

    cap_default = None
    return gpu_power.State(vddgfx_offset_mv=cur("vddgfx_offset_mv"), sclk_max_mhz=cur("sclk_max_mhz"),
                           mclk_max_mhz=cur("mclk_max_mhz"),
                           power_cap_uw=cur("power_cap_w"), power_cap_default_uw=cap_default)


def _hwmon_path(g: power_probe.GpuCapability) -> str | None:
    return f"{gpu_power.SYSFS_PCI}/{g.bdf}/hwmon/{g.hwmon.value}" if g.hwmon.status == "supported" else None


class PowerTransaction:
    def __init__(self, run_id: str, journal_dir: Path, *, sysfs_root: Path = Path("/"), fileops: FileOps | None = None,
                 lease_ok: Callable[[], bool], platform_name: str | None = None):
        self.run_id = run_id
        self.journal_dir = Path(journal_dir)
        self.root = Path(sysfs_root)
        self.ops = fileops or FileOps(self.root)
        self.lease_ok = lease_ok
        self.platform = platform_name
        self.devs: list[_Dev] = []
        self.touched: list[_Dev] = []
        self.state = "new"
        self.events: list[dict] = []

    # ---- journal ---------------------------------------------------------------------------------------------------
    @property
    def journal_path(self) -> Path:
        return self.journal_dir / f"{self.run_id}.json"

    def _journal(self, state: str) -> None:
        self.state = state
        write_json_atomic(self.journal_path, {
            "schema_version": 1, "run_id": self.run_id, "state": state, "events": self.events,
            "devices": [{"bdf": d.bdf, "hwmon": d.hwmon, "setting": vars(d.setting), "before": d.before.to_json()}
                        for d in self.devs],
            "touched": [d.bdf for d in self.touched],
        })

    def _event(self, kind: str, **detail) -> None:
        self.events.append({"kind": kind, **detail})

    def stale_journals(self) -> list[Path]:
        out = []
        if self.journal_dir.is_dir():
            for p in sorted(self.journal_dir.glob("*.json")):
                try:
                    if read_json(p).get("state") in _ACTIVE and p != self.journal_path:
                        out.append(p)
                except (OSError, ValueError):
                    out.append(p)  # unreadable journal: treat as unresolved
        return out

    # ---- preflight -------------------------------------------------------------------------------------------------
    def prepare(self, requested: dict[str, gpu_power.Setting], selected_bdfs: list[str]) -> None:
        reasons: list[str] = []
        if not requested:
            raise PowerRefused(["empty explicit request: no settings for any selected device"])
        extra = sorted(set(requested) - set(selected_bdfs))
        if extra:
            reasons.append(f"settings for devices outside the serving layout: {', '.join(extra)}")
        if not self.lease_ok():
            reasons.append("the gpu lease is not verified as held by this run")
        stale = self.stale_journals()
        if stale:
            reasons.append(f"unresolved power journal(s): {', '.join(p.name for p in stale)} (recover first)")
        probes = {g.bdf: g for g in power_probe.probe_gpus(sorted(requested), sysfs_root=self.root, platform_name=self.platform)}
        devs: list[_Dev] = []
        for bdf in sorted(requested):
            s = requested[bdf]
            g = probes[bdf]
            if g.driver.value != "amdgpu":
                reasons.append(f"{bdf}: no power backend for driver {g.driver.value!r}")
                continue
            c = g.controls
            for field, name, val in (("vddgfx_offset_mv", "vddgfx_offset_mv", s.vddgfx_offset_mv),
                                     ("sclk_max_mhz", "sclk_max_mhz", s.sclk_max_mhz),
                                     ("mclk_max_mhz", "mclk_max_mhz", s.mclk_max_mhz)):
                if val is None:
                    continue
                cap = c[name]
                if cap.status != "supported":
                    reasons.append(f"{bdf}: {field} not available ({cap.status}: {cap.reason})")
                elif not (cap.minimum <= val <= cap.maximum):
                    reasons.append(f"{bdf}: {field}={val} outside the device range {cap.minimum}..{cap.maximum}")
            if s.power_cap_w is not None:
                cap = c["power_cap_w"]
                uw = s.power_cap_w * 1_000_000
                if cap.status != "supported":
                    reasons.append(f"{bdf}: power cap not available ({cap.status}: {cap.reason})")
                elif not (cap.minimum <= uw <= cap.maximum):
                    reasons.append(f"{bdf}: power_cap_w={s.power_cap_w} outside the device range "
                                   f"{cap.minimum // 1_000_000}..{cap.maximum // 1_000_000}")
            before = _state_from_probe(g)
            # an OverDrive commit couples the clock and the voltage offset: both originals must be readable to restore
            if any((s.vddgfx_offset_mv is not None, s.sclk_max_mhz is not None, s.mclk_max_mhz is not None)):
                if before.vddgfx_offset_mv is None and c["vddgfx_offset_mv"].status == "supported":
                    reasons.append(f"{bdf}: original voltage offset unreadable")
                if c["sclk_max_mhz"].status == "supported" and before.sclk_max_mhz is None:
                    reasons.append(f"{bdf}: original clock unreadable")
                if c["mclk_max_mhz"].status == "supported" and before.mclk_max_mhz is None:
                    reasons.append(f"{bdf}: original memory clock unreadable")
            if s.power_cap_w is not None and before.power_cap_uw is None:
                reasons.append(f"{bdf}: original power cap unreadable")
            devs.append(_Dev(bdf, _hwmon_path(g), s, before))
        if reasons:
            raise PowerRefused(reasons)
        self.devs = devs
        try:
            self._journal("prepared")  # also proves the journal is writable before any hardware write
        except OSError as exc:
            self.devs = []
            raise PowerRefused([f"journal cannot be persisted: {exc}"]) from exc

    # ---- apply / rollback / restore --------------------------------------------------------------------------------
    def _write_all(self, writes: list[tuple[str, str]]) -> None:
        for path, value in writes:
            self.ops.write(path, value)

    def _verify(self, d: _Dev, want: gpu_power.State, which: gpu_power.Setting) -> list[str]:
        g = power_probe.probe_gpus([d.bdf], sysfs_root=self.root, platform_name=self.platform)[0]
        now = _state_from_probe(g)
        bad = []
        if any((which.vddgfx_offset_mv is not None, which.sclk_max_mhz is not None, which.mclk_max_mhz is not None)) and now.vddgfx_offset_mv != want.vddgfx_offset_mv:
            bad.append(f"voltage offset {now.vddgfx_offset_mv} != {want.vddgfx_offset_mv}")
        if which.sclk_max_mhz is not None and now.sclk_max_mhz != want.sclk_max_mhz:
            bad.append(f"max clock {now.sclk_max_mhz} != {want.sclk_max_mhz}")
        if which.mclk_max_mhz is not None and now.mclk_max_mhz != want.mclk_max_mhz:
            bad.append(f"max memory clock {now.mclk_max_mhz} != {want.mclk_max_mhz}")
        if which.power_cap_w is not None and now.power_cap_uw != want.power_cap_uw:
            bad.append(f"power cap {now.power_cap_uw} != {want.power_cap_uw}")
        return bad

    def _target_state(self, d: _Dev) -> gpu_power.State:
        s = d.setting
        return gpu_power.State(
            vddgfx_offset_mv=s.vddgfx_offset_mv if s.vddgfx_offset_mv is not None else d.before.vddgfx_offset_mv,
            sclk_max_mhz=s.sclk_max_mhz if s.sclk_max_mhz is not None else d.before.sclk_max_mhz,
            mclk_max_mhz=s.mclk_max_mhz if s.mclk_max_mhz is not None else d.before.mclk_max_mhz,
            power_cap_uw=s.power_cap_w * 1_000_000 if s.power_cap_w is not None else d.before.power_cap_uw)

    def apply(self) -> None:
        if self.state != "prepared":
            raise PowerFailed(f"apply from state {self.state!r}", rolled_back=True)
        self._journal("applying")  # durable BEFORE the first write
        try:
            for d in self.devs:
                self.touched.append(d)
                self._journal("applying")
                # Both OverDrive clock writes reset gfx1100 VDDGFX. Preserve the
                # measured original when no offset change was requested.
                setting = d.setting
                if (setting.sclk_max_mhz is not None or setting.mclk_max_mhz is not None) and setting.vddgfx_offset_mv is None:
                    setting = gpu_power.Setting(vddgfx_offset_mv=d.before.vddgfx_offset_mv,
                                                sclk_max_mhz=setting.sclk_max_mhz,
                                                mclk_max_mhz=setting.mclk_max_mhz,
                                                power_cap_w=setting.power_cap_w)
                self._write_all(gpu_power.apply_writes(d.bdf, "amdgpu", d.hwmon, setting))
                bad = self._verify(d, self._target_state(d), setting)
                if bad:
                    raise OSError(f"{d.bdf}: readback mismatch: {'; '.join(bad)}")
                self._event("applied", bdf=d.bdf)
        except Exception as exc:  # noqa: BLE001 - any failure after the first write rolls back
            self._event("apply_failed", error=str(exc))
            ok = self._restore_touched()
            self._journal("rolled_back" if ok else "rollback_failed")
            raise PowerFailed(str(exc), rolled_back=ok) from exc
        self._journal("applied")

    def _restore_touched(self) -> bool:
        ok = True
        for d in reversed(self.touched):
            try:
                self._write_all(gpu_power.restore_writes(d.bdf, "amdgpu", d.hwmon, d.before, d.setting))
                bad = self._verify(d, d.before, d.setting)
                if bad:
                    ok = False
                    self._event("restore_mismatch", bdf=d.bdf, detail=bad)
                else:
                    self._event("restored", bdf=d.bdf)
            except Exception as exc:  # noqa: BLE001
                ok = False
                self._event("restore_failed", bdf=d.bdf, error=str(exc))
        return ok

    def restore(self, *, processes_exited: bool) -> bool:
        """Put the original values back and verify them. Refuses while the caller cannot say the GPU users are gone."""
        if not processes_exited:
            self._event("restore_refused", reason="serving processes still running")
            self._journal(self.state)
            return False
        ok = self._restore_touched()
        self._journal("restored" if ok else "restore_failed")
        return ok

    @classmethod
    def recover(cls, journal_path: Path, *, sysfs_root: Path = Path("/"), fileops: FileOps | None = None,
                lease_ok: Callable[[], bool], processes_exited: bool, platform_name: str | None = None) -> bool:
        """Restore from a retained journal (after a crash or a failed restore). Needs the lease and exited processes."""
        doc = read_json(journal_path)
        tx = cls(doc["run_id"], Path(journal_path).parent, sysfs_root=sysfs_root, fileops=fileops, lease_ok=lease_ok,
                 platform_name=platform_name)
        if not lease_ok():
            raise PowerRefused(["the gpu lease is not verified as held by this run"])
        tx.events = list(doc.get("events", []))
        by_bdf = {}
        for dd in doc["devices"]:
            by_bdf[dd["bdf"]] = _Dev(dd["bdf"], dd["hwmon"], gpu_power.Setting(**dd["setting"]),
                                     gpu_power.State.from_json(dd["before"]))
        tx.devs = list(by_bdf.values())
        tx.touched = [by_bdf[b] for b in doc.get("touched", []) if b in by_bdf]
        return tx.restore(processes_exited=processes_exited)
