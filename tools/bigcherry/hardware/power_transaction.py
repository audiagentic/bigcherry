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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import gpu_power, power_probe
from .power_receipt import read_json, write_json_atomic

_BDF = re.compile(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]")
_ACTIVE = ("applying", "applied", "rollback_failed", "restore_failed")
_TERMINAL = ("rolled_back", "restored")
# A prepared journal has no writes. Retain it and require a distinct run ID.
_SAFE_IDLE = ("prepared", *_TERMINAL)


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
    identity: dict


def _identity(root: Path, bdf: str, hwmon: str | None) -> dict:
    pci = root / "sys/bus/pci/devices" / bdf
    out = {k: power_probe.read_field(pci / k).value
           for k in ("vendor", "device", "subsystem_vendor", "subsystem_device",
                     "unique_id", "vbios_version", "power_dpm_force_performance_level")}
    out["default_cap_uw"] = (power_probe.read_int(root / hwmon.lstrip("/") / "power1_cap_default").value
                             if hwmon else None)
    return out


def _state_from_probe(g: power_probe.GpuCapability) -> gpu_power.State:
    c = g.controls

    def cur(name):
        v = c[name].current
        return v.value if v.status == "supported" else None

    cap_default = None
    return gpu_power.State(vddgfx_offset_mv=cur("vddgfx_offset_mv"), sclk_max_mhz=cur("sclk_max_mhz"),
                           mclk_max_mhz=cur("mclk_max_mhz"), sclk_offset_mhz=cur("sclk_offset_mhz"),
                           power_cap_uw=cur("power_cap_w"), power_cap_default_uw=cap_default)


def _hwmon_path(g: power_probe.GpuCapability) -> str | None:
    return f"{gpu_power.SYSFS_PCI}/{g.bdf}/hwmon/{g.hwmon.value}" if g.hwmon.status == "supported" else None


class PowerTransaction:
    def __init__(self, run_id: str, journal_dir: Path, *, sysfs_root: Path = Path("/"), fileops: FileOps | None = None,
                 lease_ok: Callable[[], bool], platform_name: str | None = None):
        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
            raise PowerRefused(["invalid power journal run ID"])
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
        write_json_atomic(self.journal_path, {
            "schema_version": 1, "run_id": self.run_id, "state": state, "events": self.events,
            "devices": [{"bdf": d.bdf, "hwmon": d.hwmon, "setting": vars(d.setting), "before": d.before.to_json(),
                         "identity": d.identity}
                        for d in self.devs],
            "touched": [d.bdf for d in self.touched],
        })
        self.state = state

    def _event(self, kind: str, **detail) -> None:
        self.events.append({"kind": kind, **detail})

    def _peer_journal_dir(self) -> Path | None:
        # New physical applies use the shared host work root, never a private
        # run directory. Fake-sysfs callers retain their isolated test roots.
        if self.root.resolve() == Path("/"):
            root = os.environ.get("BC_WORK_ROOT")
            if not root or not Path(root).is_absolute() or not Path(root).is_dir():
                raise PowerRefused(["physical apply requires an existing absolute BC_WORK_ROOT"])
            expected = Path(root).resolve() / "flashnext/power/journals"
            if self.journal_dir.absolute() != expected or self.journal_dir.resolve() != expected:
                raise PowerRefused(["physical apply requires the shared platform journal directory"])
            return Path(root).resolve() / "power/journals"
        # The peer's native records are not written or recovered here.
        parts = self.journal_dir.absolute().parts
        if parts[-3:] == ("flashnext", "power", "journals"):
            return self.journal_dir.absolute().parents[2] / "power/journals"
        return None

    def stale_journals(self) -> list[Path]:
        paths = [self.journal_dir]
        peer = self._peer_journal_dir()
        if peer is not None:
            paths.append(peer)
        return [p for directory in paths for p in self._stale_in(directory, native=(directory == peer))]

    @staticmethod
    def _stale_in(directory: Path, *, native: bool) -> list[Path]:
        out = []
        if directory.is_symlink():
            return [directory]
        try:
            entries = sorted(directory.iterdir())
        except FileNotFoundError:
            return out
        except OSError:
            return [directory]
        for p in entries:
            if p.suffix != ".json":
                continue
            try:
                if p.is_symlink() or not p.is_file() or p.stat().st_size > 1024 * 1024:
                    raise ValueError("journal must be a bounded regular file")
                doc = read_json(p)
                if (not isinstance(doc, dict) or doc.get("schema_version") != 1
                        or doc.get("state") not in _SAFE_IDLE or doc.get("run_id") != p.stem
                        or not isinstance(doc.get("devices"), list)
                        or not isinstance(doc.get("events"), list)
                        or not isinstance(doc.get("touched"), list)
                        or (doc["state"] == "prepared" and doc["touched"] != [])
                        or (native and not all(isinstance(doc.get(key), str) and
                                               re.fullmatch(r"sha256:[0-9a-f]{64}", doc[key])
                                               for key in ("manifest_digest", "profile_digest")))):
                    out.append(p)
            except (OSError, ValueError, TypeError):
                out.append(p)  # unreadable journal: treat as unresolved
        return out

    # ---- preflight -------------------------------------------------------------------------------------------------
    def prepare(self, requested: dict[str, gpu_power.Setting], selected_bdfs: list[str]) -> None:
        reasons: list[str] = []
        if not requested:
            raise PowerRefused(["empty explicit request: no settings for any selected device"])
        if any(not isinstance(b, str) or not _BDF.fullmatch(b) for b in requested):
            raise PowerRefused(["invalid physical BDF"])
        if any(not isinstance(s, gpu_power.Setting) or
               any(v is not None and type(v) is not int for v in vars(s).values())
               for s in requested.values()):
            raise PowerRefused(["invalid power setting"])
        extra = sorted(set(requested) - set(selected_bdfs))
        if extra:
            reasons.append(f"settings for devices outside the serving layout: {', '.join(extra)}")
        if not self.lease_ok():
            reasons.append("the gpu lease is not verified as held by this run")
        if self.journal_path.exists() or self.journal_path.is_symlink():
            reasons.append("power journal run ID already exists")
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
                                     ("mclk_max_mhz", "mclk_max_mhz", s.mclk_max_mhz),
                                     ("sclk_offset_mhz", "sclk_offset_mhz", s.sclk_offset_mhz)):
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
            if s.sclk_offset_mhz is not None and s.sclk_max_mhz is not None:
                reasons.append(f"{bdf}: mutually exclusive shader controls")
            # an OverDrive commit couples the clock and the voltage offset: both originals must be readable to restore
            if any(x is not None for x in (s.vddgfx_offset_mv, s.sclk_max_mhz, s.mclk_max_mhz, s.sclk_offset_mhz)):
                if before.vddgfx_offset_mv is None and c["vddgfx_offset_mv"].status == "supported":
                    reasons.append(f"{bdf}: original voltage offset unreadable")
                if c["sclk_max_mhz"].status == "supported" and before.sclk_max_mhz is None:
                    reasons.append(f"{bdf}: original clock unreadable")
                if c["mclk_max_mhz"].status == "supported" and before.mclk_max_mhz is None:
                    reasons.append(f"{bdf}: original memory clock unreadable")
            if s.power_cap_w is not None or before.power_cap_uw is not None:
                cap = c["power_cap_w"]
                if (type(before.power_cap_uw) is not int or before.power_cap_uw <= 0
                        or cap.minimum is None or cap.maximum is None
                        or not cap.minimum <= before.power_cap_uw <= cap.maximum):
                    reasons.append(f"{bdf}: original power cap invalid or unreadable")
            hwmon = _hwmon_path(g)
            identity = _identity(self.root, bdf, hwmon)
            if identity["vendor"] != "0x1002" or not identity["device"]:
                reasons.append(f"{bdf}: physical device identity unavailable")
            if not any(v is not None for v in vars(s).values()):
                reasons.append(f"{bdf}: empty setting")
            dev = _Dev(bdf, hwmon, s, before, identity)
            try:
                self._restore_preflight(dev)
            except (OSError, TypeError, ValueError) as exc:
                reasons.append(str(exc))
            devs.append(dev)
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
            if not self.lease_ok():
                raise OSError("GPU lease lost before control write")
            self.ops.write(path, value)

    def _verify(self, d: _Dev, want: gpu_power.State) -> list[str]:
        g = power_probe.probe_gpus([d.bdf], sysfs_root=self.root, platform_name=self.platform)[0]
        now = _state_from_probe(g)
        bad = []
        if (g.driver.value != "amdgpu" or _hwmon_path(g) != d.hwmon
                or _identity(self.root, d.bdf, d.hwmon) != d.identity):
            bad.append("device driver or hwmon identity changed")
        for field, label in (("vddgfx_offset_mv", "voltage offset"), ("sclk_max_mhz", "max clock"),
                             ("mclk_max_mhz", "max memory clock"), ("sclk_offset_mhz", "shader offset"), ("power_cap_uw", "power cap")):
            value, expected = getattr(now, field), getattr(want, field)
            if value != expected:
                bad.append(f"{label} {value} != {expected}")
        return bad

    def _target_state(self, d: _Dev) -> gpu_power.State:
        s = d.setting
        return gpu_power.State(
            vddgfx_offset_mv=s.vddgfx_offset_mv if s.vddgfx_offset_mv is not None else d.before.vddgfx_offset_mv,
            sclk_max_mhz=s.sclk_max_mhz if s.sclk_max_mhz is not None else d.before.sclk_max_mhz,
            mclk_max_mhz=s.mclk_max_mhz if s.mclk_max_mhz is not None else d.before.mclk_max_mhz,
            sclk_offset_mhz=s.sclk_offset_mhz if s.sclk_offset_mhz is not None else d.before.sclk_offset_mhz,
            power_cap_uw=s.power_cap_w * 1_000_000 if s.power_cap_w is not None else d.before.power_cap_uw)

    def apply(self) -> None:
        if self.state != "prepared":
            raise PowerFailed(f"apply from state {self.state!r}", rolled_back=True)
        if not self.lease_ok():
            raise PowerRefused(["GPU lease lost before apply"])
        if self.stale_journals():
            raise PowerRefused(["unresolved power journal appeared before apply"])
        for d in self.devs:
            bad = self._verify(d, d.before)
            if bad:
                raise PowerRefused([f"{d.bdf}: original state changed before apply: " + "; ".join(bad)])
        self._journal("applying")  # durable BEFORE the first write
        try:
            for d in self.devs:
                if not self.lease_ok():
                    raise OSError("GPU lease lost before device write")
                bad = self._verify(d, d.before)
                if bad:
                    raise OSError(f"{d.bdf}: original state changed before device write: " + "; ".join(bad))
                self._restore_preflight(d)
                self.touched.append(d)
                self._journal("applying")
                # Both OverDrive clock writes reset gfx1100 VDDGFX. Preserve the
                # measured original when no offset change was requested.
                setting = d.setting
                if d.before.sclk_offset_mhz is not None and any(x is not None for x in (
                        setting.sclk_offset_mhz, setting.vddgfx_offset_mv, setting.mclk_max_mhz)):
                    target = self._target_state(d)
                    setting = gpu_power.Setting(sclk_offset_mhz=target.sclk_offset_mhz,
                                                mclk_max_mhz=target.mclk_max_mhz,
                                                vddgfx_offset_mv=target.vddgfx_offset_mv,
                                                power_cap_w=setting.power_cap_w)
                elif (setting.sclk_max_mhz is not None or setting.mclk_max_mhz is not None) and setting.vddgfx_offset_mv is None:
                    setting = gpu_power.Setting(vddgfx_offset_mv=d.before.vddgfx_offset_mv,
                                                sclk_max_mhz=setting.sclk_max_mhz,
                                                mclk_max_mhz=setting.mclk_max_mhz,
                                                power_cap_w=setting.power_cap_w)
                self._write_all(gpu_power.apply_writes(d.bdf, "amdgpu", d.hwmon, setting))
                bad = self._verify(d, self._target_state(d))
                if bad:
                    raise OSError(f"{d.bdf}: readback mismatch: {'; '.join(bad)}")
                self._event("applied", bdf=d.bdf)
            for d in self.devs:
                bad = self._verify(d, self._target_state(d))
                if bad:
                    raise OSError(f"{d.bdf}: final apply readback mismatch: " + "; ".join(bad))
            if not self.lease_ok():
                raise OSError("GPU lease lost after apply")
            self._journal("applied")
        except Exception as exc:  # noqa: BLE001 - any failure after the first write rolls back
            self._event("apply_failed", error=str(exc))
            ok = self._restore_touched()
            self._journal("rolled_back" if ok else "rollback_failed")
            raise PowerFailed(str(exc), rolled_back=ok) from exc

    def _restore_preflight(self, d: _Dev) -> None:
        g, = power_probe.probe_gpus([d.bdf], sysfs_root=self.root, platform_name=self.platform)
        if (g.driver.value != "amdgpu" or _hwmon_path(g) != d.hwmon
                or _identity(self.root, d.bdf, d.hwmon) != d.identity):
            raise OSError(f"{d.bdf}: device identity changed before restoration")
        s, b = d.setting, d.before
        od = any(x is not None for x in (s.sclk_max_mhz, s.mclk_max_mhz, s.vddgfx_offset_mv, s.sclk_offset_mhz))
        signed = b.sclk_offset_mhz is not None
        if od:
            # Require an unambiguous shader grammar and the saved original on the live device.
            live_signed = g.controls["sclk_offset_mhz"].status == "supported"
            live_absolute = g.controls["sclk_max_mhz"].status == "supported"
            if live_signed == live_absolute or signed != live_signed or (signed and b.sclk_max_mhz is not None):
                raise OSError(f"{d.bdf}: ambiguous or changed shader control kind")
            if (signed and s.sclk_max_mhz is not None) or (not signed and s.sclk_offset_mhz is not None):
                raise OSError(f"{d.bdf}: incompatible shader control")
        for required, name, value in (
            (not signed and (s.sclk_max_mhz is not None or s.vddgfx_offset_mv is not None), "sclk_max_mhz", b.sclk_max_mhz),
            (od and signed, "sclk_offset_mhz", b.sclk_offset_mhz),
            (s.mclk_max_mhz is not None or (od and signed), "mclk_max_mhz", b.mclk_max_mhz),
            (od and (signed or b.vddgfx_offset_mv is not None or s.vddgfx_offset_mv is not None),
             "vddgfx_offset_mv", b.vddgfx_offset_mv),
            (s.power_cap_w is not None, "power_cap_w", b.power_cap_uw),
        ):
            if not required:
                continue
            c = g.controls[name]
            if (type(value) is not int or (name not in ("vddgfx_offset_mv", "sclk_offset_mhz") and value <= 0)
                    or c.status != "supported" or not c.minimum <= value <= c.maximum):
                raise OSError(f"{d.bdf}: original {name} cannot be restored safely")

    def _restore_touched(self) -> bool:
        try:
            if self.touched and not self.lease_ok():
                raise OSError("GPU lease not verified for rollback/restore")
            for d in self.touched:
                self._restore_preflight(d)
        except (OSError, TypeError, ValueError) as exc:
            self._event("restore_refused", reason=str(exc))
            return False
        ok = True
        for d in reversed(self.touched):
            try:
                self._restore_preflight(d)
                self._write_all(gpu_power.restore_writes(d.bdf, "amdgpu", d.hwmon, d.before, d.setting))
                bad = self._verify(d, d.before)
                if bad:
                    ok = False
                    self._event("restore_mismatch", bdf=d.bdf, detail=bad)
                else:
                    self._event("restored", bdf=d.bdf)
            except Exception as exc:  # noqa: BLE001
                ok = False
                self._event("restore_failed", bdf=d.bdf, error=str(exc))
        # Re-read the entire touched cohort before resolving the durable journal.
        # Another control may have drifted during a later card restoration.
        for d in self.touched:
            try:
                bad = self._verify(d, d.before)
                if bad:
                    ok = False
                    self._event("restore_mismatch", bdf=d.bdf, detail=bad)
            except Exception as exc:
                ok = False
                self._event("restore_failed", bdf=d.bdf, error=str(exc))
        if self.touched and not self.lease_ok():
            ok = False
            self._event("restore_refused", reason="GPU lease lost before final readback")
        return ok

    def restore(self, *, processes_exited: bool) -> bool:
        """Put the original values back and verify them. Refuses while the caller cannot say the GPU users are gone."""
        if self.state in _TERMINAL:
            return self.lease_ok() and processes_exited and all(not self._verify(d, d.before) for d in self.touched)
        if self.state not in _ACTIVE:
            raise PowerRefused(["no active power transaction to restore"])
        if not processes_exited or not self.lease_ok():
            self._event("restore_refused", reason="serving processes still running" if not processes_exited
                        else "GPU lease not verified")
            self._journal(self.state)
            return False
        ok = self._restore_touched()
        self._journal("restored" if ok else "restore_failed")
        return ok

    def invalidate_restoration(self, reason: str) -> None:
        """A later caller observation contradicted a resolved journal."""
        if self.touched and self.state in _TERMINAL:
            self._event("restore_failed", reason=reason)
            self._journal("restore_failed")

    @classmethod
    def recover(cls, journal_path: Path, *, sysfs_root: Path = Path("/"), fileops: FileOps | None = None,
                lease_ok: Callable[[], bool], processes_exited: bool, platform_name: str | None = None) -> bool:
        """Restore from a retained journal (after a crash or a failed restore). Needs the lease and exited processes."""
        journal_path = Path(journal_path)
        if not lease_ok() or not processes_exited:
            raise PowerRefused(["recovery requires verified lease and exited GPU processes"])
        try:
            if journal_path.is_symlink():
                raise ValueError("journal symlink")
            doc = read_json(journal_path)
            if (doc["schema_version"] != 1 or doc["run_id"] != journal_path.stem
                    or doc["state"] not in _ACTIVE or not isinstance(doc["events"], list)
                    or not isinstance(doc["devices"], list) or not isinstance(doc["touched"], list)
                    or not doc["touched"] or len(set(doc["touched"])) != len(doc["touched"])):
                raise ValueError("invalid active journal inventory")
            tx = cls(doc["run_id"], journal_path.parent, sysfs_root=sysfs_root, fileops=fileops,
                     lease_ok=lease_ok, platform_name=platform_name)
            tx.state = doc["state"]
            tx.events = list(doc["events"])
            by_bdf = {}
            for dd in doc["devices"]:
                bdf = dd["bdf"]
                if not _BDF.fullmatch(bdf) or bdf in by_bdf:
                    raise ValueError("invalid/duplicate journal BDF")
                if dd["hwmon"] is not None and not re.fullmatch(
                        re.escape(f"{gpu_power.SYSFS_PCI}/{bdf}/hwmon/") + r"hwmon[0-9]+", dd["hwmon"]):
                    raise ValueError("invalid journal hwmon")
                if set(dd["setting"]) != set(gpu_power.Setting.__dataclass_fields__) or set(dd["before"]) != set(gpu_power.State.__dataclass_fields__):
                    raise ValueError("incomplete journal controls")
                if any(v is not None and type(v) is not int for v in (*dd["setting"].values(), *dd["before"].values())):
                    raise ValueError("invalid journal control values")
                if not any(v is not None for v in dd["setting"].values()):
                    raise ValueError("empty journal setting")
                identity = dd["identity"]
                if not isinstance(identity, dict) or identity.get("vendor") != "0x1002" or not identity.get("device"):
                    raise ValueError("journal lacks physical identity; manual owner recovery required")
                by_bdf[bdf] = _Dev(bdf, dd["hwmon"], gpu_power.Setting(**dd["setting"]),
                                   gpu_power.State.from_json(dd["before"]), identity)
            tx.devs = list(by_bdf.values())
            tx.touched = [by_bdf[b] for b in doc["touched"]]
            if doc["touched"] != list(by_bdf)[:len(tx.touched)] or (doc["state"] == "applied" and len(tx.touched) != len(tx.devs)):
                raise ValueError("journal touched inventory mismatch")
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise PowerRefused([f"invalid recovery journal: {exc}"]) from exc
        return tx.restore(processes_exited=True)
