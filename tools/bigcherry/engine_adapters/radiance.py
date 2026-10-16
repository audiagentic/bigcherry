"""radiance: no dispatch modes or build instrumentation, an unparsed attestation, and the libr3 MoE helper processes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import EngineAdapter, SidecarSpec

HELPER_SWITCH = "R3_MOE_HELPER"
HELPER_DEVICES = "R3_MOE_HELPER_DEVICES"
HELPER_SHM = "R3_MOE_HELPER_SHM"
DEFAULT_SHM = "/r3-moe-helper"
#: what fn-r9700-engine.sh waits for: the helper is up (helper 0 has created the segment) or waits for the engine
#: runtime expert maps. ("runtime map owns" is logged only after the engine has connected, so it cannot gate the
#: start; it is the activation marker, read from the helper logs afterwards.)
READY_PATTERN = r"^r3_moe_helper: (ready |waiting for engine runtime expert maps)"
READY_TIMEOUT_S = 900.0
STOP_TIMEOUT_S = 60.0


class RadianceAdapter(EngineAdapter):
    parses_execution_attestation = False

    def dispatch_modes(self) -> tuple[str, ...]:
        return ("stock",)

    def inspect_build(self, build_dir: Path) -> dict[str, Any] | None:
        return None

    def check_arm_environment(self, controls: dict[str, str]) -> None:
        return None

    def sidecars(self, arm: dict[str, Any], env: dict[str, str], run_dir: Path, binary: Path) -> list[SidecarSpec]:
        if not env.get(HELPER_SWITCH):
            return []
        devices = env.get(HELPER_DEVICES, "").split()
        if not devices:
            raise ValueError(f"{arm['name']}: {HELPER_SWITCH} is set; {HELPER_DEVICES} must list the helper device indices")
        if len(set(devices)) != len(devices) or not all(d.isdigit() for d in devices):
            raise ValueError(f"{arm['name']}: {HELPER_DEVICES} must be distinct device indices, got {devices}")
        home = binary.parent.parent / "radiance_home"
        helper = home / "helpers" / "r3_moe_helper"
        shm = env.get(HELPER_SHM, DEFAULT_SHM)
        # The POSIX segment /name lives at /dev/shm/name. Helper 0 creates it; SidecarSet starts in list order and
        # waits for each helper to be ready, so helper 1 attaches to an existing segment.
        cleanup = (Path("/dev/shm") / shm.lstrip("/"),)
        # The helper environment is the arm environment without the engine-side switch (fn-r9700-engine.sh does
        # not set R3_MOE_HELPER on the helpers) and without the harness-only device list.
        base = {k: v for k, v in env.items() if k not in (HELPER_SWITCH, HELPER_DEVICES)}
        specs = []
        for index, device in enumerate(devices):
            helper_env = dict(base)
            helper_env.setdefault("RADIANCE_HOME", str(home))
            helper_env.update({
                "HIP_VISIBLE_DEVICES": device, "ROCR_VISIBLE_DEVICES": device,
                "R3_MOE_HELPER_INDEX": str(index), "R3_MOE_HELPER_COUNT": str(len(devices)),
            })
            specs.append(SidecarSpec(
                name=f"helper{index}", argv=(str(helper), str(arm["model"])), env=helper_env,
                ready_pattern=READY_PATTERN, ready_timeout_s=READY_TIMEOUT_S,
                log_path=run_dir / f"helper{index}.log", stop_timeout_s=STOP_TIMEOUT_S,
                pre_start_cleanup=cleanup,
            ))
        return specs
