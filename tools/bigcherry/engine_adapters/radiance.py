"""radiance: no dispatch modes or build instrumentation, an unparsed attestation, and the libr3 MoE helper processes."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .base import EngineAdapter, QualityCommand, SidecarSpec, SourceCheck

if TYPE_CHECKING:
    from ..core.context import ProjectContext

HELPER_SWITCH = "R3_MOE_HELPER"
HELPER_DEVICES = "R3_MOE_HELPER_DEVICES"
HELPER_SHM = "R3_MOE_HELPER_SHM"
DEFAULT_SHM = "/r3-moe-helper"
#: what the retired fn-r9700-engine.sh waited for: the helper is up (helper 0 has created the segment) or waits for the engine
#: runtime expert maps. ("runtime map owns" is logged only after the engine has connected, so it cannot gate the
#: start; it is the activation marker, read from the helper logs afterwards.)
READY_PATTERN = r"^r3_moe_helper: (ready |waiting for engine runtime expert maps)"
READY_TIMEOUT_S = 900.0
STOP_TIMEOUT_S = 60.0
KLD_TIMEOUT_S = 5400.0


class RadianceAdapter(EngineAdapter):
    parses_execution_attestation = False

    def dispatch_modes(self) -> tuple[str, ...]:
        return ("stock",)

    def inspect_build(self, build_dir: Path) -> dict[str, Any] | None:
        return None

    def quality_command(self, kind, arm, env, run_dir, binary, model, quality) -> QualityCommand | None:
        """radiance's KL mode is offline and prefill-only: ``radiance --model M <args> --kld-record DIR --kld-corpus
        JSONL`` writes the reference, ``... --kld-ref DIR --kld-out JSON`` scores against it. ``quality["args"]`` are
        the engine arguments of the KL run (layout, placement, ``--max-num-batched-tokens`` no larger than the helper
        lane cap); the arm's settings and sidecars apply as in serving."""
        if kind not in ("record", "score"):
            raise ValueError(f"quality kind must be record or score, not {kind!r}")
        argv = [str(binary), "--model", str(model), *quality["args"]]
        result = None
        if kind == "record":
            argv += ["--kld-record", quality["reference_dir"], "--kld-corpus", quality["corpus"]]
        else:
            result = run_dir / f"{arm['name']}.kld.json"
            argv += ["--kld-ref", quality["reference_dir"], "--kld-out", str(result)]
        return QualityCommand(
            argv=tuple(argv), env=dict(env), log_path=run_dir / f"{arm['name']}.{kind}.log",
            result_path=result, timeout_s=float(quality.get("timeout_s", KLD_TIMEOUT_S)),
        )

    def source_checks(self, context: ProjectContext) -> tuple[SourceCheck, ...]:
        return ()

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
        # The helper environment is the arm environment without the engine-side switch (the helpers never set
        # R3_MOE_HELPER) and without the harness-only device list.
        base = {k: v for k, v in env.items() if k not in (HELPER_SWITCH, HELPER_DEVICES)}
        specs = []
        for index, device in enumerate(devices):
            helper_env = dict(base)
            helper_env.setdefault("RADIANCE_HOME", str(home))
            helper_env.update({
                # ROCR selects the physical card; HIP then numbers the one visible card 0.
                "ROCR_VISIBLE_DEVICES": device, "HIP_VISIBLE_DEVICES": "0",
                "R3_MOE_HELPER_INDEX": str(index), "R3_MOE_HELPER_COUNT": str(len(devices)),
            })
            specs.append(SidecarSpec(
                name=f"helper{index}", argv=(str(helper), str(arm["model"])), env=helper_env,
                ready_pattern=READY_PATTERN, ready_timeout_s=READY_TIMEOUT_S,
                log_path=run_dir / f"helper{index}.log", stop_timeout_s=STOP_TIMEOUT_S,
                pre_start_cleanup=cleanup,
            ))
        return specs
