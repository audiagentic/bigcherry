"""radiance: no dispatch modes or build instrumentation, an unparsed attestation, and the libr3 MoE helper processes."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core import environment
from . import radiance_layout
from .base import (
    BumpCheck, CompletionApi, CpuSmokeSpec, DeviceAssertion, DriftCheck, EngineAdapter, QualityCommand, SidecarSpec, SourceCheck, SMOKE_SEED,
    build_cpu_smoke,
)

if TYPE_CHECKING:
    from ..core.context import ProjectContext

HELPER_SWITCH = "R3_MOE_WORKER"
HELPER_DEVICES = "R3_MOE_WORKER_DEVICES"
#: harness-only: a named layout (engines/radiance/layouts.toml) in place of an explicit device list; it fixes the helper cards, their budgets and placement
HELPER_LAYOUT = "R3_MOE_WORKER_LAYOUT"
HELPER_SHM = "R3_MOE_WORKER_SHM"
DEFAULT_SHM = "/r3-moe-helper"
#: what the retired fn-r9700-engine.sh waited for: the helper is up (helper 0 has created the segment) or waits for the engine
#: runtime expert maps. ("runtime map owns" is logged only after the engine has connected, so it cannot gate the
#: start; it is the activation marker, read from the helper logs afterwards.)
READY_PATTERN = r"^r3_moe_worker: (ready |waiting for engine runtime expert maps)"
READY_TIMEOUT_S = 900.0
STOP_TIMEOUT_S = 60.0
KLD_TIMEOUT_S = 5400.0


def _host() -> environment.Host:
    """The host whose device inventory says which architecture each card is (a seam for tests)."""
    return environment.load_default().host(None)


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

    def cpu_smoke(self, context: ProjectContext, build_dir: Path, model_root: Path) -> CpuSmokeSpec | None:
        """radiance on its CPU kernel plugins only: ``--kernels libavx,libref`` (libavx is the CPU host backend, libref
        the reference implementation; tools/lab/radiance/README.md) and ``--tp 1``, with the emptied GPU variables so
        no device plugin can open a card. Served through the OpenAI route ``/v1/completions`` (the one engine-bench
        drives); the served model id comes from ``/v1/models``."""
        return build_cpu_smoke(
            context, build_dir, model_root,
            engine_args=("--kernels", "libavx,libref", "--tp", "1"), extra_env={}, env_unset_prefixes=("RADIANCE_", "R3_", "BIGCHERRY_"),
            completion=CompletionApi(
                path="/v1/completions", prompt_field="prompt", token_field="max_tokens",
                fixed=(("temperature", 0), ("seed", SMOKE_SEED), ("stream", False)),
                text_path=("choices", 0, "text"), models_path="/v1/models",
            ),
            # radiance names its backend once at start-up: "I device: host backend, 1 device(s) [0:host]" on the CPU,
            # "I device: AQL backend, N device(s) [0:gfx1100 ...]" on GPUs (radiance 1.7.0, dev 2026-10-17). Card
            # names alone are not evidence: 1.7.0 also warns that it serves on the CPU while listing the cards present.
            device=DeviceAssertion((
                r"(?m)^I device: (?!host backend\b)\S+ backend\b",
                r"(?m)^I device: .*\[0:gfx\d",
            )),
        )

    def source_checks(self, context: ProjectContext) -> tuple[SourceCheck, ...]:
        return ()

    def compile_checks(
        self, context: ProjectContext, source_dir: Path, workdir: Path, *, platform_name: str | None,
    ) -> tuple[BumpCheck, ...]:
        """The component gate: build the engine and its components, check every plugin's ABI, run the component tests."""
        from . import radiance_bump

        return radiance_bump.compile_checks(context, source_dir, workdir, platform_name=platform_name)

    def source_drift_checks(self, context: ProjectContext, source_dir: Path) -> tuple[DriftCheck, ...]:
        """libr3's native rewrites, run against the bumped libr4d sources."""
        from . import radiance_bump

        return radiance_bump.source_drift_checks(context, source_dir)

    def audit_checks(self, context: ProjectContext):
        from . import radiance_bump

        return radiance_bump.audit_checks(context)

    def check_arm_environment(self, controls: dict[str, str]) -> None:
        return None

    def sidecars(self, arm: dict[str, Any], env: dict[str, str], run_dir: Path, binary: Path) -> list[SidecarSpec]:
        if not env.get(HELPER_SWITCH):
            return []
        layout, listed = env.get(HELPER_LAYOUT, ""), env.get(HELPER_DEVICES, "")
        if layout and listed:
            raise ValueError(f"{arm['name']}: give {HELPER_LAYOUT} or {HELPER_DEVICES}, not both")
        host = _host()
        try:
            if layout:
                resolved = radiance_layout.resolve(layout, host)
                devices = [str(h.index) for h in resolved.helpers]
                derived = [{**resolved.common_env, **resolved.helper_env(i)} for i in range(len(devices))]
            else:
                devices = listed.split()
                if not devices:
                    raise ValueError(f"{arm['name']}: {HELPER_SWITCH} is set; {HELPER_DEVICES} must list the helper device indices (or {HELPER_LAYOUT} name a layout)")
                if len(set(devices)) != len(devices) or not all(d.isdigit() for d in devices):
                    raise ValueError(f"{arm['name']}: {HELPER_DEVICES} must be distinct device indices, got {devices}")
                archs, by_index = radiance_layout.load_archs(), {d.index: d for d in host.devices}
                derived = []
                for d in devices:
                    card = by_index.get(int(d))
                    if card is None or card.arch not in archs:
                        raise ValueError(f"{arm['name']}: device {d} is "
                                         f"{'not in the host inventory' if card is None else card.arch + ', which engines/radiance/archs.toml does not describe'}")
                    derived.append(radiance_layout.card_env(archs[card.arch]))
        except radiance_layout.LayoutError as exc:
            raise ValueError(f"{arm['name']}: {exc}") from exc
        home = binary.parent.parent / "radiance_home"
        helper = home / "helpers" / "r3_moe_worker"
        shm = env.get(HELPER_SHM, DEFAULT_SHM)
        # The POSIX segment /name lives at /dev/shm/name. Helper 0 creates it; SidecarSet starts in list order and
        # waits for each helper to be ready, so helper 1 attaches to an existing segment.
        cleanup = (Path("/dev/shm") / shm.lstrip("/"),)
        # The helper environment is the arm environment without the engine-side switch (the helpers never set
        # R3_MOE_WORKER) and without the harness-only device list.
        base = {k: v for k, v in env.items() if k not in (HELPER_SWITCH, HELPER_DEVICES, HELPER_LAYOUT)}
        specs = []
        for index, device in enumerate(devices):
            helper_env = dict(base)
            # what the card type or layout implies is a default: a setting the arm names itself wins
            for key, value in derived[index].items():
                helper_env.setdefault(key, value)
            helper_env.setdefault("RADIANCE_HOME", str(home))
            helper_env.update({
                # ROCR selects the physical card; HIP then numbers the one visible card 0.
                "ROCR_VISIBLE_DEVICES": device, "HIP_VISIBLE_DEVICES": "0",
                "R3_MOE_WORKER_INDEX": str(index), "R3_MOE_WORKER_COUNT": str(len(devices)),
            })
            specs.append(SidecarSpec(
                name=f"helper{index}", argv=(str(helper), str(arm["model"])), env=helper_env,
                ready_pattern=READY_PATTERN, ready_timeout_s=READY_TIMEOUT_S,
                log_path=run_dir / f"helper{index}.log", stop_timeout_s=STOP_TIMEOUT_S,
                pre_start_cleanup=cleanup,
            ))
        return specs
