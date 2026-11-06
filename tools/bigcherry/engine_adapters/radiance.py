"""radiance: no dispatch modes or build instrumentation, an unparsed attestation, and the libr3 MoE helper processes."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core import environment
from . import radiance_manifest
from .base import (
    BumpCheck, CompletionApi, CpuSmokeSpec, DeviceAssertion, DriftCheck, EngineAdapter, QualityCommand, SidecarSpec, SourceCheck, SMOKE_SEED,
    build_cpu_smoke,
)

if TYPE_CHECKING:
    from ..core.context import ProjectContext

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

    def manifest_port(self, manifest: Path) -> int:
        try:
            return radiance_manifest.read(manifest).port
        except radiance_manifest.ManifestError as exc:
            raise ValueError(str(exc)) from exc

    def sidecars(self, arm: dict[str, Any], env: dict[str, str], run_dir: Path, binary: Path) -> list[SidecarSpec]:
        manifest_path = arm.get("manifest")
        if manifest_path is None:
            return []
        try:
            manifest = radiance_manifest.read(Path(manifest_path))
        except radiance_manifest.ManifestError as exc:
            raise ValueError(f"{arm['name']}: {exc}") from exc
        by_locator = {d.locator: d for d in _host().devices if d.locator}
        devices = []
        for gpu in manifest.helper_gpus:
            card = by_locator.get(gpu)
            if card is None:
                raise ValueError(f"{arm['name']}: the manifest's helper card {gpu} is not in the host inventory (a device with that locator)")
            devices.append(str(card.index))
        helper = binary.parent.parent / "radiance_home" / "helpers" / "r3_moe_worker"
        # The POSIX segment /name lives at /dev/shm/name. Helper 0 creates it; SidecarSet starts in list order and
        # waits for each helper to be ready, so helper 1 attaches to an existing segment.
        cleanup = (Path("/dev/shm") / manifest.shm.lstrip("/"),)
        # A helper's whole configuration is the manifest; the environment only says which card it sees.
        base = {k: v for k, v in env.items() if not k.startswith("R3_") and k != "RADIANCE_HOME"}
        specs = []
        for index, device in enumerate(devices):
            helper_env = {**base, "ROCR_VISIBLE_DEVICES": device, "HIP_VISIBLE_DEVICES": "0"}
            specs.append(SidecarSpec(
                name=f"helper{index}",
                argv=(str(helper), "--manifest", str(manifest_path), "--role", f"helper{index}"),
                env=helper_env, ready_pattern=READY_PATTERN,
                ready_timeout_s=READY_TIMEOUT_S,
                log_path=run_dir / f"helper{index}.log", stop_timeout_s=STOP_TIMEOUT_S,
                pre_start_cleanup=cleanup,
            ))
        return specs
