"""radiance: no dispatch modes or build instrumentation, an unparsed attestation, and the libr3 MoE helper processes."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core import environment
from . import radiance_inventory
from .base import (
    SMOKE_SEED,
    BumpCheck,
    CompletionApi,
    CpuSmokeSpec,
    DeviceAssertion,
    DriftCheck,
    EngineAdapter,
    QualityCommand,
    ServingLayout,
    SidecarSpec,
    SourceCheck,
    build_cpu_smoke,
)

if TYPE_CHECKING:
    from ..core.context import ProjectContext

KLD_TIMEOUT_S = 5400.0


def _host() -> environment.Host:
    """Host scalar paths and BDF-scoped power overrides; never device selection."""
    return environment.load_default().host(None)


class RadianceAdapter(EngineAdapter):
    parses_execution_attestation = False

    def dispatch_modes(self) -> tuple[str, ...]:
        return ("stock",)

    def inspect_build(self, build_dir: Path) -> dict[str, Any] | None:
        return None

    def quality_command(self, kind, arm, env, run_dir, binary, model, quality) -> QualityCommand | None:
        """radiance's KL mode is offline and prefill-only and runs through the native owner: ``bc-serve quality`` reads and
        seals the arm's own YAML (``serving`` = ``config`` + ``native_config``), plans it and either records a reference
        (``--kld-record DIR --kld-corpus JSONL``) or scores against one (``--kld-ref DIR --kld-out JSON``). The engine is
        never started from flags: an arm without a native YAML has no quality run."""
        if kind not in ("record", "score"):
            raise ValueError(f"quality kind must be record or score, not {kind!r}")
        serving = arm.get("serving") or {}
        if "native_config" not in serving:
            raise ValueError(f"{arm['name']}: radiance quality runs a native arm (serving with config and native_config)")
        if quality["args"]:
            raise ValueError(f"{arm['name']}: a native arm takes its engine configuration from its YAML, so quality.args must be empty")
        if kind == "record" and not quality.get("corpus"):
            raise ValueError(f"{arm['name']}: recording a reference needs quality.corpus")
        host = _host()
        root = os.environ.get("BC_MODEL_ROOT") or host.model_root
        if not root:
            raise ValueError(f"{arm['name']}: no model root (BC_MODEL_ROOT or host model-root)")
        bc_serve = binary.parent.parent / "radiance_home" / "bin" / "bc-serve"
        argv = [str(bc_serve), "quality", serving["config"], "--native-config", serving["native_config"],
                "--model-root", str(root), "--engine-binary", str(binary), "--run-dir", str(run_dir / f"{arm['name']}.owner")]
        result = None
        if kind == "record":
            argv += ["--kld-record", str(quality["reference_dir"]), "--kld-corpus", str(quality["corpus"])]
        else:
            result = run_dir / f"{arm['name']}.kld.json"
            argv += ["--kld-ref", str(quality["reference_dir"]), "--kld-out", str(result)]
        return QualityCommand(
            argv=tuple(argv), env=dict(env), log_path=run_dir / f"{arm['name']}.{kind}.log", result_path=result,
            timeout_s=float(quality.get("timeout_s", KLD_TIMEOUT_S)))

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
        canonical_build: Callable[[], Path] | None = None,
    ) -> tuple[BumpCheck, ...]:
        """The component gate: build the engine and its components, check every plugin's ABI, run the component tests."""
        from . import radiance_bump

        return radiance_bump.compile_checks(context, source_dir, workdir, platform_name=platform_name,
                                            canonical_build=canonical_build)

    def source_drift_checks(self, context: ProjectContext, source_dir: Path) -> tuple[DriftCheck, ...]:
        """libr3's native rewrites, run against the bumped libr4d sources."""
        from . import radiance_bump

        return radiance_bump.source_drift_checks(context, source_dir)

    def audit_checks(self, context: ProjectContext):
        from . import radiance_bump

        return radiance_bump.audit_checks(context)

    def check_arm_environment(self, controls: dict[str, str]) -> None:
        return None

    def serving_layout(self, config: str, native_config: Path, binary: Path) -> ServingLayout:
        """The model and engine cards of a native arm, as the native owner's offline admission (``bc-serve explain``) reads
        them from the arm's own YAML; card ordinals come from the live detector, never from host settings."""
        host = _host()
        root = os.environ.get("BC_MODEL_ROOT") or host.model_root
        if not root:
            raise ValueError(f"{config}: no model root (BC_MODEL_ROOT or host model-root)")
        bc_serve = binary.parent.parent / "radiance_home" / "bin" / "bc-serve"
        try:
            explained = radiance_inventory._run(bc_serve, [
                "explain", config, "--native-config", str(native_config), "--model-root", str(root),
                "--engine-binary", str(binary), "--json"])
            values = explained["values"]
            model = Path(values["engine"]["model"])
            engine_bdfs = tuple(values["layout"]["engine_cards"])
            inventory = radiance_inventory.detect(binary=bc_serve)
            engine = [inventory.by_bdf(bdf) for bdf in engine_bdfs]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{config}: bc-serve explain lacks the structured field {exc}") from exc
        except radiance_inventory.InventoryError as exc:
            raise ValueError(str(exc)) from exc
        if not engine:
            raise ValueError(f"{config}: bc-serve explain names no engine card")
        return ServingLayout(
            model=model.expanduser().resolve(),
            environment={"ROCR_VISIBLE_DEVICES": ",".join(str(g.ordinal) for g in engine),
                         "HIP_VISIBLE_DEVICES": ",".join(str(i) for i in range(len(engine)))},
            expected_execution={"backend": "ROCm", "architectures": [g.arch for g in engine],
                                "locators": list(engine_bdfs)},
        )

    def sidecars(self, arm: dict[str, Any], env: dict[str, str], run_dir: Path, binary: Path) -> list[SidecarSpec]:
        """The native owner starts and stops its own helpers; the harness runs none."""
        return []
