"""Fail-closed migration of the retired plan-qualification wrapper syntax.

This is deliberately not a general shell parser.  It accepts only the exact
``run_campaign.sh`` argv shape used by the lab queue and a narrow set of
scientific extra arguments that have direct JobSpec equivalents.  Device
ordinals, run names and output roots are reported as legacy metadata but never
become scientific resource identity.
"""
from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .model import BatchSpec, TargetPolicy


class LegacyMigrationError(ValueError):
    pass


@dataclass(frozen=True)
class LegacyMigration:
    batch: BatchSpec
    legacy_device: str
    legacy_run_name: str


_VALUE_FLAGS = {
    "--baseline-source",
    "--common-patches",
    "--producer-corpus",
    "--producer-input",
    "--validation-producer",
}
_IGNORED_VALUE_FLAGS = {
    # The service owns these locations/physical allocation now.
    "--device-map",
    "--workdir",
    "--worktree-root",
    "--build-root",
}
_BOOL_FLAGS = {"--production-lane"}


def _extra_args(tokens: tuple[str, ...]) -> dict[str, object]:
    result: dict[str, object] = {
        "baseline_source": "bigcherry-tuning",
        "common_patches": (),
        "producer_inputs": [],
        "producer_corpus": None,
        "production_lane": False,
        "validation_producer": None,
    }
    index = 0
    while index < len(tokens):
        flag = tokens[index]
        if flag in _BOOL_FLAGS:
            result["production_lane"] = True
            index += 1
            continue
        if flag in _VALUE_FLAGS or flag in _IGNORED_VALUE_FLAGS:
            if index + 1 >= len(tokens):
                raise LegacyMigrationError(f"{flag} requires a value")
            value = tokens[index + 1]
            index += 2
            if flag in _IGNORED_VALUE_FLAGS:
                continue
            if flag == "--baseline-source":
                result["baseline_source"] = value
            elif flag == "--common-patches":
                result["common_patches"] = tuple(
                    sorted({item for item in value.split(",") if item})
                )
            elif flag == "--producer-corpus":
                result["producer_corpus"] = value
            elif flag == "--validation-producer":
                result["validation_producer"] = value
            elif flag == "--producer-input":
                if "=" not in value:
                    raise LegacyMigrationError(
                        "--producer-input must use NAME=VALUE"
                    )
                key, raw = value.split("=", 1)
                if not key or any(existing[0] == key for existing in result["producer_inputs"]):
                    raise LegacyMigrationError(
                        f"duplicate/empty producer input key: {key!r}"
                    )
                result["producer_inputs"].append((key, raw))
            continue
        raise LegacyMigrationError(
            f"legacy argument {flag!r} has no canonical JobSpec mapping; migrate it explicitly"
        )
    return result


def migrate_run_campaign_command(
    command: str,
    *,
    environment: Mapping[str, str],
    planned_sessions: int,
    target: TargetPolicy | None = None,
    code_ref: str = "patch-refactor",
) -> LegacyMigration:
    """Translate one lab ``run_campaign.sh`` invocation into a BatchSpec.

    ``BC_MODEL`` and ``BC_HIP_PATH`` are mandatory because the shell wrapper
    obtained them from its environment.  ``planned_sessions`` is also
    mandatory: historical run names do not encode an honest predeclared
    scientific N, so migration refuses to infer one from queue length/results.
    """
    if planned_sessions < 1:
        raise LegacyMigrationError("planned_sessions must be >= 1")
    argv = shlex.split(command, posix=True)
    wrapper_index = next(
        (
            index
            for index, token in enumerate(argv)
            if Path(token).name == "run_campaign.sh"
        ),
        None,
    )
    if wrapper_index is None:
        raise LegacyMigrationError("command is not a run_campaign.sh invocation")
    args = argv[wrapper_index + 1 :]
    if len(args) < 5:
        raise LegacyMigrationError(
            "run_campaign.sh requires PATCH PRODUCER ARCH DEVICE RUN_NAME"
        )
    patch, producer, architecture, device, run_name = args[:5]
    extras = _extra_args(tuple(args[5:]))
    model = environment.get("BC_MODEL")
    hip_path = environment.get("BC_HIP_PATH")
    if not model or not hip_path:
        raise LegacyMigrationError(
            "legacy migration requires BC_MODEL and BC_HIP_PATH from the original queue environment"
        )
    selector = extras["validation_producer"]
    if producer != "-" and selector is not None and selector != producer:
        raise LegacyMigrationError(
            "positional producer disagrees with --validation-producer"
        )
    resolved_producer = None if producer == "-" else producer
    if resolved_producer is None and selector is not None:
        resolved_producer = str(selector)
    if resolved_producer is not None:
        expected_prefix = f"{patch}/"
        if not resolved_producer.startswith(expected_prefix):
            raise LegacyMigrationError(
                "validation producer must belong to the focal patch"
            )
    batch = BatchSpec(
        patch=patch,
        architectures=(architecture,),
        model=model,
        planned_sessions=planned_sessions,
        producer=resolved_producer,
        hip_path=hip_path,
        baseline_source=str(extras["baseline_source"]),
        common_patches=tuple(extras["common_patches"]),
        producer_inputs=tuple(extras["producer_inputs"]),
        producer_corpus=(
            None
            if extras["producer_corpus"] is None
            else str(extras["producer_corpus"])
        ),
        production_lane=bool(extras["production_lane"]),
        code_ref=code_ref,
        target=target or TargetPolicy(),
    )
    return LegacyMigration(
        batch=batch,
        legacy_device=device,
        legacy_run_name=run_name,
    )
