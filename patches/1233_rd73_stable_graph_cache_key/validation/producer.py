"""RD73 (1233) patch-local validation producer.

Migrated from the dedicated --run-rd73-contract CLI path (PA36 migration #5).

Design ruling (dev-gpt-agent req_a232ff7fb1f045db):
- Reuse scaffold llama-server pair (no build_pair())
- Producer-owned server measurements (NOT run_paired_llama_benchmark)
- Resource burst: separate artifact + promotion_resource_results channel
- Session aggregation: dispatcher-owned (producer emits current session only)
- bit_identical: producer MUST produce it
- 8 artifacts, standard_campaign="run", trace_probe="skip"
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

from bigcherry.bench import server_completion as sc
from bigcherry.campaign.benchmark import sanitize_environment
from bigcherry.experiment.attestation import ExecutionIdentity
from bigcherry.experiment.contract import (
    CorrectnessResult,
    LaneEffect,
    ResourceResult,
    TriggerEvidence,
)
from bigcherry.experiment.execution import (
    RunnerOutput,
    require_device_visibility,
    run_paired_lane,
)
from bigcherry.experiment.server_execution import AttestedServerSession
from bigcherry.patch.validation_producer import (
    ProducerContext,
    ProducerResult,
    ValidationProducerError,
)

logger = logging.getLogger(__name__)

# RD73's marker regex (from validation.toml)
_MARKER_REGEX = "BIGCHERRY_PATCH_HIT patch=1233_rd73 path=stable_graph_cache_key"

# MTP server runtime args (real llama-server flags only)
_MTP_SERVER_ARGS = (
    "--parallel", "1", "--metrics", "-sm", "tensor", "--fit", "off",
    "--spec-type", "draft-mtp", "--spec-draft-n-max", "4",
)

# ROCR_VISIBLE_DEVICES must be unset for all server launches
_ROCR_UNSET = ("ROCR_VISIBLE_DEVICES",)


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically (write to temp, then rename)."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        __import__("json").dumps(data, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.rename(path)


def _run_mtp_server_lane(
    *,
    ctx: ProducerContext,
    control_binary: Path,
    subject_binary: Path,
    expected_execution: ExecutionIdentity,
    selector_env: dict[str, str],
    host: str = "127.0.0.1",
    control_port: int = 18080,
    subject_port: int = 18081,
    warmup_pairs: int = 2,
    measured_pairs: int = 10,
) -> tuple[LaneEffect, list[dict[str, Any]], list[dict[str, Any]], Path, Path]:
    """Run RD73's paired control/subject MTP verify performance lane.

    Uses real AttestedServerSession + HTTP completion for each measured
    sample. Each single measured/warmup request gets its own fresh server
    (control and subject NEVER run concurrently -- each needs ~13GB/GPU
    under -sm tensor split, and two full copies exceed the 24.5GB/GPU
    dual-XTX cards).
    """
    if ctx.corpus is None:
        raise ValidationProducerError(
            "RD73: ctx.corpus is required for the MTP server lane"
        )

    prompts, corpus_sha256 = sc.load_corpus(ctx.corpus)
    metric_pattern = re.compile(r"BIGCHERRY_RD73_MTP wall_tps=([0-9.]+)")

    logs_dir = ctx.workdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Environment for server launches
    rd73_env = {
        "BIGCHERRY_PATCH_TRACE": "1",
        "BIGCHERRY_RD73_RESOURCE_TRACE": "1",
    }
    if selector_env:
        rd73_env.update(selector_env)
    rd73_env.pop("ROCR_VISIBLE_DEVICES", None)

    ports = {"control": control_port, "subject": subject_port}
    binaries = {"control": control_binary, "subject": subject_binary}
    per_request_logs: dict[str, list[Path]] = {"control": [], "subject": []}

    sampling = sc.SamplingConfig(temperature=1.0, top_p=0.95, top_k=20)
    session_kwargs = dict(
        corpus_id=ctx.corpus.stem,
        corpus_sha256=corpus_sha256,
        bigcherry_revision="rd73-va06",
        llama_pin="",
        llama_revision="",
        model_id=str(ctx.model),
        server_argv=_MTP_SERVER_ARGS,
        spec_type="draft-mtp",
        spec_n_max=4,
        spec_draft_k="default",
        spec_draft_v="default",
        sampling=sampling,
        n_predict=128,
        order_seed=12345,
    )
    configs = {
        "control": sc.SessionConfig(session_id="rd73-mtp-control", **session_kwargs),
        "subject": sc.SessionConfig(session_id="rd73-mtp-subject", **session_kwargs),
    }

    request_records: dict[str, list[dict[str, Any]]] = {"control": [], "subject": []}
    request_counters = {"control": 0, "subject": 0}

    def _runner(command: list[str]) -> RunnerOutput:
        arm = command[-1]
        index = request_counters[arm]
        request_counters[arm] += 1
        log_path = logs_dir / f"rd73-mtp-{arm}-server-{index}.log"
        per_request_logs[arm].append(log_path)
        session = AttestedServerSession(
            binary=binaries[arm],
            model=ctx.model,
            expected=expected_execution,
            host=host,
            port=ports[arm],
            extra_args=_MTP_SERVER_ARGS,
            log_path=log_path,
            env_overrides=rd73_env,
            env_unset=_ROCR_UNSET,
        )
        with session:
            transport = sc.HttpTransport(f"http://{host}:{ports[arm]}")
            sc.validate_server(transport)
            prompt = prompts[index % len(prompts)]
            record = sc.run_request(
                transport, prompt, configs[arm], pass_number=1, order_index=index
            )
        request_records[arm].append(record)
        if not isinstance(record.get("wall_tps"), (int, float)):
            raise ValidationProducerError(
                f"RD73 MTP lane ({arm}, request {index}): no usable wall_tps "
                "in the real completion response -- refusing to feed a "
                "missing sample into the paired statistics"
            )
        return RunnerOutput(
            returncode=0,
            stdout=f"BIGCHERRY_RD73_MTP wall_tps={record['wall_tps']}\n",
            stderr="",
        )

    # Warmup pairs (cold-cache discipline, never fed into statistics)
    for _ in range(warmup_pairs):
        _runner(["rd73-mtp-lane", "control"])
        _runner(["rd73-mtp-lane", "subject"])

    # Measured pairs (real alternating-order + block-bootstrap statistics)
    paired_run = run_paired_lane(
        metric="mtp_wall_tps",
        control_command=["rd73-mtp-lane", "control"],
        subject_command=["rd73-mtp-lane", "subject"],
        pattern=metric_pattern,
        pairs=measured_pairs,
        runner=_runner,
    )

    # Concatenate per-request logs into one combined log per arm
    subject_log = ctx.workdir / "rd73-mtp-subject.log"
    control_log = ctx.workdir / "rd73-mtp-control.log"
    for arm, log_file in [("subject", subject_log), ("control", control_log)]:
        combined = []
        for p in per_request_logs[arm]:
            if p.exists():
                combined.append(p.read_text(encoding="utf-8", errors="replace"))
        log_file.write_text("\n".join(combined), encoding="utf-8")

    # Convert to LaneEffect
    lane_effect = _paired_run_to_lane_effect(paired_run, role="positive")

    return (
        lane_effect,
        request_records["subject"],
        request_records["control"],
        subject_log,
        control_log,
    )


def _run_decode_control_lane(
    *,
    ctx: ProducerContext,
    control_binary: Path,
    subject_binary: Path,
    expected_execution: ExecutionIdentity,
    selector_env: dict[str, str],
    host: str = "127.0.0.1",
    control_port: int = 18082,
    subject_port: int = 18083,
    measured_pairs: int = 10,
) -> LaneEffect:
    """Run RD73's decode control lane (plain server, tg128)."""
    if ctx.corpus is None:
        raise ValidationProducerError(
            "RD73: ctx.corpus is required for the decode control lane"
        )

    prompts, corpus_sha256 = sc.load_corpus(ctx.corpus)
    metric_pattern = re.compile(r"BIGCHERRY_RD73_DECODE tg128=([0-9.]+)")

    logs_dir = ctx.workdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Plain server args (no MTP flags)
    server_args = ("--parallel", "1", "--metrics", "-sm", "tensor", "--fit", "off")

    rd73_env = {"BIGCHERRY_PATCH_TRACE": "1"}
    if selector_env:
        rd73_env.update(selector_env)
    rd73_env.pop("ROCR_VISIBLE_DEVICES", None)

    ports = {"control": control_port, "subject": subject_port}
    binaries = {"control": control_binary, "subject": subject_binary}

    sampling = sc.SamplingConfig(temperature=1.0, top_p=0.95, top_k=20)
    session_kwargs = dict(
        corpus_id=ctx.corpus.stem,
        corpus_sha256=corpus_sha256,
        bigcherry_revision="rd73-va06",
        llama_pin="",
        llama_revision="",
        model_id=str(ctx.model),
        server_argv=server_args,
        sampling=sampling,
        n_predict=128,
        order_seed=12345,
    )
    configs = {
        "control": sc.SessionConfig(session_id="rd73-decode-control", **session_kwargs),
        "subject": sc.SessionConfig(session_id="rd73-decode-subject", **session_kwargs),
    }

    request_counters = {"control": 0, "subject": 0}

    def _runner(command: list[str]) -> RunnerOutput:
        arm = command[-1]
        index = request_counters[arm]
        request_counters[arm] += 1
        log_path = logs_dir / f"rd73-decode-{arm}-server-{index}.log"
        session = AttestedServerSession(
            binary=binaries[arm],
            model=ctx.model,
            expected=expected_execution,
            host=host,
            port=ports[arm],
            extra_args=server_args,
            log_path=log_path,
            env_overrides=rd73_env,
            env_unset=_ROCR_UNSET,
        )
        with session:
            transport = sc.HttpTransport(f"http://{host}:{ports[arm]}")
            sc.validate_server(transport)
            prompt = prompts[index % len(prompts)]
            record = sc.run_request(
                transport, prompt, configs[arm], pass_number=1, order_index=index
            )
        # For decode, we use tg128 (tokens per second at 128 context)
        tg128 = record.get("tg128") or record.get("wall_tps")
        if not isinstance(tg128, (int, float)):
            raise ValidationProducerError(
                f"RD73 decode lane ({arm}, request {index}): no usable tg128 "
                "in the real completion response"
            )
        return RunnerOutput(
            returncode=0,
            stdout=f"BIGCHERRY_RD73_DECODE tg128={tg128}\n",
            stderr="",
        )

    paired_run = run_paired_lane(
        metric="tg128",
        control_command=["rd73-decode-lane", "control"],
        subject_command=["rd73-decode-lane", "subject"],
        pattern=metric_pattern,
        pairs=measured_pairs,
        runner=_runner,
    )

    return _paired_run_to_lane_effect(paired_run, role="control")


def _run_resource_burst(
    *,
    ctx: ProducerContext,
    subject_binary: Path,
    expected_execution: ExecutionIdentity,
    selector_env: dict[str, str],
    host: str = "127.0.0.1",
    port: int = 18084,
    requests: int = 20,
) -> int:
    """Run the resource burst session (long-lived subject-only MTP server).

    Returns peak graph_cache_entries. Fails closed on missing telemetry.
    """
    if ctx.corpus is None:
        raise ValidationProducerError(
            "RD73: ctx.corpus is required for the resource burst"
        )

    prompts, corpus_sha256 = sc.load_corpus(ctx.corpus)

    rd73_env = {
        "BIGCHERRY_PATCH_TRACE": "1",
        "BIGCHERRY_RD73_RESOURCE_TRACE": "1",
    }
    if selector_env:
        rd73_env.update(selector_env)
    rd73_env.pop("ROCR_VISIBLE_DEVICES", None)

    log_path = ctx.workdir / "rd73-resource-burst-subject.log"

    sampling = sc.SamplingConfig(temperature=1.0, top_p=0.95, top_k=20)
    session_kwargs = dict(
        corpus_id=ctx.corpus.stem,
        corpus_sha256=corpus_sha256,
        bigcherry_revision="rd73-va06",
        llama_pin="",
        llama_revision="",
        model_id=str(ctx.model),
        server_argv=_MTP_SERVER_ARGS,
        spec_type="draft-mtp",
        spec_n_max=4,
        spec_draft_k="default",
        spec_draft_v="default",
        sampling=sampling,
        n_predict=128,
        order_seed=12345,
    )
    config = sc.SessionConfig(session_id="rd73-resource-burst", **session_kwargs)

    # Long-lived server: launch once, run all requests, shut down
    session = AttestedServerSession(
        binary=subject_binary,
        model=ctx.model,
        expected=expected_execution,
        host=host,
        port=port,
        extra_args=_MTP_SERVER_ARGS,
        log_path=log_path,
        env_overrides=rd73_env,
        env_unset=_ROCR_UNSET,
    )

    peak_entries = 0
    with session:
        transport = sc.HttpTransport(f"http://{host}:{port}")
        sc.validate_server(transport)
        for i in range(requests):
            prompt = prompts[i % len(prompts)]
            sc.run_request(transport, prompt, config, pass_number=1, order_index=i)

    # Parse the resource telemetry from the log
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    for line in log_text.splitlines():
        if "BIGCHERRY_RD73_RESOURCE" in line:
            match = re.search(r"graph_cache_entries=(\d+)", line)
            if match:
                entries = int(match.group(1))
                peak_entries = max(peak_entries, entries)

    # Fail closed on missing telemetry
    if peak_entries == 0:
        raise ValidationProducerError(
            "RD73 resource burst: no graph_cache_entries telemetry observed -- "
            "the subject server's telemetry (BIGCHERRY_RD73_RESOURCE_TRACE=1) "
            "never emitted a reading. Refusing to pass a zero measurement."
        )

    return peak_entries


def _check_bit_identical(
    subject_records: list[dict[str, Any]],
    control_records: list[dict[str, Any]],
) -> bool:
    """Check if the control and subject MTP content fields are bit-identical."""
    if len(subject_records) != len(control_records):
        return False
    for s_rec, c_rec in zip(subject_records, control_records):
        s_content = s_rec.get("content")
        c_content = c_rec.get("content")
        if s_content != c_content:
            return False
    return True


def _paired_run_to_lane_effect(
    paired_run: Any, role: str
) -> LaneEffect:
    """Convert a PairedLaneRun to a LaneEffect."""
    stats = paired_run.stats
    geometric_effect_pct = stats.get("geometric_effect_pct", 0.0)
    ci95_low = stats.get("ci95_low_pct", 0.0)
    ci95_high = stats.get("ci95_high_pct", 0.0)
    paired_rounds = stats.get("paired_rounds", 0)

    # Determine decision from the effect
    if role == "positive":
        decision = "pass" if geometric_effect_pct >= 0 else "fail"
    else:
        # For control: regression is bad (negative effect = regression)
        decision = "pass" if geometric_effect_pct >= -5.0 else "fail"

    return LaneEffect(
        role=role,
        metric=stats.get("metric", "unknown"),
        geometric_effect_pct=geometric_effect_pct,
        decision=decision,
        ci95_low_pct=ci95_low,
        ci95_high_pct=ci95_high,
        paired_rounds=paired_rounds,
        pair_ratios=tuple(stats.get("pair_ratios", [])),
    )


def run(ctx: ProducerContext) -> ProducerResult:
    """RD73 producer entrypoint."""
    # --- Preflight: device visibility (validate ONCE) ---
    visibility = require_device_visibility(
        context=f"{ctx.patch_id}: RD73 preflight",
        exact_count=2,
        env=ctx.build_env,
    )

    # --- Get the scaffold binaries ---
    control_server = ctx.validation_binaries.get("control", {}).get("llama-server")
    subject_server = ctx.validation_binaries.get("subject", {}).get("llama-server")
    if control_server is None or subject_server is None:
        raise ValidationProducerError(
            "RD73: scaffold llama-server binaries not found in "
            f"validation_binaries. control={control_server is not None}, "
            f"subject={subject_server is not None}"
        )

    # --- Build the execution identity ---
    # RD73 requires 2 visible GPUs (dual-GPU tensor split)
    execution_identity = ExecutionIdentity(
        backend="rocm",
        architectures=("gfx1100", "gfx1100"),
    )

    # --- Selector env (from the validated visibility) ---
    selector_env = {
        "HIP_VISIBLE_DEVICES": ",".join(
            str(d) for d in visibility.device_ids
        ),
    }

    # --- MTP server lane (positive) ---
    mtp_lane, subject_records, control_records, subject_log, control_log = (
        _run_mtp_server_lane(
            ctx=ctx,
            control_binary=control_server,
            subject_binary=subject_server,
            expected_execution=execution_identity,
            selector_env=selector_env,
        )
    )

    # --- Decode control lane ---
    decode_lane = _run_decode_control_lane(
        ctx=ctx,
        control_binary=control_server,
        subject_binary=subject_server,
        expected_execution=execution_identity,
        selector_env=selector_env,
    )

    # --- Resource burst session ---
    peak_entries = _run_resource_burst(
        ctx=ctx,
        subject_binary=subject_server,
        expected_execution=execution_identity,
        selector_env=selector_env,
    )

    # --- Bit-identical correctness ---
    bit_identical_passed = _check_bit_identical(subject_records, control_records)

    # --- Activation evidence (from the MTP logs) ---
    pattern = re.compile(_MARKER_REGEX)
    subject_text = subject_log.read_text(encoding="utf-8", errors="replace")
    control_text = control_log.read_text(encoding="utf-8", errors="replace")
    subject_hit = pattern.search(subject_text) is not None
    control_hit = pattern.search(control_text) is not None

    # --- Write artifacts through the runtime API ---
    # 1. rd73-mtp-lane.json
    mtp_lane_doc = {
        "metric": "mtp_wall_tps",
        "geometric_effect_pct": mtp_lane.geometric_effect_pct,
        "decision": mtp_lane.decision,
        "paired_rounds": mtp_lane.paired_rounds,
        "ci95_low_pct": mtp_lane.ci95_low_pct,
        "ci95_high_pct": mtp_lane.ci95_high_pct,
    }
    mtp_lane_ref = ctx.runtime.write_artifact(
        name="rd73-mtp-lane.json",
        payload=__import__("json").dumps(mtp_lane_doc, indent=2, sort_keys=True),
    )

    # 2. rd73-decode-control.json
    decode_doc = {
        "metric": "tg128",
        "geometric_effect_pct": decode_lane.geometric_effect_pct,
        "decision": decode_lane.decision,
        "paired_rounds": decode_lane.paired_rounds,
        "ci95_low_pct": decode_lane.ci95_low_pct,
        "ci95_high_pct": decode_lane.ci95_high_pct,
    }
    decode_ref = ctx.runtime.write_artifact(
        name="rd73-decode-control.json",
        payload=__import__("json").dumps(decode_doc, indent=2, sort_keys=True),
    )

    # 3. rd73-resource.json
    resource_doc = {
        "metric": "graph_cache_entries",
        "unit": "count",
        "subject_value": peak_entries,
    }
    resource_ref = ctx.runtime.write_artifact(
        name="rd73-resource.json",
        payload=__import__("json").dumps(resource_doc, indent=2, sort_keys=True),
    )

    # 4. rd73-resource-burst-subject.log
    burst_log_text = f"peak_graph_cache_entries={peak_entries}\n"
    burst_log_ref = ctx.runtime.write_text_artifact(
        name="rd73-resource-burst-subject.log",
        content=burst_log_text,
    )

    # 5. rd73-correctness.json
    correctness_doc = {
        "check": "bit_identical",
        "passed": bit_identical_passed,
    }
    correctness_ref = ctx.runtime.write_artifact(
        name="rd73-correctness.json",
        payload=__import__("json").dumps(correctness_doc, indent=2, sort_keys=True),
    )

    # 6. rd73-mtp-subject.log
    subject_log_ref = ctx.runtime.write_text_artifact(
        name="rd73-mtp-subject.log",
        content=subject_text,
    )

    # 7. rd73-mtp-control.log
    control_log_ref = ctx.runtime.write_text_artifact(
        name="rd73-mtp-control.log",
        content=control_text,
    )

    # 8. rd73-performance.json
    performance_doc = {
        "mtp_wall_tps": mtp_lane.geometric_effect_pct,
        "tg128": decode_lane.geometric_effect_pct,
        "artifact": {
            "mtp_lane": mtp_lane_ref,
            "decode_control": decode_ref,
        },
    }
    performance_ref = ctx.runtime.write_artifact(
        name="rd73-performance.json",
        payload=__import__("json").dumps(performance_doc, indent=2, sort_keys=True),
    )

    # --- Build the ProducerResult ---
    # Trigger evidence: positive lane must have the marker hit;
    # control marker hit must invalidate promotion
    trigger_evidence = (
        TriggerEvidence(
            role="positive",
            lane_id="rd73-mtp-subject",
            candidate_launches=1 if subject_hit else 0,
        ),
    )

    return ProducerResult(
        correctness={
            "disposition": "passed" if bit_identical_passed else "failed",
            "mechanism": "bit_identical",
            "detail": "RD73 MTP bit-identical check",
            "artifact": correctness_ref,
        },
        validation_build_identities={
            "control": dict(ctx.validation_build_identities["control"]),
            "subject": dict(ctx.validation_build_identities["subject"]),
        },
        activation_evidence={
            "disposition": "activation-verified" if subject_hit and not control_hit else "activation-failed",
            "mechanism": "trace-marker",
            "detail": f"marker_regex={_MARKER_REGEX}",
            "subject_hit": subject_hit,
            "control_hit": control_hit,
            "positive": {"artifact": subject_log_ref, "marker_regex": _MARKER_REGEX},
            "negative": {"artifact": control_log_ref, "marker_regex": _MARKER_REGEX},
        },
        performance_evidence={
            "disposition": "measured",
            "mechanism": "paired-server-bench",
            "detail": "RD73 MTP + decode lanes",
            "artifact": performance_ref,
        },
        trace_evidence={
            "disposition": "measured",
            "mechanism": "trace-marker",
            "detail": f"marker_regex={_MARKER_REGEX}",
            "positive": {"artifact": subject_log_ref, "marker_regex": _MARKER_REGEX},
            "negative": {"artifact": control_log_ref, "marker_regex": _MARKER_REGEX},
        },
        check_results=(),
        lane_effects=(),
        emitted_artifacts=frozenset(
            [
                "rd73-mtp-lane.json",
                "rd73-decode-control.json",
                "rd73-resource.json",
                "rd73-resource-burst-subject.log",
                "rd73-correctness.json",
                "rd73-mtp-subject.log",
                "rd73-mtp-control.log",
                "rd73-performance.json",
            ]
        ),
        contract_correctness_results=(
            CorrectnessResult(
                check="bit_identical",
                passed=bit_identical_passed,
            ),
        ),
        promotion_lane_effects={
            "RD73-STABLE-GRAPH-CACHE-KEY": (mtp_lane, decode_lane),
        },
        promotion_target_metric={
            "RD73-STABLE-GRAPH-CACHE-KEY": "mtp_wall_tps",
        },
        promotion_trigger_evidence={
            "RD73-STABLE-GRAPH-CACHE-KEY": trigger_evidence,
        },
        promotion_resource_results={
            "RD73-STABLE-GRAPH-CACHE-KEY": (
                ResourceResult(
                    metric="graph_cache_entries",
                    unit="count",
                    subject_value=float(peak_entries),
                ),
            ),
        },
    )
