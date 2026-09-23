"""HI81 (informal): unattended patch-validation campaign.

Given only a patch module name and a model path, materializes an isolated,
content-addressed source worktree for that one patch (patch_source_
isolation.py, HI82 -- never the shared vendor/llama.cpp tree), builds the
tune-mode/replay-mode/stock-baseline trees from it, then runs the full
record->tune->promote->export->replay->bench->report smoke campaign
(tools/bigcherry/e2e_smoke_campaign.py) against it -- the same process used
to validate HI30/HI31 this session, generalized to any single untested
patch so a human only has to choose the patch and the model.

This intentionally does NOT decide whether the patch is good -- it produces
the same report.md/bench.json/measurements.jsonl a human (or a follow-up
GPT review) reads to make that call. It also does not touch git or the
patch's catalog state (validated/rejected/untested) -- promoting a patch
out of "untested" is a separate, deliberate decision.

Usage:
    python -m bigcherry.patch.validation_campaign \\
        --patch <patch-id> \\
        --model $BC_MODEL_ROOT/qwen3.5-2b/Qwen_Qwen3.5-2B-Q4_K_M.gguf \\
        --hip-path vendor/rocm/7.1 --amdgpu-targets gfx1100 \\
        --manifest artifacts/<rev>/hip-autotune-manifest.json \\
        --workdir work/patch-1204-qwen2b

Safe to re-run: source materialization and every build/campaign stage below
reuse existing output where present (patch_source_isolation.py's manifest-
verified worktree reuse, cmake incremental builds, e2e_smoke_campaign.py's
per-stage resume-check).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from bigcherry.build.builds import capture_completed_build_evidence
from bigcherry.campaign.bench_runner import run_bench_runner_server_bench
from bigcherry.core.context import ProjectContext
from bigcherry.core.paths import REPO_ROOT
from bigcherry.experiment import contract as experiment_contract
from bigcherry.experiment.attestation import ExecutionIdentity
from bigcherry.experiment.server_execution import AttestedServerSession
from bigcherry.patch.activation import (
    ActivationEvidence,
    verdict,
    write_activation_json,
)
from bigcherry.patch.campaign.benchmark import _run_performance_benchmark
from bigcherry.patch.campaign.build import (
    _atomic_write_json,
    _full_requested_cmake_args,
    _hip_env,
    _hip_only,
    _print,
    _ROCR_VISIBLE_DEVICES_UNSET,
    _write_bound_artifact,
    build_tree,
    generate_registry,
    PatchCampaignError,
)
from bigcherry.patch.campaign.contract import (
    build_contract_evidence_for_persistence,
    collect_lane_effect_records,
    compute_contract_correctness_gate,
    compute_persisted_validation_eligible,
)
from bigcherry.patch.campaign.producer import (
    _parse_producer_inputs,
    _parse_validation_producer_selector,
    _run_validation_producer,
)
from bigcherry.patch.campaign.scaffold import _build_standard_campaign_scaffold
from bigcherry.patch.campaign.trace import run_trace_activation_probes


_RD73_RESOURCE_PREFIX = "BIGCHERRY_RD73_RESOURCE"


_RD73_RESOURCE_PATTERN = re.compile(
    r"BIGCHERRY_RD73_RESOURCE graph_cache_entries=(\d+)\s*$"
)


def parse_rd73_resource_telemetry(text: str) -> tuple[int, ...]:
    """VA06: pure parser for RD73's opt-in graph-cache-entry telemetry
    (BIGCHERRY_RD73_RESOURCE_TRACE=1, common.cuh's cuda_graph() insertion
    site) -- extracts every real ``graph_cache_entries=N`` reading from a
    process's combined stdout/stderr, in emission order. Returns an empty
    tuple when the patch never emitted (e.g. the control binary, which
    has no RD73 telemetry code at all). Fails closed: any line carrying
    the ``BIGCHERRY_RD73_RESOURCE`` prefix that doesn't match the exact
    expected shape is treated as corrupt evidence, not silently ignored --
    even when other lines in the same text parse cleanly."""
    readings = []
    for line in text.splitlines():
        if _RD73_RESOURCE_PREFIX not in line:
            continue
        match = _RD73_RESOURCE_PATTERN.search(line)
        if match is None:
            raise PatchCampaignError(
                f"rd73 resource telemetry: malformed BIGCHERRY_RD73_RESOURCE line: {line!r}"
            )
        readings.append(int(match.group(1)))
    return tuple(readings)


def peak_rd73_resource_result(
    subject_readings: "tuple[int, ...] | list[int]",
    control_readings: "tuple[int, ...] | list[int] | None" = None,
) -> "experiment_contract.ResourceResult":
    """VA06: reduce raw telemetry readings into the real
    ``ResourceResult(metric="graph_cache_entries", unit="count", ...)``
    evaluate_resource_gate() checks against. Fails closed (raises) on
    missing/malformed evidence -- an empty subject reading set means the
    real telemetry was never observed at all, which must never silently
    read as a zero/passing measurement."""
    from bigcherry.experiment import contract as experiment_contract

    if not subject_readings:
        raise PatchCampaignError(
            "rd73 resource evidence: no graph_cache_entries readings observed -- the "
            "subject binary's telemetry (BIGCHERRY_RD73_RESOURCE_TRACE=1) never emitted"
        )
    if any(
        not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in subject_readings
    ):
        raise PatchCampaignError(
            f"rd73 resource evidence: malformed subject reading(s) in {subject_readings!r}"
        )
    control_value = None
    if control_readings:
        if any(
            not isinstance(v, int) or isinstance(v, bool) or v < 0
            for v in control_readings
        ):
            raise PatchCampaignError(
                f"rd73 resource evidence: malformed control reading(s) in {control_readings!r}"
            )
        control_value = float(max(control_readings))
    return experiment_contract.ResourceResult(
        metric="graph_cache_entries",
        unit="count",
        subject_value=float(max(subject_readings)),
        control_value=control_value,
    )


def evaluate_rd73_activation_evidence(
    *,
    marker_regex: str,
    control_log_path: Path,
    subject_log_path: Path,
    run_dir: Path,
) -> dict[str, object]:
    """VA06 (user redirect, 2026-09-01): RD73's real subject-hit/control-miss
    activation evidence, read from the control/subject llama-server LOG
    FILES run_rd73_mtp_server_lane() already produced (BIGCHERRY_PATCH_TRACE=1
    is always set on those servers) -- no separate llama-bench probe.
    llama-bench itself has proven unworkable for RD73's real 27B/dual-GPU/
    -sm-tensor config on real hardware (repeated crashes: OOM under
    resource contention, --fit argument-parse errors), and a second probe
    would be redundant anyway: the MTP servers already ran the patched/
    control binaries under real repeated traffic. Mirrors RD08's own
    control-vs-subject-binary negative control (never the generic tune-
    binary/GGML_CUDA_DISABLE_FUSION mechanism, which is invalid for RD73's
    graph-cache-key marker for the same reason RD08's own docstring
    already establishes)."""
    pattern = re.compile(marker_regex)
    subject_text = Path(subject_log_path).read_text(encoding="utf-8", errors="replace")
    control_text = Path(control_log_path).read_text(encoding="utf-8", errors="replace")
    subject_hit = pattern.search(subject_text) is not None
    control_hit = pattern.search(control_text) is not None
    subject_rel = Path(subject_log_path).relative_to(run_dir).as_posix()
    control_rel = Path(control_log_path).relative_to(run_dir).as_posix()
    doc = {
        "marker_regex": marker_regex,
        "subject_hit": subject_hit,
        "control_hit": control_hit,
        "positive": {
            "artifact": {
                "path": subject_rel,
                "sha256": hashlib.sha256(
                    Path(subject_log_path).read_bytes()
                ).hexdigest(),
            },
        },
        "control": {
            "artifact": {
                "path": control_rel,
                "sha256": hashlib.sha256(
                    Path(control_log_path).read_bytes()
                ).hexdigest(),
            },
        },
    }
    artifact_ref = _write_bound_artifact(run_dir, "rd73-activation.json", doc)
    return {
        "subject_hit": subject_hit,
        "control_hit": control_hit,
        "artifact": artifact_ref,
        "subject_log_path": subject_rel,
        "control_log_path": control_rel,
        # VA23: the per-log bound refs, so the campaign can build the
        # positive/negative trace_evidence that _builtin_trace_marker()
        # requires. It re-reads both logs and re-verifies the marker itself,
        # so this exposes evidence for independent checking rather than
        # asserting a result -- subject_hit/control_hit above are NOT what
        # the validator trusts.
        "positive": {
            "artifact": doc["positive"]["artifact"],
            "marker_regex": marker_regex,
        },
        "negative": {
            "artifact": doc["control"]["artifact"],
            "marker_regex": marker_regex,
        },
    }


def run_rd73_mtp_server_lane(
    *,
    control_binary: Path,
    subject_binary: Path,
    model: Path,
    corpus_path: Path,
    run_dir: Path,
    expected_execution: ExecutionIdentity,
    host: str = "127.0.0.1",
    control_port: int = 18080,
    subject_port: int = 18081,
    spec_draft_n_max: int = 4,
    n_predict: int = 128,
    warmup_pairs: int = 2,
    measured_pairs: int = 10,
    selector_env: "dict[str, str] | None" = None,
) -> dict[str, object]:
    """VA06 next slice: RD73's paired control/subject mtp_verify
    performance lane over a real llama-server HTTP harness (GPT scoping,
    session ses_89a3ef2b02b94469, req_a25bb805975c43c0/req corrected):
    upstream llama-bench does not support speculative/MTP flags at all,
    so unlike RD08's simple paired-subprocess lanes, this reuses
    tuning/server_runner.py's ServerRunner for real process lifecycle
    (launch/health-check/shutdown) and bench/server_completion.py's real
    request/metrics machinery for each measured sample.

    Target metric is wall_tps (client-measured, real request-to-response
    wall-clock throughput) -- deliberately NOT predicted_tps, which is
    the server's own self-reported decode timing and can exclude HTTP/
    queueing overhead; per GPT direction, "the number an end user
    actually experiences" is what this contract's end_to_end_gain_pct
    must measure.

    Reuses experiment/execution.py's run_paired_lane() (RD08's own
    alternating-order + block-bootstrap statistics engine) via a
    synthetic-stdout adapter rather than duplicating that statistics
    code: each paired-lane "command" is a control/subject arm tag, and
    the injected runner performs one real HTTP completion request against
    the already-launched server for that arm, encoding the real wall_tps
    it measured into a parseable stdout line. warmup_pairs real paired
    requests execute first (cold-cache discipline, matching
    server_completion.run_session()'s own pattern) and are never fed into
    the paired statistics; only the following measured_pairs are.

    Every per-request record (including generated ``content``) is
    retained and returned for RD73's separate bit-identical correctness
    lane to consume -- this function does not itself judge correctness.
    Fails closed: a request with no usable wall_tps raises
    PatchCampaignError immediately, never silently drops a sample."""
    from bigcherry.bench import server_completion as sc
    from bigcherry.experiment import execution as experiment_execution

    prompts, corpus_sha256 = sc.load_corpus(corpus_path)
    metric_pattern = re.compile(r"BIGCHERRY_RD73_MTP wall_tps=([0-9.]+)")

    # Real llama-server CLI flags only (verified against vendor/llama.cpp's
    # own common/arg.cpp -- an earlier draft of this function invented
    # "--spec-n-max"/"--spec-draft-k"/"--spec-draft-v", none of which
    # exist; the real flag is --spec-draft-n-max, and there is no
    # separate draft-cache-type flag this lane needs to set (the
    # production dual-XTX/27B baseline profile leaves cache types at
    # their defaults too). -sm tensor is REQUIRED for this 27B model on
    # 2x gfx1100 -- the default -sm layer understates throughput by
    # roughly 2-10x (a real, previously-confirmed production finding).
    # --fit off is ALSO required alongside -sm tensor for llama-SERVER
    # specifically: llama.cpp's automatic device-memory-fit feature
    # (default on) raises "llama_params_fit is not implemented for
    # SPLIT_MODE_TENSOR" and aborts (common/fit.cpp) -- a real hardware
    # crash found running this exact lane on Brutus. llama-BENCH (used
    # by RD73's other lanes) does not register this flag at all --
    # passing --fit to it is itself a hard error ("invalid parameter for
    # argument: --fit"), also found on real hardware -- so it must never
    # be added to those lanes' extra_flags.
    server_args = (
        "--parallel",
        "1",
        "--metrics",
        "-sm",
        "tensor",
        "--fit",
        "off",
        "--spec-type",
        "draft-mtp",
        "--spec-draft-n-max",
        str(spec_draft_n_max),
    )
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # User redirect (2026-09-01, real hardware finding): control and
    # subject servers must NEVER run concurrently for this model --
    # each needs ~13GB/GPU under -sm tensor split, and two full copies
    # exceed the 24.5GB/GPU Brutus dual-XTX cards (a real cudaMalloc
    # out-of-memory abort, confirmed on hardware). So each single
    # measured/warmup request gets its own fresh server: launch, one
    # request, shut down -- alternating control/subject in the same
    # order run_paired_lane already calls them, preserving the real
    # alternating-order/thermal-drift discipline this project's own
    # prior production benchmarking found necessary (a non-alternating
    # "all control then all subject" design previously produced a real,
    # since-corrected measurement artifact on this exact model/hardware
    # -- see patches/1233.../README.md's "Historical evidence" section).
    #
    # PVPS02 step 7 (2026-09-11): ``selector_env`` carries the already-
    # validated HIP_VISIBLE_DEVICES value from
    # run_rd73_contract_qualification()'s single require_device_visibility()
    # call (validate ONCE, not per-lane/per-request) -- ServerRunner
    # starts each server from ambient env then applies these overrides,
    # so this is enough to make the selector explicit for every server
    # this lane launches without touching global os.environ.
    rd73_env = {"BIGCHERRY_PATCH_TRACE": "1", "BIGCHERRY_RD73_RESOURCE_TRACE": "1"}
    if selector_env:
        rd73_env.update(selector_env)
    # GPT review follow-up (req_d1ef22d846854960, 2026-09-11): defense in
    # depth alongside env_unset=_ROCR_VISIBLE_DEVICES_UNSET below -- a
    # future direct caller passing ROCR_VISIBLE_DEVICES via selector_env
    # must not be able to reintroduce it through env_overrides ordering
    # (ServerRunner.launch() applies env_unset BEFORE env_overrides).
    rd73_env.pop("ROCR_VISIBLE_DEVICES", None)
    ports = {"control": control_port, "subject": subject_port}
    binaries = {"control": control_binary, "subject": subject_binary}
    per_request_logs: dict[str, list[Path]] = {"control": [], "subject": []}

    sampling = sc.SamplingConfig(temperature=1.0, top_p=0.95, top_k=20)
    session_kwargs = dict(
        corpus_id=corpus_path.stem,
        corpus_sha256=corpus_sha256,
        bigcherry_revision="rd73-va06",
        llama_pin="",
        llama_revision="",
        model_id=str(model),
        server_argv=server_args,
        spec_type="draft-mtp",
        spec_n_max=spec_draft_n_max,
        # SessionConfig's spec_draft_k/spec_draft_v fields are provenance
        # labels only (there is no real --spec-draft-k/--spec-draft-v
        # llama-server flag); "default" records that this lane leaves the
        # draft cache type at its build default, matching the production
        # dual-XTX/27B baseline profile, which does not override it either.
        spec_draft_k="default",
        spec_draft_v="default",
        sampling=sampling,
        n_predict=n_predict,
        order_seed=12345,
    )
    configs = {
        "control": sc.SessionConfig(session_id="rd73-mtp-control", **session_kwargs),
        "subject": sc.SessionConfig(session_id="rd73-mtp-subject", **session_kwargs),
    }

    request_records: dict[str, list[dict[str, object]]] = {"control": [], "subject": []}
    request_counters = {"control": 0, "subject": 0}

    def _runner(command: list[str]) -> "experiment_execution.RunnerOutput":
        arm = command[-1]
        index = request_counters[arm]
        request_counters[arm] += 1
        log_path = logs_dir / f"rd73-mtp-{arm}-server-{index}.log"
        per_request_logs[arm].append(log_path)
        session = AttestedServerSession(
            binary=binaries[arm],
            model=model,
            expected=expected_execution,
            host=host,
            port=ports[arm],
            extra_args=server_args,
            log_path=log_path,
            env_overrides=rd73_env,
            env_unset=_ROCR_VISIBLE_DEVICES_UNSET,
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
            raise PatchCampaignError(
                f"rd73 mtp lane ({arm}, request {index}): no usable wall_tps in the "
                f"real completion response -- refusing to feed a missing sample into "
                f"the paired statistics"
            )
        return experiment_execution.RunnerOutput(
            returncode=0,
            stdout=f"BIGCHERRY_RD73_MTP wall_tps={record['wall_tps']}\n",
            stderr="",
        )

    for _ in range(warmup_pairs):
        _runner(["rd73-mtp-lane", "control"])
        _runner(["rd73-mtp-lane", "subject"])

    paired_run = experiment_execution.run_paired_lane(
        metric="mtp_wall_tps",
        control_command=["rd73-mtp-lane", "control"],
        subject_command=["rd73-mtp-lane", "subject"],
        pattern=metric_pattern,
        pairs=measured_pairs,
        runner=_runner,
    )

    # Concatenate each arm's per-request server logs into one combined
    # log file, so downstream evidence readers (correctness/activation
    # investigation, manual debugging) see one file per arm as before,
    # even though each request used its own fresh process.
    combined_log_paths: dict[str, Path] = {}
    for arm in ("control", "subject"):
        combined_path = logs_dir / f"rd73-mtp-{arm}-server.log"
        combined_path.write_text(
            "".join(
                p.read_text(encoding="utf-8", errors="replace")
                for p in per_request_logs[arm]
            ),
            encoding="utf-8",
        )
        combined_log_paths[arm] = combined_path

    effect = experiment_execution.lane_effect_from_run(
        "positive", "mtp_wall_tps", paired_run
    )
    doc = {
        "metric": "mtp_wall_tps",
        "stats": paired_run.stats,
        "warmup_pairs": warmup_pairs,
        "measured_pairs": measured_pairs,
        "control_requests": request_records["control"],
        "subject_requests": request_records["subject"],
    }
    artifact_ref = _write_bound_artifact(run_dir, "rd73-mtp-lane.json", doc)
    return {
        "effect": effect,
        "artifact": artifact_ref,
        "stats": paired_run.stats,
        "control_requests": request_records["control"],
        "subject_requests": request_records["subject"],
        "control_log_path": combined_log_paths["control"],
        "subject_log_path": combined_log_paths["subject"],
    }


# VA26: run_bench_runner_server_bench() and its constants moved to
# campaign/bench_runner.py -- the documented server-bench harness is not
# patch-specific, and the qualification matrix needs it without importing
# patch internals. Imported below; no alias is kept here.
def run_rd73_decode_control_lane(
    *,
    control_binary: Path,
    subject_binary: Path,
    model: Path,
    run_dir: Path,
    expected_execution: ExecutionIdentity,
    host: str = "127.0.0.1",
    control_port: int = 18082,
    subject_port: int = 18083,
    pairs: int = 3,
    extra_flags: tuple[str, ...] = ("-sm", "tensor", "--fit", "off"),
    selector_env: "dict[str, str] | None" = None,
) -> dict[str, object]:
    """VA06 (user redirect, 2026-09-01): RD73's decode control lane --
    launches real, plain (non-speculative) control/subject llama-server
    processes (ServerRunner, matching run_rd73_mtp_server_lane()'s
    lifecycle pattern) and drives each paired measurement via the
    documented Brutus bench runner (run_bench_runner_server_bench(),
    "tg128" config) rather than a raw llama-bench subprocess -- see that
    function's docstring for why llama-bench itself is unworkable here.
    Reuses experiment/execution.py's run_paired_lane() via the same
    synthetic-stdout adapter pattern as the MTP lane, rather than
    duplicating its alternating-order + block-bootstrap statistics."""
    from bigcherry.experiment import execution as experiment_execution

    metric_pattern = re.compile(r"BIGCHERRY_RD73_DECODE tg128_tps=([0-9.]+)")
    server_args = ("--parallel", "1", *extra_flags)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Real hardware finding (2026-09-01): control and subject servers
    # must never run concurrently for this model -- each needs
    # ~13GB/GPU under -sm tensor, exceeding the 24.5GB/GPU cards
    # together (a real cudaMalloc OOM abort). One fresh server per
    # single bench-runner call, alternating arms, mirrors
    # run_rd73_mtp_server_lane()'s same fix.
    ports = {"control": control_port, "subject": subject_port}
    binaries = {"control": control_binary, "subject": subject_binary}
    request_counters = {"control": 0, "subject": 0}
    raw_metrics: dict[str, list[dict[str, float]]] = {"control": [], "subject": []}

    def _runner(command: list[str]) -> "experiment_execution.RunnerOutput":
        arm = command[-1]
        index = request_counters[arm]
        request_counters[arm] += 1
        session = AttestedServerSession(
            binary=binaries[arm],
            model=model,
            expected=expected_execution,
            host=host,
            port=ports[arm],
            extra_args=server_args,
            log_path=logs_dir / f"rd73-decode-{arm}-server-{index}.log",
            env_overrides=_hip_only(selector_env),
            env_unset=_ROCR_VISIBLE_DEVICES_UNSET,
        )
        with session:
            metrics = run_bench_runner_server_bench(
                server_url=f"http://{host}:{ports[arm]}",
                bench_configs="tg128",
                repetitions=1,
            )
        raw_metrics[arm].append(metrics)
        if "tg128_tps" not in metrics:
            raise PatchCampaignError(
                f"rd73 decode control lane ({arm}): bench runner produced no tg128_tps "
                f"metric (got {sorted(metrics)})"
            )
        return experiment_execution.RunnerOutput(
            returncode=0,
            stdout=f"BIGCHERRY_RD73_DECODE tg128_tps={metrics['tg128_tps']}\n",
            stderr="",
        )

    decode_run = experiment_execution.run_paired_lane(
        metric="tg128",
        control_command=["rd73-decode-lane", "control"],
        subject_command=["rd73-decode-lane", "subject"],
        pattern=metric_pattern,
        pairs=pairs,
        runner=_runner,
    )

    effect = experiment_execution.lane_effect_from_run("control", "tg128", decode_run)
    doc = {
        "metric": "tg128",
        "stats": decode_run.stats,
        "control_raw_metrics": raw_metrics["control"],
        "subject_raw_metrics": raw_metrics["subject"],
        "runs": list(decode_run.runs),
    }
    artifact_ref = _write_bound_artifact(run_dir, "rd73-decode-control.json", doc)
    return {"effect": effect, "artifact": artifact_ref, "stats": decode_run.stats}


def evaluate_rd73_resource_evidence(
    *,
    subject_log_path: Path,
    run_dir: Path,
) -> dict[str, object]:
    """VA06 (user redirect, 2026-09-01): RD73's real graph-cache-entries
    resource evidence, read from the subject llama-server LOG FILE
    run_rd73_mtp_server_lane() already produced
    (BIGCHERRY_RD73_RESOURCE_TRACE=1 is always set on that server) --
    no separate llama-bench probe. Subject-only (GPT's phase-1 scoping:
    the contract's resource_limits only bounds max_value, so no paired
    control reading is needed). Parses every real graph_cache_entries=N
    reading (parse_rd73_resource_telemetry(), fails closed on any
    malformed line) and reduces to a peak ResourceResult
    (peak_rd73_resource_result())."""
    subject_text = Path(subject_log_path).read_text(encoding="utf-8", errors="replace")
    readings = parse_rd73_resource_telemetry(subject_text)
    result = peak_rd73_resource_result(readings)
    subject_rel = Path(subject_log_path).relative_to(run_dir).as_posix()
    doc = {
        "readings": list(readings),
        "peak": result.subject_value,
        "artifact": {
            "path": subject_rel,
            "sha256": hashlib.sha256(Path(subject_log_path).read_bytes()).hexdigest(),
        },
    }
    artifact_ref = _write_bound_artifact(run_dir, "rd73-resource.json", doc)
    return {"result": result, "artifact": artifact_ref, "readings": readings}


def run_rd73_resource_burst_session(
    *,
    subject_binary: Path,
    model: Path,
    corpus_path: Path,
    run_dir: Path,
    expected_execution: ExecutionIdentity,
    host: str = "127.0.0.1",
    port: int = 18084,
    burst_requests: int = 20,
    n_predict: int = 32,
    selector_env: "dict[str, str] | None" = None,
) -> dict[str, object]:
    """VA06 (real hardware finding, 2026-09-01): RD73's graph-cache-entries
    resource evidence needs a real accumulated-cache burst -- repeated
    requests against ONE long-lived subject server (matching the
    contract's own documented methodology: "a fixed repeated-shape MTP
    completion burst", patches/1233.../README.md). This is NOT compatible
    with run_rd73_mtp_server_lane()'s per-request server restart (needed
    there to avoid a real control+subject concurrent-VRAM OOM): a fresh
    process resets the in-memory graph cache every single request, so
    that lane's own combined logs would only ever show a trivial
    peak (~1), never the real accumulated cache size the contract's
    max_value=800 bound was calibrated against (subject peak 651 under
    VA06's original characterization run). This session is subject-only
    (no concurrent control server), so it needs no restart discipline --
    launch once, drive burst_requests real repeated requests against the
    SAME live process, read the resulting log, shut down."""
    from bigcherry.bench import server_completion as sc

    prompts, _ = sc.load_corpus(corpus_path)
    burst_prompt = prompts[0]
    server_args = (
        "--parallel",
        "1",
        "--metrics",
        "-sm",
        "tensor",
        "--fit",
        "off",
        "--spec-type",
        "draft-mtp",
        "--spec-draft-n-max",
        "4",
    )
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "rd73-resource-burst-subject-server.log"
    rd73_env = {"BIGCHERRY_PATCH_TRACE": "1", "BIGCHERRY_RD73_RESOURCE_TRACE": "1"}
    if selector_env:
        rd73_env.update(selector_env)
    # GPT review follow-up (req_d1ef22d846854960, 2026-09-11): defense in
    # depth alongside env_unset=_ROCR_VISIBLE_DEVICES_UNSET below -- a
    # future direct caller passing ROCR_VISIBLE_DEVICES via selector_env
    # must not be able to reintroduce it through env_overrides ordering
    # (ServerRunner.launch() applies env_unset BEFORE env_overrides).
    rd73_env.pop("ROCR_VISIBLE_DEVICES", None)
    session = AttestedServerSession(
        binary=subject_binary,
        model=model,
        expected=expected_execution,
        host=host,
        port=port,
        extra_args=server_args,
        log_path=log_path,
        env_overrides=rd73_env,
        env_unset=_ROCR_VISIBLE_DEVICES_UNSET,
    )
    sampling = sc.SamplingConfig(temperature=1.0, top_p=0.95, top_k=20)
    config = sc.SessionConfig(
        session_id="rd73-resource-burst",
        corpus_id=corpus_path.stem,
        corpus_sha256="",
        bigcherry_revision="rd73-va06",
        llama_pin="",
        llama_revision="",
        model_id=str(model),
        server_argv=server_args,
        spec_type="draft-mtp",
        spec_n_max=4,
        spec_draft_k="default",
        spec_draft_v="default",
        sampling=sampling,
        n_predict=n_predict,
        order_seed=12345,
    )
    with session:
        transport = sc.HttpTransport(f"http://{host}:{port}")
        sc.validate_server(transport)
        for index in range(burst_requests):
            sc.run_request(
                transport, burst_prompt, config, pass_number=1, order_index=index
            )

    return evaluate_rd73_resource_evidence(subject_log_path=log_path, run_dir=run_dir)


class Rd73CorrectnessError(PatchCampaignError):
    """RD73's bit-identical correctness check found a real content
    mismatch -- distinct from PatchCampaignError's other, infrastructure-
    level failure modes only in name (still fails the campaign)."""


def evaluate_rd73_mtp_correctness(
    *,
    control_requests: list[dict[str, object]],
    subject_requests: list[dict[str, object]],
    run_dir: Path,
) -> dict[str, object]:
    """VA06 next slice: RD73's bit-identical correctness check, evaluated
    from run_rd73_mtp_server_lane()'s already-retained per-request
    ``content`` fields -- reuses the exact same real MTP requests already
    executed for the performance lane; never launches a second server
    lane just for correctness. Pairs control/subject requests by
    order_index (both arms ran the identical corpus/order, so index
    alignment is real pairing, not a coincidence) and requires EXACT
    string equality -- no trimming/normalization/tolerance. Fails closed
    on a mismatch, missing/non-string content, or an unpairable
    (differently-sized) record set; never silently skips a bad pair."""
    if len(control_requests) != len(subject_requests):
        raise Rd73CorrectnessError(
            f"rd73 correctness: control has {len(control_requests)} request(s) but subject "
            f"has {len(subject_requests)} -- cannot pair records for comparison"
        )
    rows: list[dict[str, object]] = []
    mismatches: list[str] = []
    for control_record, subject_record in zip(control_requests, subject_requests):
        control_index = control_record.get("order_index")
        subject_index = subject_record.get("order_index")
        if control_index != subject_index:
            raise Rd73CorrectnessError(
                f"rd73 correctness: control/subject request order_index mismatch "
                f"({control_index!r} vs {subject_index!r}) -- records are not aligned"
            )
        control_content = control_record.get("content")
        subject_content = subject_record.get("content")
        if not isinstance(control_content, str) or not isinstance(subject_content, str):
            raise Rd73CorrectnessError(
                f"rd73 correctness: request order_index={control_index!r} has non-string "
                f"content (control={type(control_content).__name__}, "
                f"subject={type(subject_content).__name__}) -- cannot compare"
            )
        ok = control_content == subject_content
        if not ok:
            mismatches.append(f"order_index={control_index!r}")
        rows.append({"order_index": control_index, "ok": ok})
    if mismatches:
        raise Rd73CorrectnessError(
            f"rd73 correctness: {len(mismatches)} request(s) mismatched: {', '.join(mismatches)}"
        )
    # VA23: "ops" is what _builtin_backend_ops() matches against the check's
    # declared config. patch 1233's validation.toml declares
    # ops = ["RD73_MTP_BIT_IDENTICAL"] for its correctness check, so the
    # producer must emit that exact identifier or the validator cannot tell
    # this artifact apart from any other correctness evidence. "passed" stays
    # the real comparison outcome; only the identifier is added.
    doc = {
        "check": "bit_identical",
        "passed": True,
        "rows": rows,
        "ops": ["RD73_MTP_BIT_IDENTICAL"],
    }
    artifact_ref = _write_bound_artifact(run_dir, "rd73-correctness.json", doc)
    return {"artifact": artifact_ref, "rows": rows}


def run_rd73_contract_qualification(
    *,
    contract: object,
    control_server_binary: Path,
    subject_server_binary: Path,
    model: Path,
    marker_regex: str,
    corpus_path: Path,
    run_dir: Path,
    # VA24: decode_pairs raised 3 -> 10 to match measured_pairs. min_paired_rounds
    # is a minimum VALIDITY requirement for every interval used to establish an
    # acceptance bound, so a control decision taken on 3 rounds violates exactly
    # what a contract declaring 10 claims. The previous asymmetry meant RD73
    # demanded 10 rounds to prove its own gain while accepting 3 to prove it had
    # broken nothing. "Controls need less evidence" is not defensible as a
    # general rule: required sample size depends on variance, distance from the
    # acceptance boundary, and the estimator -- not on lane role (dev-gpt-agent,
    # req_875d13b29a204075). Costs ~5 extra minutes on a ~15-minute
    # qualification, measured.
    decode_pairs: int = 10,
    warmup_pairs: int = 2,
    measured_pairs: int = 10,
    # RV99: the patch's already-committed validation records, each a prior
    # measurement SESSION. Required, not defaulted: under a session policy a
    # caller that forgets them silently under-counts sessions and the gate
    # reports "collect more" for ever. An empty tuple is the honest value for
    # a first session, and is meaningless under a non-session policy.
    prior_session_records: "Iterable[Mapping[str, object]]",
    # Required: session aggregation must pool only same-hardware sessions,
    # and this is the run's own architecture.
    amdgpu_targets: str,
) -> dict[str, object]:
    """VA06 next slice: the authoritative RD73 full-qualification path
    (``--run-rd73-contract``), mirroring RD08's own
    run_rd08_contract_qualification() result/schema/promotion semantics
    (real lane execution + real correctness + real trigger proof, composed
    via evaluate_promotion_gate()) -- no RD73-specific parallel gate model.
    Every threshold comes from ``contract`` itself
    (aggregate_contract_effects() / evaluate_resource_gate() /
    evaluate_promotion_gate()); nothing here hardcodes a number.

    User redirect (2026-09-01): every lane now runs entirely over real
    llama-server processes driven by HTTP requests / the documented
    Brutus bench runner -- never a raw llama-bench subprocess, which
    proved unworkable for RD73's real 27B/dual-GPU/-sm-tensor config on
    real hardware. The MTP performance lane runs first; its own
    control/subject server log files (BIGCHERRY_PATCH_TRACE=1 is always
    set on them) are the real source for activation evidence. Resource
    evidence uses a SEPARATE subject-only burst session
    (run_rd73_resource_burst_session()): the MTP lane restarts a fresh
    server per single request (a real hardware constraint -- control and
    subject cannot run concurrently, each needs ~13GB/GPU and two copies
    exceed the 24.5GB/GPU cards), which resets the in-memory graph cache
    every request and would make a peak reading trivial/meaningless;
    the burst session's one long-lived process gets the real
    accumulated-cache reading the contract's resource bound needs.
    Decode control similarly launches its own control/subject servers
    one-request-at-a-time (never concurrently) for the same VRAM
    reason."""
    from bigcherry.experiment import contract as experiment_contract
    from bigcherry.experiment.execution import (
        DeviceVisibilityError as _DeviceVisibilityError,
        require_device_visibility as _require_device_visibility,
    )

    # VA25: RD73's dual-XTX qualification is always `-sm tensor` across 2
    # homogeneous devices of this run's own architecture -- the exact
    # topology every RD73 server lane's docstring already documents as a
    # real hardware constraint (concurrent control+subject exceeds VRAM,
    # etc.). Constructed once here, not per-lane, so all three lanes
    # attest against the identical expectation.
    expected_execution = ExecutionIdentity(
        backend="ROCm",
        architectures=(amdgpu_targets, amdgpu_targets),
    )

    # PVPS02 step 7 (2026-09-11): validate the real HIP_VISIBLE_DEVICES
    # selector ONCE here (the same fail-closed contract RD58/the generic
    # matrix use), then copy that validated value into every lane's own
    # server-session env overrides below -- ServerRunner starts each
    # server from ambient env then applies overrides, so this makes the
    # selector explicit for every process this qualification launches
    # without ever mutating global os.environ. ExecutionIdentity above
    # remains the separate architecture/device-count attestation this
    # was already doing; selector validation and attestation are
    # deliberately two different checks, not merged into one.
    #
    # Real-hardware finding (2026-09-11, PNRO17): only HIP_VISIBLE_DEVICES
    # is set here -- also setting ROCR_VISIBLE_DEVICES to the same value
    # actively breaks non-prefix-from-0 device selection (confirmed on
    # real gfx1100/gfx1201/gfx1030 hardware; see require_device_visibility()
    # and DeviceVisibility's own docstrings for the full mechanism).
    try:
        rd73_visibility = _require_device_visibility(
            context=f"--run-rd73-contract ({amdgpu_targets})",
            exact_count=2,
        )
    except _DeviceVisibilityError as exc:
        raise PatchCampaignError(str(exc)) from exc
    rd73_selector_env = {"HIP_VISIBLE_DEVICES": rd73_visibility.hip_visible_devices}

    mtp = run_rd73_mtp_server_lane(
        control_binary=control_server_binary,
        subject_binary=subject_server_binary,
        model=model,
        corpus_path=corpus_path,
        run_dir=run_dir,
        expected_execution=expected_execution,
        warmup_pairs=warmup_pairs,
        measured_pairs=measured_pairs,
        selector_env=rd73_selector_env,
    )
    activation = evaluate_rd73_activation_evidence(
        marker_regex=marker_regex,
        control_log_path=mtp["control_log_path"],
        subject_log_path=mtp["subject_log_path"],
        run_dir=run_dir,
    )
    resource = run_rd73_resource_burst_session(
        subject_binary=subject_server_binary,
        model=model,
        corpus_path=corpus_path,
        run_dir=run_dir,
        expected_execution=expected_execution,
        selector_env=rd73_selector_env,
    )
    decode_control = run_rd73_decode_control_lane(
        control_binary=control_server_binary,
        subject_binary=subject_server_binary,
        model=model,
        run_dir=run_dir,
        expected_execution=expected_execution,
        pairs=decode_pairs,
        selector_env=rd73_selector_env,
    )
    # A real content mismatch (or a missing/non-string/unpaired record) is a
    # genuine correctness RESULT, not an infrastructure failure -- it must
    # flow into correctness_gate/promotion as passed=False, never abort the
    # whole qualification run (mirrors RD08's Rd08CorrectnessError handling
    # in run_rd08_contract_correctness()).
    try:
        correctness = evaluate_rd73_mtp_correctness(
            control_requests=mtp["control_requests"],
            subject_requests=mtp["subject_requests"],
            run_dir=run_dir,
        )
        correctness_result = experiment_contract.CorrectnessResult(
            check="bit_identical", passed=True
        )
    except Rd73CorrectnessError as exc:
        correctness = {"artifact": None, "rows": [], "error": str(exc)}
        correctness_result = experiment_contract.CorrectnessResult(
            check="bit_identical",
            passed=False,
            detail=str(exc),
        )
    correctness_gate = compute_contract_correctness_gate(
        contract, {"bit_identical": correctness_result}
    )
    aggregated_effects = experiment_contract.aggregate_contract_effects(
        contract,
        [mtp["effect"], decode_control["effect"]],
        target_metric="mtp_wall_tps",
    )
    # RV99: under a session policy the gain bound is established across
    # repeated SESSIONS, not from the pairs inside this one run. Fold the
    # prior sessions' persisted lane effects together with the one just
    # measured and re-aggregate over all of them.
    #
    # prior_session_records are the patch's already-committed validation
    # records; this run's own measurement is appended last, so the gate always
    # sees every valid session including the current one. Nothing is selected
    # or dropped -- aggregate_session_effects() consumes them all, and the
    # stopping rule decides whether that is yet enough.
    #
    # Only the gain field is re-aggregated. The control-regression budget is a
    # per-run property (this build must not have broken the control lane in
    # THIS run), not a claim being established across occasions.
    if contract.acceptance.effect_evidence_policy == "session_ci95_threshold_bound_v1":
        # The stub must carry gpu_architectures like a real record does, or
        # aggregate_session_effects' hardware filter would drop the very
        # session just measured.
        this_session = {
            "gpu_architectures": [amdgpu_targets],
            "lane_effects": collect_lane_effect_records(
                rd08_qualification=None,
                rd73_qualification={"mtp": mtp, "decode_control": decode_control},
            ),
        }
        gain_field = (
            "end_to_end_gain_pct"
            if contract.acceptance.end_to_end_gain_pct is not None
            else "target_kernel_gain_pct"
        )
        aggregated_effects = dict(aggregated_effects)
        aggregated_effects.update(
            experiment_contract.aggregate_session_effects(
                [*prior_session_records, this_session],
                field=gain_field,
                role="positive",
                metric="mtp_wall_tps",
                # Only sessions measured on THIS hardware may be pooled.
                architectures=[amdgpu_targets],
            )
        )
    resource_gate = experiment_contract.evaluate_resource_gate(
        contract,
        {"graph_cache_entries": resource["result"]},
    )
    trigger_proof = experiment_contract.evaluate_trigger_proof(
        [
            experiment_contract.TriggerEvidence(
                role="positive",
                lane_id="rd73-mtp-subject",
                candidate_launches=1 if activation["subject_hit"] else 0,
            ),
        ]
    )
    if activation["control_hit"]:
        trigger_proof = {
            "passed": False,
            "reasons": list(trigger_proof.get("reasons") or [])
            + [
                "control-role lane observed the target marker -- the negative "
                "control is invalid, so trigger proof cannot be trusted"
            ],
            "checked_lanes": trigger_proof.get("checked_lanes", 0),
            "untriggered_lanes": list(trigger_proof.get("untriggered_lanes") or []),
        }
    promotion = experiment_contract.evaluate_promotion_gate(
        contract,
        correctness_gate=correctness_gate,
        aggregated_effects=aggregated_effects,
        trigger_proof=trigger_proof,
        resource_gate=resource_gate,
    )
    qualification_doc = {
        "contract_id": contract.id,
        "contract_hash": contract.contract_hash,
        "activation_artifact": activation["artifact"],
        "mtp_lane_artifact": mtp["artifact"],
        "decode_control_artifact": decode_control["artifact"],
        "resource_artifact": resource["artifact"],
        "correctness_artifact": correctness["artifact"],
        "correctness_gate": correctness_gate,
        "aggregated_effects": aggregated_effects,
        "resource_gate": resource_gate,
        "trigger_proof": trigger_proof,
        "promotion": promotion,
    }
    artifact_ref = _write_bound_artifact(
        run_dir, "rd73-contract-qualification.json", qualification_doc
    )

    # VA23: emit the generic-adapter performance artifact.
    #
    # _builtin_benchmark() (patch/validation.py) requires an artifact with a
    # non-empty "metrics" dict; without one the declared performance/controls
    # checks ERROR with "benchmark artifact requires non-empty metrics", and
    # patch-verify-evidence then reports "no recorded benchmark execution"
    # even though a real, paired, bootstrapped benchmark demonstrably ran.
    # That is a false negative in the direction that HIDES real results.
    #
    # This is a faithful projection of already-measured values into the
    # schema the generic validator reads -- the same thing RD58's producer does via its controls_doc. Nothing here is
    # computed for the first time, and nothing is invented: every number
    # below is copied from the lane effects the contract gate itself just
    # consumed. The lane artifacts remain the authoritative record and stay
    # separately hash-bound; this document references them rather than
    # replacing them.
    performance_doc = {
        "campaign_id": contract.contract_hash,
        "passed": bool(promotion.get("passed")),
        "contract_id": contract.id,
        "target_metric": "mtp_wall_tps",
        "metrics": {
            # LaneEffect is a frozen dataclass (experiment/contract.py);
            # dataclasses.asdict() is its faithful serialisation, so the
            # recorded fields are exactly the measured
            # role/metric/geometric_effect_pct/decision the contract gate
            # consumed -- not a re-derivation.
            "mtp_verify": {
                "effect": dataclasses.asdict(mtp["effect"]),
                "artifact": mtp["artifact"],
            },
            "decode_control": {
                "effect": dataclasses.asdict(decode_control["effect"]),
                "artifact": decode_control["artifact"],
            },
            "aggregated_effects": aggregated_effects,
        },
        "promotion": promotion,
    }
    performance_artifact = _write_bound_artifact(
        run_dir,
        "rd73-performance.json",
        performance_doc,
    )
    return {
        "performance_artifact": performance_artifact,
        "activation": activation,
        "mtp": mtp,
        "decode_control": decode_control,
        "resource": resource,
        "correctness": correctness,
        "correctness_gate": correctness_gate,
        "aggregated_effects": aggregated_effects,
        # VA23: the NAMED correctness result, so the adapter's own
        # _contract_correctness_gate can be fed the same way RD08's and
        # RD58's are (see the compute_contract_correctness_gate() call in
        # run()). Without this RD73 passes None there and the gate reports
        # missing_checks -> BLOCKED, even though bit_identical was really
        # evaluated here. Returned as the CorrectnessResult itself, not a
        # bool, so a failure carries its detail through unchanged.
        "correctness_named_results": {"bit_identical": correctness_result},
        "resource_gate": resource_gate,
        "trigger_proof": trigger_proof,
        "promotion": promotion,
        "artifact": artifact_ref,
    }


def _run_framework_configuration(args: argparse.Namespace, descriptor, cfg) -> int:
    """Build the canonical native framework composition and persist schema-5 proof."""
    from bigcherry.build import generated_tree
    from bigcherry.patch import evidence as patch_validation_evidence
    from bigcherry.patch import source as psi
    from bigcherry.patch import validation_policy
    from bigcherry.patch import validation
    from bigcherry.core import paths as bc_paths

    if not validation_policy.is_framework_configuration_patch(descriptor):
        raise PatchCampaignError(
            "--framework-configuration requires a local packaged framework patch without an RD/contract binding"
        )
    if any(
        getattr(args, name, False)
        for name in (
            "run_rd08_lanes",
            "run_rd08_contract",
            "run_rd73_contract",
            "correctness_evidence",
        )
    ):
        raise PatchCampaignError(
            "framework configuration cannot be combined with runtime qualification modes"
        )
    if args.amdgpu_targets is None:
        raise PatchCampaignError(
            "framework configuration requires explicit AMDGPU compile targets"
        )
    targets = tuple(
        target.strip()
        for target in re.split(r"[,;]", args.amdgpu_targets)
        if target.strip()
    )
    if not targets or any(
        not re.fullmatch(r"gfx[0-9a-f]+", target) for target in targets
    ):
        raise PatchCampaignError(
            "framework configuration requires explicit AMDGPU compile targets"
        )
    args.amdgpu_targets = ";".join(targets)
    from bigcherry.core.context import ProjectContext

    base_repo = ProjectContext.resolve(
        work_root=os.environ.get("BC_CACHE")
    ).upstream_repo
    baseline_source = "bigcherry-qualification-tuning"
    base_revision, composition = psi.resolve_source_composition(
        baseline_source,
        focal=None,
        base_ref=cfg.pinned,
        base_repo=base_repo,
    )
    if (descriptor.patch_id, descriptor.implementation_digest) not in composition:
        raise PatchCampaignError(
            f"framework source {baseline_source!r} does not contain focal patch {descriptor.patch_id!r}"
        )
    source = psi.materialize_composition(
        base_repo=base_repo,
        worktree_root=args.worktree_root / "framework",
        resolved_revision=base_revision,
        composition=composition,
        overlay_root=psi.REPO_ROOT / "src",
        requested_revision=cfg.pinned,
    )
    idempotent = psi.verify_composition_idempotent(
        base_repo=base_repo,
        source=source,
        worktree_root=args.worktree_root / "framework",
        resolved_revision=base_revision,
        composition=composition,
        overlay_root=psi.REPO_ROOT / "src",
        requested_revision=cfg.pinned,
    )
    if not idempotent:
        raise PatchCampaignError("framework composition did not reapply idempotently")
    source_tree = psi.git_worktree_tree(source)
    source_manifest = psi._read_manifest(source)
    if not source_manifest or source_manifest.get("source_tree_oid") != source_tree:
        raise PatchCampaignError("framework source attestation is missing or stale")
    source_identity = psi._make_source_identity_v2(
        resolved_revision=base_revision,
        composition=composition,
        overlay_root=psi.REPO_ROOT / "src",
    )
    source_identity["materialization_plan_id"] = source_identity["source_key"]
    if any(source_manifest.get(key) != value for key, value in source_identity.items()):
        raise PatchCampaignError("framework materialization identity is stale")
    build_root = (args.build_root or args.workdir) / source.name
    # Qualification owns fresh directories, never retroactively attests a
    # historical build whose inputs were not observed during compilation.
    for role in ("production", "diagnostic"):
        if (build_root / f"framework-{role}").exists():
            raise PatchCampaignError(
                "framework qualification requires a fresh build-root; preserve the previous run"
            )
    generated_dir = build_root / "generated"
    generate_registry(
        source=source, amdgpu_targets=args.amdgpu_targets, generated_dir=generated_dir
    )
    # Same four compile inputs returned by catalog.emit().compile_input_paths;
    # JSON manifests contain timestamps and are not compiler inputs.
    compile_inputs = tuple(
        generated_dir / name
        for name in (
            "hip-autotune-registry.inc",
            "hip-autotune-build-hash.h",
            "hip-autotune-arch.h",
            "hip-autotune-mmvq-instances.inc",
        )
    )
    missing = [str(path) for path in compile_inputs if not path.is_file()]
    if missing:
        raise PatchCampaignError(f"generated compiler inputs missing: {missing}")
    generated_manifest = generated_tree.build_manifest(
        generated_dir, compile_inputs=compile_inputs
    )
    proof = {}

    def generated_proof(phase, build_dir):
        compiled_copy = build_dir / "generated-inputs"
        if psi.git_worktree_tree(source) != source_tree:
            raise PatchCampaignError(f"source changed at {phase}")
        generated_tree.verify_tree(generated_dir, generated_manifest)
        generated_tree.verify_tree(compiled_copy, generated_manifest)
        copied_manifest = generated_tree.build_manifest(
            compiled_copy,
            compile_inputs=tuple(
                compiled_copy / name for name in generated_manifest["compile_inputs"]
            ),
        )
        if (
            copied_manifest["compile_inputs_hash"]
            != generated_manifest["compile_inputs_hash"]
        ):
            raise PatchCampaignError(
                f"{build_dir.name}: compiled-copy input hash disagrees with generated manifest"
            )
        proof[build_dir.name] = copied_manifest

    import shutil

    for role in ("production", "diagnostic"):
        shutil.copytree(
            generated_dir, build_root / f"framework-{role}" / "generated-inputs"
        )
    common = [
        "-DGGML_HIP_RCCL=ON",
        "-DGGML_HIP_DISPATCH_REPLAY=ON",
        "-DGGML_HIP_AUTOTUNE=OFF",
        "-DGGML_HIP_AUTOTUNE_RECORD=OFF",
        "-DGGML_HIP_REPLAY_DIAGNOSTICS=OFF",
    ]
    production_args = common + [
        "-DGGML_HIP_DISPATCH_DIAGNOSTICS=OFF",
        f"-DGGML_HIP_AUTOTUNE_GENERATED_DIR={build_root / 'framework-production' / 'generated-inputs'}",
    ]
    diagnostic_args = common + [
        "-DGGML_HIP_DISPATCH_DIAGNOSTICS=ON",
        f"-DGGML_HIP_AUTOTUNE_GENERATED_DIR={build_root / 'framework-diagnostic' / 'generated-inputs'}",
    ]
    production_bin = build_tree(
        name="framework-production",
        hip_path=args.hip_path,
        amdgpu_targets=args.amdgpu_targets,
        workdir=build_root,
        targets=["llama-server"],
        source=source,
        extra_cmake_args=production_args,
        generated_proof_callback=generated_proof,
    )
    diagnostic_bin = build_tree(
        name="framework-diagnostic",
        hip_path=args.hip_path,
        amdgpu_targets=args.amdgpu_targets,
        workdir=build_root,
        targets=["llama-server"],
        source=source,
        extra_cmake_args=diagnostic_args,
        generated_proof_callback=generated_proof,
    )
    env = _hip_env(args.hip_path)
    exe = ".exe" if sys.platform == "win32" else ""
    production = capture_completed_build_evidence(
        build_root / "framework-production",
        source_root=source,
        architecture=args.amdgpu_targets,
        binary=production_bin / f"llama-server{exe}",
        requested_cmake_args=_full_requested_cmake_args(
            hip_path=args.hip_path,
            amdgpu_targets=args.amdgpu_targets,
            extra_cmake_args=production_args,
        ),
        build_env=env,
    )
    diagnostic = capture_completed_build_evidence(
        build_root / "framework-diagnostic",
        source_root=source,
        architecture=args.amdgpu_targets,
        binary=diagnostic_bin / f"llama-server{exe}",
        requested_cmake_args=_full_requested_cmake_args(
            hip_path=args.hip_path,
            amdgpu_targets=args.amdgpu_targets,
            extra_cmake_args=diagnostic_args,
        ),
        build_env=env,
    )
    from bigcherry.build.builds import inspect_dispatch_build

    compiler_observations = {}
    for role, diagnostic_on in (("production", False), ("diagnostic", True)):
        observed = inspect_dispatch_build(build_root / f"framework-{role}")
        counts = observed["compiled_definition_counts"]
        if (
            observed["issues"]
            or bool(counts["GGML_HIP_DISPATCH_DIAGNOSTICS"]) != diagnostic_on
        ):
            raise PatchCampaignError(
                f"{role} diagnostic compiler state disagrees with qualification role"
            )
        compiler_observations[role] = {
            key: observed[key]
            for key in (
                "hip_compile_command_count",
                "compiled_definition_counts",
                "coverage_translation_unit",
                "issues",
            )
        }
    run_dir = args.workdir / "framework" / descriptor.patch_id
    run_dir.mkdir(parents=True, exist_ok=False)
    generated_artifact = _write_bound_artifact(
        run_dir, "generated-tree.json", generated_manifest
    )
    source_artifact = _write_bound_artifact(
        run_dir, "source-tree.json", source_manifest
    )
    builds = {
        "production": production.campaign_identity(),
        "diagnostic": diagnostic.campaign_identity(),
    }
    for role in builds:
        compiler_observations[role]["build_identity"] = builds[role]
    build_artifacts = {
        role: _write_bound_artifact(
            run_dir,
            f"{role}-build.json",
            {
                **completed.to_dict(),
                "generated_inputs_verification": "compiled-copy-v1",
                "generated_inputs": proof[f"framework-{role}"],
                "source_slice_id": source_manifest["source_slice_id"],
                "source_tree": source_tree,
            },
        )
        for role, completed in (("production", production), ("diagnostic", diagnostic))
    }
    plan = validation_policy.require_execution_package(
        descriptor, root=bc_paths.PATCHES
    )
    ctx = validation.ValidationContext(
        descriptor=descriptor,
        base_revision=base_revision,
        control_source=None,
        subject_source=None,
        package_root=bc_paths.PATCHES / descriptor.package_root,
        run_dir=run_dir,
        register_artifact=validation.make_default_register_artifact(run_dir),
        configuration_evidence={
            "apply": {
                "single_composition": True,
                "verified": True,
                "idempotent": idempotent,
                "artifact": source_artifact,
            },
            "builds": {
                role: {"completed": True, "artifact": artifact}
                for role, artifact in build_artifacts.items()
            },
        },
    )
    results = {
        spec.check_id: validation.evaluate_check(spec, ctx) for spec in plan.checks
    }
    verdict = validation.compute_verdict(plan, results)
    _print(f"adapter eligible: {verdict.eligible}")
    for check_id, result in results.items():
        _print(f"{check_id}: {result.status}: {result.summary}")
    checks = {name: asdict(result) for name, result in results.items()}
    artifacts = {
        artifact.path: artifact.sha256
        for result in results.values()
        for artifact in result.artifacts
    }
    artifacts[generated_artifact["path"]] = generated_artifact["sha256"]
    record = patch_validation_evidence.make_framework_configuration_record(
        descriptor=descriptor,
        patch_path=bc_paths.PATCHES / descriptor.implementation_path,
        base_ref=cfg.pinned,
        base_revision=base_revision,
        source_name=baseline_source,
        source_composition=composition,
        source_tree=source_tree,
        source_slice_id=source_manifest["source_slice_id"],
        compiled_targets=tuple(
            target.strip()
            for target in re.split(r"[,;]", args.amdgpu_targets)
            if target.strip()
        ),
        builds=builds,
        source_identity=source_identity,
        compiler_observations=compiler_observations,
        generated_inputs={
            role: {
                "proof": "compiled-copy-v1",
                "compile_inputs_hash": proof[f"framework-{role}"][
                    "compile_inputs_hash"
                ],
                "tree_manifest": proof[f"framework-{role}"],
                "build_identity": builds[role],
            }
            for role in builds
        },
        check_results=checks,
        artifact_hashes=artifacts,
        campaign_workdir=run_dir,
    )
    path = patch_validation_evidence.write_record(record)
    _print(f"framework configuration evidence: {path}")
    return 0 if record["eligible_for_validated_state"] else 1


def _prepare_standard_campaign(args: argparse.Namespace, st: SimpleNamespace) -> None:
    """PA43 run() stage. standard-campaign stage: materialize the five standard builds (via the shared
    scaffold), bind the campaign identity and construct the S1-S7 Campaign."""
    cfg = st.cfg
    psi = st.psi
    workdir = st.workdir
    worktree_root: Path = args.worktree_root
    # RV80/B6: the baseline is the source's EXPLICIT named composition from
    # config/recipes.toml (never the retired implicit state=='validated'
    # scan), resolved through the exact-composition validator; the base ref
    # resolves to an immutable SHA that enters the v2 source identity.
    baseline_source = getattr(args, "baseline_source", "bigcherry")
    # PA36 sub-slice 2 (dev-gpt-agent req_2ecda033763949a9, T2): the five
    # standard-campaign builds (tune -> replay -> stock -> control ->
    # validation-subject) live in _build_standard_campaign_scaffold(),
    # shared with the generic standard_campaign="run" producer path. Local
    # aliases below keep the rest of run() structurally unchanged.
    scaffold = _build_standard_campaign_scaffold(
        patch_id=args.patch,
        base_ref=cfg.pinned,
        baseline_source=baseline_source,
        hip_path=args.hip_path,
        amdgpu_targets=args.amdgpu_targets,
        workdir=workdir,
        worktree_root=worktree_root,
        build_root=args.build_root,
    )
    base_revision = scaffold.base_revision
    control_composition = scaffold.control_composition
    subject_composition = scaffold.subject_composition
    control_src = scaffold.control_source
    patched_src = scaffold.subject_source
    stock_src = scaffold.stock_source
    control_idempotent = scaffold.control_idempotent
    subject_idempotent = scaffold.subject_idempotent
    build_root = scaffold.build_root
    build_env = scaffold.build_env
    tune_bin = scaffold.tune_bin
    replay_bin = scaffold.replay_bin
    stock_bin = scaffold.stock_bin
    control_bin = scaffold.control_bin
    validation_subject_bin = scaffold.validation_subject_bin
    tune_build_evidence = scaffold.tune_build_evidence
    replay_build_evidence = scaffold.replay_build_evidence
    stock_build_evidence = scaffold.stock_build_evidence
    control_build_evidence = scaffold.control_build_evidence
    validation_subject_build_evidence = scaffold.validation_subject_build_evidence
    exe = ".exe" if sys.platform == "win32" else ""

    from bigcherry.e2e_smoke_campaign import (  # noqa: E402
        Campaign,
        CampaignError,
        CampaignIdentityContext,
    )

    # Hoisted: HI83's evidence record (below) needs the same values.
    patch_digest = psi.patch_implementation_digest(args.patch)
    control_source_tree = psi.git_worktree_tree(control_src)
    patched_source_tree = psi.git_worktree_tree(patched_src)

    identity_context = CampaignIdentityContext(
        patch_name=args.patch,
        patch_digest=patch_digest,
        patched_source_tree=patched_source_tree,
        gpu_architecture=args.amdgpu_targets,
        build_identities={
            "tune": tune_build_evidence.campaign_identity(),
            "replay": replay_build_evidence.campaign_identity(),
            "stock": stock_build_evidence.campaign_identity(),
        },
    )

    campaign = Campaign(
        model=args.model,
        tune_server=tune_bin / f"llama-server{exe}",
        replay_server=replay_bin / f"llama-server{exe}",
        manifest=args.manifest,
        workdir=workdir / "campaign",
        stock_bench=stock_bin / f"llama-bench{exe}",
        tune_bench=tune_bin / f"llama-bench{exe}",
        replay_bench=replay_bin / f"llama-bench{exe}",
        bench_prompt=args.bench_prompt,
        bench_gen=args.bench_gen,
        bench_repetitions=args.bench_repetitions,
        identity_context=identity_context,
    )

    # Bind the workdir before writing any patch-specific activation
    # evidence. campaign.run() will check it again; this earlier call
    # prevents a trace probe from writing evidence into a stale/mismatched
    # campaign directory.
    campaign.ensure_campaign_identity()
    st.CampaignError = CampaignError
    st.base_revision = base_revision
    st.baseline_source = baseline_source
    st.campaign = campaign
    st.control_bin = control_bin
    st.control_build_evidence = control_build_evidence
    st.control_composition = control_composition
    st.control_idempotent = control_idempotent
    st.control_source_tree = control_source_tree
    st.control_src = control_src
    st.exe = exe
    st.identity_context = identity_context
    st.patch_digest = patch_digest
    st.patched_source_tree = patched_source_tree
    st.patched_src = patched_src
    st.stock_src = stock_src
    st.subject_composition = subject_composition
    st.subject_idempotent = subject_idempotent
    st.tune_bin = tune_bin
    st.validation_subject_bin = validation_subject_bin
    st.validation_subject_build_evidence = validation_subject_build_evidence


def _run_activation_probe_stage(args: argparse.Namespace, st: SimpleNamespace) -> None:
    """PA43 run() stage. Activation stage: resolve the trace-marker check and run the generic
    positive/negative trace probe, binding its logs as trace evidence."""
    campaign = st.campaign
    exe = st.exe
    tune_bin = st.tune_bin
    validation_plan = st.validation_plan
    workdir = st.workdir
    activation_evidence = None
    activation_verdict = None
    trace_marker_regex = args.trace_marker_regex
    trace_description = args.trace_description
    if validation_plan is not None:
        trace_specs = tuple(
            spec
            for spec in validation_plan.checks
            if spec.capability == "activation" and spec.validator == "trace-marker"
        )
        if len(trace_specs) > 1:
            raise PatchCampaignError(
                f"{args.patch}: validation plan declares multiple trace-marker activation checks"
            )
        if trace_specs:
            configured_marker = trace_specs[0].config.get("marker-regex")
            if not isinstance(configured_marker, str) or not configured_marker:
                raise PatchCampaignError(
                    f"{args.patch}: trace-marker activation check has no marker-regex"
                )
            if (
                trace_marker_regex is not None
                and trace_marker_regex != configured_marker
            ):
                raise PatchCampaignError(
                    f"{args.patch}: CLI trace marker conflicts with validation.toml"
                )
            trace_marker_regex = configured_marker
            trace_description = trace_description or f"{args.patch} activation"
        elif trace_marker_regex is not None or trace_description is not None:
            raise PatchCampaignError(
                f"{args.patch}: trace CLI options require a trace-marker validation check"
            )
    # GPT round 6 (req_bc329f6ae30c4e4c, VA15 real-hardware finding): the
    # generic tune-binary/fusion-disabled probe is redundant for
    # --run-rd08-contract -- it is replaced by RD08's own authoritative
    # validation-subject/control trigger probe below, which is a valid
    # negative control for RD08's specific marker (the generic probe's
    # negative control, GGML_CUDA_DISABLE_FUSION=1, is not). Skipping it
    # here also avoids wasted GPU time on a probe whose result gets
    # overwritten anyway.
    # VA06: --run-rd73-contract also skips the generic probe -- the
    # generic tune-binary/GGML_CUDA_DISABLE_FUSION negative control is
    # not valid for RD73 (graph-cache keying, not a fusion path), and
    # the generic probe's plain llama-bench invocation (no -sm tensor)
    # cannot even load RD73's real 27B contract model on Brutus's dual
    # gfx1100 GPUs. RD73's own authoritative activation evidence comes
    # from evaluate_rd73_activation_evidence() inside
    # run_rd73_contract_qualification().
    trace_result = (
        None
        if (args.run_rd08_contract or args.run_rd73_contract)
        else run_trace_activation_probes(
            marker_regex=trace_marker_regex,
            description=trace_description,
            binary=tune_bin / f"llama-bench{exe}",
            model=args.model,
            hip_path=args.hip_path,
            workdir=workdir / "campaign",
            bench_prompt=args.bench_prompt,
            bench_gen=args.bench_gen,
        )
    )
    # VA11A: real bound trace_evidence for ValidationContext (was {}
    # unconditionally, which made _builtin_trace_marker always BLOCKED --
    # GPT round-7 review, req_3d12aa6668b14bb1). Built from the real
    # positive/negative probe logs run_trace_activation_probes() already
    # wrote to workdir/"campaign"/"logs"/... -- the same directory
    # ValidationContext.run_dir (campaign_run_dir) resolves bound artifacts
    # against, so no copy is needed, only a path+sha256 reference.
    trace_evidence: dict[str, object] = {}
    if trace_result is not None:
        activation_evidence, trace_detail = trace_result
        # This stage establishes activation evidence only -- the existing
        # patch_activation verdict contract accepts correctness_passed=None;
        # a later patch-class-specific correctness check can strengthen
        # this without changing the trace-probe mechanism.
        activation_verdict = verdict(activation_evidence, correctness_passed=None)
        write_activation_json(
            workdir / "campaign" / "activation.json",
            activation_evidence,
            activation_verdict,
            extra={
                "campaign_identity_digest": campaign.campaign_identity_digest,
                "trace_probe": trace_detail,
            },
        )
        _print(
            f"activation: {activation_evidence.status} ({activation_evidence.mechanism})"
        )

        def _bind_existing(relative_log_path: str) -> dict[str, str]:
            target = (workdir / "campaign" / relative_log_path).resolve()
            return {
                "path": relative_log_path,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }

        trace_evidence = {
            "positive": {
                "marker_regex": trace_detail["marker_regex"],
                "artifact": _bind_existing(trace_detail["positive"]["log"]),
            },
            "negative": {
                "marker_regex": trace_detail["marker_regex"],
                "artifact": _bind_existing(trace_detail["negative_control"]["log"]),
            },
        }
    st.activation_evidence = activation_evidence
    st.activation_verdict = activation_verdict
    st.trace_evidence = trace_evidence
    st.trace_marker_regex = trace_marker_regex


def _collect_build_and_correctness_evidence(args: argparse.Namespace, st: SimpleNamespace) -> None:
    """PA43 run() stage. Bind correctness, build and apply evidence for the validation record."""
    base_revision = st.base_revision
    campaign = st.campaign
    control_build_evidence = st.control_build_evidence
    control_composition = st.control_composition
    control_idempotent = st.control_idempotent
    control_source_tree = st.control_source_tree
    descriptor = st.descriptor
    patched_source_tree = st.patched_source_tree
    registry = st.registry
    subject_composition = st.subject_composition
    subject_idempotent = st.subject_idempotent
    validation_subject_build_evidence = st.validation_subject_build_evidence
    workdir = st.workdir
    # HI83: record what this campaign proved (or didn't), tracked so
    # STATE="validated" can eventually be checked against it. This is
    # purely additive evidence production -- it does not gate anything in
    # this campaign, and nothing in bigcherry apply/build consumes it yet
    # (see plan item HI83's notes for why hard enforcement is deliberately
    # deferred). A campaign with no correctness evidence and/or no
    # activation probe for this patch still writes a real record; it is
    # simply not eligible_for_validated_state.
    from bigcherry.patch import evidence as patch_validation_evidence  # noqa: E402

    # cfg is loaded once, above, before source resolution -- reused here
    # (was previously loaded a second time in this exact spot, after
    # source materialization had already resolved against "HEAD").
    # RS04: the evidence record's patch file path resolves through the
    # registry descriptor (flat or packaged) -- no f"{patch_id}.py" guessing
    # in this caller.
    _descriptor = descriptor
    _patch_file = registry.root / _descriptor.implementation_path
    campaign_run_dir = workdir / "campaign"
    correctness_summary = None
    # VA11A: ctx.correctness_evidence must carry a BOUND artifact reference
    # ({"artifact": {"path", "sha256"}}) -- _builtin_backend_ops reads via
    # ctx.correctness_evidence.get("artifact"), and a raw decoded dict with
    # no "artifact" key made every backend-ops check unconditionally BLOCKED
    # (real bug, confirmed by reading validation.py::_builtin_backend_ops
    # before this fix -- GPT round-7 review, req_3d12aa6668b14bb1).
    correctness_evidence: dict[str, object] = {}
    performance_evidence: dict[str, object] = {}
    if args.correctness_evidence is not None:
        correctness_summary = patch_validation_evidence.load_correctness_summary(
            args.correctness_evidence,
            patch_id=args.patch,
            subject_digest=patch_validation_evidence.patch_validation_subject_digest(
                _patch_file
            ),
            base_revision=base_revision,
            patched_source_tree=patched_source_tree,
            campaign_identity_digest=campaign.campaign_identity_digest,
            gpu_architectures=(args.amdgpu_targets,),
        )
        correctness_path = campaign_run_dir / "correctness.json"
        _atomic_write_json(correctness_path, correctness_summary)
        correctness_evidence = {
            "artifact": {
                "path": correctness_path.relative_to(campaign_run_dir).as_posix(),
                "sha256": hashlib.sha256(correctness_path.read_bytes()).hexdigest(),
            }
        }
    build_evidence = {
        "control": {
            "build_id": control_build_evidence.effective_build_id,
            "source_tree": control_source_tree,
            "architecture": args.amdgpu_targets,
            "options": control_build_evidence.effective_configure,
            "compile_commands": _write_bound_artifact(
                campaign_run_dir,
                "build/control-compile-commands.json",
                control_build_evidence.verification.to_dict(),
            ),
            "runtime_bundle": _write_bound_artifact(
                campaign_run_dir,
                "build/control-runtime-bundle.json",
                control_build_evidence.runtime_artifacts,
            ),
        },
        "subject": {
            "build_id": validation_subject_build_evidence.effective_build_id,
            "source_tree": patched_source_tree,
            "architecture": args.amdgpu_targets,
            "options": validation_subject_build_evidence.effective_configure,
            "compile_commands": _write_bound_artifact(
                campaign_run_dir,
                "build/subject-compile-commands.json",
                validation_subject_build_evidence.verification.to_dict(),
            ),
            "runtime_bundle": _write_bound_artifact(
                campaign_run_dir,
                "build/subject-runtime-bundle.json",
                validation_subject_build_evidence.runtime_artifacts,
            ),
        },
    }
    apply_evidence = {
        "control": {
            "verified": True,
            "idempotent": control_idempotent,
            "artifact": _write_bound_artifact(
                campaign_run_dir,
                "apply/control.json",
                {
                    "source_tree": control_source_tree,
                    "composition": list(control_composition),
                },
            ),
        },
        "subject": {
            "verified": True,
            "idempotent": subject_idempotent,
            "artifact": _write_bound_artifact(
                campaign_run_dir,
                "apply/subject.json",
                {
                    "source_tree": patched_source_tree,
                    "composition": list(subject_composition),
                },
            ),
        },
    }
    st._descriptor = _descriptor
    st._patch_file = _patch_file
    st.apply_evidence = apply_evidence
    st.build_evidence = build_evidence
    st.campaign_run_dir = campaign_run_dir
    st.correctness_evidence = correctness_evidence
    st.correctness_summary = correctness_summary
    st.patch_validation_evidence = patch_validation_evidence
    st.performance_evidence = performance_evidence


def _run_contract_evidence_modes(args: argparse.Namespace, st: SimpleNamespace) -> None:
    """PA43 run() stage. Contract-evidence stage: the specialized RD08/RD73 evidence-producer modes."""
    _patch_file = st._patch_file
    activation_evidence = st.activation_evidence
    activation_verdict = st.activation_verdict
    base_revision = st.base_revision
    campaign = st.campaign
    campaign_run_dir = st.campaign_run_dir
    control_bin = st.control_bin
    correctness_evidence = st.correctness_evidence
    correctness_summary = st.correctness_summary
    descriptor = st.descriptor
    exe = st.exe
    patch_validation_evidence = st.patch_validation_evidence
    patched_source_tree = st.patched_source_tree
    performance_evidence = st.performance_evidence
    trace_evidence = st.trace_evidence
    trace_marker_regex = st.trace_marker_regex
    validation_subject_bin = st.validation_subject_bin
    rd73_qualification = None
    # VA14-B/VA14-final: RD08 execution, opt-in and scoped to RD08 only.
    # --run-rd08-lanes stays diagnostic-only (execution + evidence, never
    # feeds eligibility). --run-rd08-contract is the authoritative full-
    # qualification path (lanes + real named correctness + real trigger
    # proof, composed via evaluate_promotion_gate()) and is the ONLY thing
    # allowed to populate contract_promotions below. The two are mutually
    # exclusive to avoid a redundant duplicate lane run.
    if args.run_rd08_lanes and args.run_rd08_contract:
        raise PatchCampaignError(
            f"{args.patch}: --run-rd08-contract already runs the lanes -- "
            "do not also pass --run-rd08-lanes"
        )
    if args.run_rd73_contract:
        if args.run_rd08_lanes or args.run_rd08_contract:
            raise PatchCampaignError(
                f"{args.patch}: --run-rd73-contract is mutually exclusive with the "
                "other specialized evidence-producer modes"
            )
        if descriptor.experiment_contract != "RD73-STABLE-GRAPH-CACHE-KEY":
            raise PatchCampaignError(
                f"{args.patch}: --run-rd73-contract is RD73-only today"
            )
        if args.rd73_corpus is None:
            raise PatchCampaignError(
                f"{args.patch}: --run-rd73-contract requires --rd73-corpus"
            )
        from bigcherry.patch import validation as _pv

        rd73_contract = _pv.load_contract_for_descriptor(descriptor)
        if rd73_contract is None:
            raise PatchCampaignError(
                f"{args.patch}: --run-rd73-contract requires a resolvable RD73 contract"
            )
        rd73_qualification = run_rd73_contract_qualification(
            contract=rd73_contract,
            control_server_binary=control_bin / f"llama-server{exe}",
            subject_server_binary=validation_subject_bin / f"llama-server{exe}",
            model=args.model,
            marker_regex=trace_marker_regex,
            corpus_path=args.rd73_corpus,
            run_dir=campaign_run_dir,
            # RV99: every measurement session already committed for this
            # patch. Under a session policy the gate aggregates these together
            # with the session about to be measured, so a run can establish a
            # bound that no single run could. Read from the tracked evidence
            # file -- which is exactly why lane_effects had to be persisted
            # there, and why RV96 had to make a build hold more than one
            # record before any of this could work.
            prior_session_records=patch_validation_evidence.load_records(args.patch),
            amdgpu_targets=args.amdgpu_targets,
        )
        contract_promotions[rd73_contract.id] = rd73_qualification["promotion"]

        # VA23: bind RD73's real contract-produced evidence into the generic
        # adapter, exactly as RD58 does. Before this, the RD73 branch produced
        # authoritative artifacts but bound none of them, so the declared
        # performance/controls checks ERRORed ("benchmark artifact requires
        # non-empty metrics") and correctness stayed BLOCKED -- making
        # patch-verify-evidence report that no benchmark ran when one had.
        #
        # correctness is bound ONLY when the bit_identical evaluation actually
        # produced an artifact. On Rd73CorrectnessError the artifact is None
        # and correctness must stay BLOCKED rather than silently pass: a
        # correctness check that could not be evaluated is not a correctness
        # check that succeeded.
        performance_evidence = {"artifact": rd73_qualification["performance_artifact"]}
        if rd73_qualification["correctness"].get("artifact") is not None:
            correctness_evidence = {
                "artifact": rd73_qualification["correctness"]["artifact"]
            }
        # VA23: the activation lane already ran a real positive/negative
        # marker probe; bind its bound log refs so _builtin_trace_marker()
        # can re-read and re-verify them. The validator does its own regex
        # check against both logs, so this supplies evidence for independent
        # verification rather than asserting the outcome.
        trace_evidence = {
            "positive": rd73_qualification["activation"]["positive"],
            "negative": rd73_qualification["activation"]["negative"],
        }
        # RV95: the three bindings above satisfy validation.toml's DECLARED
        # checks, but not the record's own top-level activation/correctness
        # fields -- make_record() reads those from activation_evidence and
        # correctness_summary, which the RD73 branch never set. They stayed
        # at disposition="unknown", so verify_validated_patch() rejected an
        # otherwise-passing record with "activation is not executed+
        # activation-verified; correctness did not pass" even while
        # check_results._contract_correctness_gate.passed was true. Bind them
        # from the SAME real evidence RD08/RD58 use, in the same shape.
        activation_evidence = ActivationEvidence(
            status=(
                "executed"
                if rd73_qualification["activation"]["subject_hit"]
                and not rd73_qualification["activation"]["control_hit"]
                else "not_executed"
            ),
            mechanism="rd73-trigger-marker",
            detail=f"marker={trace_marker_regex!r}",
        )
        activation_verdict = verdict(activation_evidence, correctness_passed=None)
        write_activation_json(
            campaign_run_dir / "activation.json",
            activation_evidence,
            activation_verdict,
            extra={
                "campaign_identity_digest": campaign.campaign_identity_digest,
                "rd73_trigger": {
                    "subject_hit": rd73_qualification["activation"]["subject_hit"],
                    "control_hit": rd73_qualification["activation"]["control_hit"],
                    "artifact": rd73_qualification["activation"]["artifact"],
                },
            },
        )
        # Disposition comes from the contract's own correctness gate, which
        # is already fail-closed: an Rd73CorrectnessError leaves the artifact
        # None and records passed=False, so a correctness check that could
        # not be evaluated reports "failed" here rather than silently passing.
        correctness_summary = {
            "schema_version": patch_validation_evidence.CORRECTNESS_SCHEMA_VERSION,
            "patch_id": args.patch,
            "patch_validation_subject_digest": patch_validation_evidence.patch_validation_subject_digest(
                _patch_file
            ),
            "base_revision": base_revision,
            "patched_source_tree": patched_source_tree,
            "campaign_identity_digest": campaign.campaign_identity_digest,
            "gpu_architectures": [args.amdgpu_targets],
            "disposition": (
                "passed"
                if rd73_qualification["correctness_gate"].get("passed")
                else "failed"
            ),
            "mechanism": "rd73-mtp-bit-identical",
            "detail": (
                "paired MTP control/subject completions compared byte-for-byte; "
                f"{len(rd73_qualification['correctness'].get('rows') or ())} row(s) compared"
            ),
        }
        correctness_path = campaign_run_dir / "correctness.json"
        _atomic_write_json(correctness_path, correctness_summary)

        _print(f"rd73 contract qualification: {rd73_qualification['artifact']['path']}")
        _print(
            f"rd73 promotion: "
            f"{'PASS' if rd73_qualification['promotion'].get('passed') else rd73_qualification['promotion'].get('status', 'FAIL')}"
        )
    st.activation_evidence = activation_evidence
    st.activation_verdict = activation_verdict
    st.correctness_evidence = correctness_evidence
    st.correctness_summary = correctness_summary
    st.performance_evidence = performance_evidence
    st.rd73_qualification = rd73_qualification
    st.trace_evidence = trace_evidence


def _evaluate_validation_plan(args: argparse.Namespace, st: SimpleNamespace) -> None:
    """PA43 run() stage. Evaluate the declared validation plan and the contract correctness gate."""
    apply_evidence = st.apply_evidence
    base_revision = st.base_revision
    build_evidence = st.build_evidence
    campaign_run_dir = st.campaign_run_dir
    control_build_evidence = st.control_build_evidence
    control_source_tree = st.control_source_tree
    control_src = st.control_src
    correctness_evidence = st.correctness_evidence
    descriptor = st.descriptor
    patch_validation = st.patch_validation
    patched_source_tree = st.patched_source_tree
    patched_src = st.patched_src
    performance_evidence = st.performance_evidence
    rd73_qualification = st.rd73_qualification
    registry = st.registry
    stock_src = st.stock_src
    trace_evidence = st.trace_evidence
    validation_plan = st.validation_plan
    validation_subject_build_evidence = st.validation_subject_build_evidence
    validation_check_results: dict[str, object] = {}
    validation_verdict = None
    if validation_plan is not None:
        # VA11A: package_root lets a packaged patch's custom validator
        # actually resolve its check(ctx) file (was always None -- any
        # custom check would fail closed for every packaged RD patch).
        package_root = (
            (registry.root / descriptor.package_root)
            if descriptor.package_root is not None
            else {}
        )
        # VA15 real-hardware finding: validation_plan.contract is a
        # patch_validation.ContractBinding -- a lightweight PROJECTION
        # (contract_id/hash/expected_effect/etc) that deliberately does NOT
        # carry .correctness/.acceptance/etc. compute_contract_correctness_gate()
        # needs the real experiment_contract.ExperimentContract, which
        # run_rd08_contract already loaded as rd08_contract for the
        # --run-rd08-contract path; other contract-bound patches load it
        # fresh here the same way that block does. PA36-F step 2: also the
        # source of ValidationContext's plural contracts/contract_hashes
        # below -- loaded once, before the context is constructed.
        full_contract = (
            rd08_contract
            if rd08_qualification is not None
            else patch_validation.load_contract_for_descriptor(descriptor)
        )
        validation_ctx = patch_validation.ValidationContext(
            descriptor=descriptor,
            base_revision=base_revision,
            control_source=control_src,
            subject_source=patched_src,
            stock_source=stock_src,
            package_root=package_root,
            control_tree=control_source_tree,
            subject_tree=patched_source_tree,
            # GPT round 3: RD58's real build check must be evaluated
            # against ITS test-save-load-state builds, not the generic
            # llama-bench builds the final record no longer identifies it
            # with.
            build_identities={
                "control": control_build_evidence.effective_build_id,
                "subject": validation_subject_build_evidence.effective_build_id,
            },
            build_evidence=build_evidence,
            apply_evidence=apply_evidence,
            architecture=args.amdgpu_targets,
            model=str(args.model),
            contracts=(full_contract,) if full_contract is not None else (),
            contract_hashes=(
                {full_contract.id: full_contract.contract_hash}
                if full_contract is not None
                else {}
            ),
            run_dir=campaign_run_dir,
            register_artifact=patch_validation.make_default_register_artifact(
                campaign_run_dir
            ),
            trace_evidence=trace_evidence,
            correctness_evidence=correctness_evidence,
            performance_evidence=performance_evidence,
        )
        evaluated = {
            spec.check_id: patch_validation.evaluate_check(spec, validation_ctx)
            for spec in validation_plan.checks
        }
        validation_verdict = patch_validation.compute_verdict(
            validation_plan, evaluated
        )
        # GPT round 2 (req_3616cc1d90dc4512, blocker #3): RD58's own real
        # test-save-load-state evidence produces a named
        # state_restore_integrity CorrectnessResult -- thread it through
        # here so the contract's own correctness gate actually reflects
        # the real evidence instead of reporting missing_checks.
        # VA23: the legacy run() path no longer produces named
        # correctness results (RD73's bit_identical named results were
        # retired with the RD73 legacy compatibility retirement; RD08's
        # flow through the generic producer path). An empty mapping
        # preserves the gate's missing_checks fail-closed behavior for any
        # bound contract that requires a named check.
        contract_correctness_gate = compute_contract_correctness_gate(
            full_contract,
            (
                rd08_qualification["correctness"]["results"]
                if rd08_qualification is not None
                # VA23: RD73's bit_identical result is real and already
                # evaluated inside run_rd73_contract_qualification(); thread
                # it here exactly as RD08's, so the gate
                # reflects the evidence instead of reporting missing_checks.
                else (
                    rd73_qualification["correctness_named_results"]
                    if rd73_qualification is not None
                    else {}
                )
            ),
        )
        validation_check_results = {
            check_id: asdict(result) for check_id, result in evaluated.items()
        }
        if contract_correctness_gate is not None:
            validation_check_results["_contract_correctness_gate"] = (
                contract_correctness_gate
            )
            _print(
                f"contract correctness gate: "
                f"{'passed' if contract_correctness_gate.get('passed') else contract_correctness_gate.get('status', 'not passed')}"
            )
        _print(
            f"validation verdict: {'eligible' if validation_verdict.eligible else 'ineligible'} "
            f"({len(validation_verdict.reasons)} blocking reasons)"
        )
    st.validation_check_results = validation_check_results
    st.validation_verdict = validation_verdict


def _persist_validation_record(args: argparse.Namespace, st: SimpleNamespace) -> int:
    """PA43 run() stage. Record-persistence stage: write the HI83 validation evidence record."""
    _descriptor = st._descriptor
    _patch_file = st._patch_file
    activation_evidence = st.activation_evidence
    activation_verdict = st.activation_verdict
    base_revision = st.base_revision
    baseline_source = st.baseline_source
    campaign = st.campaign
    cfg = st.cfg
    control_build_evidence = st.control_build_evidence
    control_composition = st.control_composition
    control_source_tree = st.control_source_tree
    correctness_summary = st.correctness_summary
    identity_context = st.identity_context
    patch_digest = st.patch_digest
    patch_validation_evidence = st.patch_validation_evidence
    patched_source_tree = st.patched_source_tree
    psi = st.psi
    rd73_qualification = st.rd73_qualification
    stock_src = st.stock_src
    subject_composition = st.subject_composition
    validation_check_results = st.validation_check_results
    validation_plan = st.validation_plan
    validation_subject_build_evidence = st.validation_subject_build_evidence
    validation_verdict = st.validation_verdict
    workdir = st.workdir
    validation_contracts, validation_contract_verdicts = (
        build_contract_evidence_for_persistence(
            validation_plan.contracts if validation_plan is not None else (),
            contract_promotions,
        )
    )

    validation_record = patch_validation_evidence.make_record(
        patch_id=args.patch,
        patch_path=_patch_file,
        patch_implementation_digest=patch_digest,
        base_ref=cfg.pinned,
        base_revision=base_revision,
        framework_baseline_digest=psi.composition_digest(subject_composition),
        patched_source_tree=patched_source_tree,
        gpu_architectures=args.amdgpu_targets,
        activation_evidence=activation_evidence,
        activation_disposition=activation_verdict,
        correctness=correctness_summary,
        campaign_identity_digest=campaign.campaign_identity_digest,
        build_identities=identity_context.build_identities,
        # VA07: real validation-build domain, distinct from the campaign
        # {tune,replay,stock} domain above. subject is intentionally the
        # same physical build as campaign.tune today (the tune build IS
        # the patch under validation) -- the schema records both roles
        # explicitly rather than assuming that equality.
        # GPT round 2 (blocker #1): RD58's real validation build is
        # test-save-load-state, not the generic llama-bench control/
        # validation-subject builds -- record ITS identities when RD58 ran.
        validation_build_identities={
            "control": control_build_evidence.campaign_identity(),
            "subject": validation_subject_build_evidence.campaign_identity(),
        },
        campaign_workdir=workdir / "campaign",
        check_results=validation_check_results,
        # VA14 final slice: eligible_for_validated_state for a bound-contract
        # patch requires BOTH the adapter verdict AND every bound contract's
        # own evaluate_promotion_gate() PASS (contract_promotions). A bound
        # contract with no promotion result at all still forces False.
        # RV95: it additionally requires the record's own activation/
        # correctness dispositions -- the same values make_record() persists
        # just below and verify_validated_patch() later reads -- so this flag
        # can no longer report eligible for a record the evidence verifier
        # rejects. See compute_persisted_validation_eligible()'s docstring.
        validation_eligible=compute_persisted_validation_eligible(
            _descriptor,
            validation_verdict,
            contract_promotions,
            activation_disposition=activation_verdict,
            correctness=correctness_summary,
        ),
        # RV99: persist the measurements, not only the verdict derived from
        # them, so an interval can be re-derived and sessions aggregated from
        # committed evidence alone.
        lane_effects=collect_lane_effect_records(
            rd08_qualification=rd08_qualification,
            rd73_qualification=rd73_qualification,
        ),
        representation=_descriptor.representation,
        validation_implementation_digest=_descriptor.validation_digest,
        contracts=validation_contracts,
        contract_verdicts=validation_contract_verdicts,
        baseline_composition={
            "source": baseline_source,
            "base_revision": base_revision,
            "patches": list(control_composition),
        },
        control_composition={
            "base_revision": base_revision,
            "patches": list(control_composition),
        },
        subject_composition={
            "base_revision": base_revision,
            "patches": list(subject_composition),
        },
        control_tree=control_source_tree,
        subject_tree=patched_source_tree,
        stock_tree=psi.git_worktree_tree(stock_src),
    )
    validation_record_path = patch_validation_evidence.write_record(validation_record)
    _print(f"validation evidence: {validation_record_path}")
    _print(
        "STATE='validated' eligible: "
        + ("yes" if validation_record["eligible_for_validated_state"] else "no")
    )

    return 0


def run(args: argparse.Namespace) -> int:
    import os

    workdir: Path = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)

    # e2e_smoke_campaign.Campaign launches llama-server via `dict(os.environ)`
    # (this process's own environment), not through _hip_env() -- that helper
    # only covers the cmake configure/build subprocesses above. Without the
    # ROCm bin dir on PATH here, the HIP runtime DLLs are unresolved at
    # process launch (Windows exit code 0xC0000135 / STATUS_DLL_NOT_FOUND --
    # hit for real running this tool headless/backgrounded, where no
    # interactive shell had already sourced tools/rocm-env.ps1|.sh).
    os.environ["ROCM_PATH"] = str(args.hip_path)
    os.environ["HIP_PATH"] = str(args.hip_path)
    os.environ["PATH"] = os.pathsep.join(
        [str(args.hip_path / "bin"), os.environ.get("PATH", "")]
    )

    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from bigcherry.patch import source as psi  # noqa: E402
    from bigcherry.patch import (
        registry as patch_registry,
        validation as patch_validation,
    )
    from bigcherry.patch import validation_policy as patch_validation_policy  # noqa: E402
    from bigcherry.core import paths as bc_paths  # noqa: E402
    from bigcherry.core import config as campaign_config  # noqa: E402

    registry = patch_registry.load_registry(bc_paths.PATCHES)
    descriptor = registry.get(args.patch)

    # GPT round 2 (req_71217bba406f4941, VA04 real-hardware finding): the
    # pinned ref MUST be resolved before any source materialization --
    # the hardcoded literal "HEAD" below used to silently build against whatever
    # the shared vendor/llama.cpp checkout's HEAD happened to be at run
    # time, while the evidence record was later labeled base_ref=cfg.pinned
    # regardless of whether HEAD actually matched the pin. A real RD04
    # hardware run on Brutus resolved and built against vendor HEAD while
    # its own evidence claimed pin b10705 -- VA08's stale-detection
    # correctly caught the mismatch and rejected the record. cfg is loaded
    # ONCE here and reused for evidence writing below (no duplicate load).
    cfg = campaign_config.load(bc_paths.RECIPES)

    # VA02 execution-side anti-grandfather guard (unconditional, per GPT
    # round-5 code review req_86cfd3a0bff04716: this command IS "start a
    # real validation run" -- there is no tracked-status branch here,
    # because otherwise build_plan_for_patch() legitimately returning None
    # for a patch with neither a contract nor an adapter would let this
    # command continue straight into source materialization/build without
    # ever producing real evidence tied to a check, regardless of any
    # lint-side structural-grandfather exemption).
    validation_plan = patch_validation_policy.require_execution_package(
        descriptor,
        root=bc_paths.PATCHES,
    )
    if validation_plan is not None:
        _print(
            f"validation plan: {len(validation_plan.checks)} checks; required={validation_plan.required_capabilities}"
        )

    if getattr(args, "framework_configuration", False):
        return _run_framework_configuration(args, descriptor, cfg)

    if getattr(args, "run_performance_benchmark", False):
        return _run_performance_benchmark(args, descriptor, cfg)

    st = SimpleNamespace(
        cfg=cfg,
        descriptor=descriptor,
        patch_validation=patch_validation,
        psi=psi,
        registry=registry,
        validation_plan=validation_plan,
        workdir=workdir,
    )
    _prepare_standard_campaign(args, st)
    _run_activation_probe_stage(args, st)
    CampaignError = st.CampaignError
    campaign = st.campaign

    # GPT round 6 (req_bc329f6ae30c4e4c, VA15 real-hardware finding): the
    # generic S1-S7 record/tune/promote/replay/bench/report campaign is
    # unrelated to a contract's own evidence -- lanes/correctness/
    # trigger/promotion never consume promoted.jsonl, dispatch.cache,
    # replay coverage, or S6/S7 results. Making that unrelated pipeline's
    # own promotion decision (which can legitimately promote zero
    # candidates on a real, honest run -- that is not a bug) a hard
    # prerequisite of a contract run was itself the real bug,
    # discovered on real hardware (VA15). campaign.ensure_campaign_identity()
    # above still ran, so campaign.campaign_identity_digest remains valid
    # for the contract evidence below.
    # (The historical --run-rd73-contract S1-S7 pipeline skip was retired
    # with the RD73 legacy compatibility retirement; RD73's real evidence
    # now comes from its producer via the generic path. The generic
    # S1-S7 pipeline therefore always runs on the legacy run() path.)
    try:
        campaign.run()
    except CampaignError as exc:
        _print(f"CAMPAIGN FAILED: {exc}")
        return 1

    report_path = workdir / "campaign" / "report.md"
    _print(f"done -- report: {report_path}")
    print(report_path.read_text(encoding="utf-8"))

    _collect_build_and_correctness_evidence(args, st)
    _run_contract_evidence_modes(args, st)
    _evaluate_validation_plan(args, st)
    return _persist_validation_record(args, st)


def _absolute_path(value: str) -> Path:
    """argparse type: resolve a path argument to absolute at parse time.

    Source worktrees are created with ``git -C <vendor repo> worktree add
    <path>``, which interprets a relative path against the OTHER repository
    and fails (exit 128) -- so --workdir/--build-root/--worktree-root never
    reach that call relative."""
    return Path(value).resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bigcherry patch-validation-campaign")
    parser.add_argument(
        "--patch", required=True, help="patch module name under patches/"
    )
    parser.add_argument(
        "--framework-configuration",
        action="store_true",
        default=False,
        help="run the explicit schema-5 local framework-configuration build path",
    )
    parser.add_argument(
        "--baseline-source",
        default="bigcherry",
        help="explicit named source composition for CONTROL; SUBJECT adds "
        "only the focal patch. The focal must be absent from this "
        "baseline; dependencies/conflicts remain enforced.",
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--hip-path", required=True, type=Path)
    parser.add_argument(
        "--amdgpu-targets",
        default=None,
        help="e.g. gfx1100 or gfx1201 -- required for every mode EXCEPT "
        "--run-performance-benchmark, which resolves its own architecture list "
        "(--benchmark-architecture or the recipes/validation-architectures "
        "intersection) per cell.",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--workdir",
        required=True,
        type=_absolute_path,
        help="per-run campaign output (record/tune/promote/replay/bench/report)",
    )
    parser.add_argument(
        "--build-root",
        type=_absolute_path,
        default=None,
        help="shared build-tree location (tune/replay/stock), reused across "
        "multiple patch+model runs on this machine+arch; defaults to "
        "--workdir (no reuse) if omitted",
    )
    parser.add_argument(
        "--worktree-root",
        type=_absolute_path,
        default=None,
        help="content-addressed isolated source worktrees "
        "(patch_source_isolation.py, HI82) live here, one per "
        "(base_revision, patch, framework-baseline) identity; defaults to "
        "<ProjectContext work root>/worktrees (project-local, never a user folder)",
    )
    parser.add_argument("--bench-prompt", type=int, default=512)
    parser.add_argument("--bench-gen", type=int, default=128)
    parser.add_argument("--bench-repetitions", type=int, default=5)
    parser.add_argument(
        "--trace-marker-regex",
        default=None,
        help="optional generic activation marker regex; patch-specific probe configuration "
        "stays outside the campaign orchestrator",
    )
    parser.add_argument(
        "--trace-description",
        default=None,
        help="human-readable description paired with --trace-marker-regex",
    )
    parser.add_argument(
        "--correctness-evidence",
        type=Path,
        default=None,
        help="HI83: machine-readable patch-level correctness evidence bound to "
        "this patch/source/campaign identity; without it the campaign still "
        "runs and records evidence, but the record is not eligible for "
        "STATE='validated'",
    )
    parser.add_argument(
        "--run-rd73-contract",
        action="store_true",
        default=False,
        help="VA06: the RD73 full-qualification path -- activation + graph-cache resource "
        "evidence + real paired MTP-verify performance (server harness) + decode "
        "control + bit-identical correctness, composed via evaluate_promotion_gate(). "
        "Populates contract_promotions for RD73 (a real PASS/FAIL/INVALID verdict is "
        "printed) and rebinds the generic adapter's performance/correctness/trace "
        "evidence plus the record's own activation/correctness dispositions, so a "
        "passing run satisfies verify_validated_patch() as well as the eligibility "
        "flag. RD73-only; an error for any other patch. Mutually exclusive with the "
        "RD08/RD58 execution modes. Requires --rd73-corpus.",
    )
    parser.add_argument(
        "--producer-corpus",
        type=Path,
        default=None,
        help="text corpus for --validation-producer's ProducerContext.corpus "
        "-- the generic (non-RD73-specific) analog of --rd73-corpus, "
        "threaded into any patch-local producer that "
        "declares a correctness check requiring a real backend-reference "
        "corpus (e.g. 1203's RD05/RD07 backend_reference checks).",
    )
    parser.add_argument(
        "--rd73-corpus",
        type=Path,
        default=None,
        help="VA06: prompt corpus JSONL for --run-rd73-contract's MTP server lane "
        "(bench/server_completion.py's load_corpus() format).",
    )
    parser.add_argument(
        "--run-performance-benchmark",
        action="store_true",
        default=False,
        help="PVPS02: the generic paired-benchmark entry point for ANY patch whose "
        "validation.toml wires a recognized benchmark-executor on its required "
        "performance check -- not RD04/RD08/etc-specific. Builds one control/subject "
        "llama-bench per applicable architecture and runs the standard (or "
        "--benchmark-model-selected) model matrix across them. Diagnostic-only for "
        "eligibility, same as the legacy per-patch modes -- never populates "
        "contract_promotions. Mutually exclusive with the RD04/RD08/RD58/RD73 modes; "
        "does NOT require --model/--manifest/--amdgpu-targets (those are for the "
        "legacy single-architecture flow).",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=None,
        help="PVPS02: host model root config/models.toml paths are relative to (e.g. "
        "$BC_MODEL_ROOT). Required with --run-performance-benchmark.",
    )
    parser.add_argument(
        "--benchmark-model",
        action="append",
        default=None,
        help="PVPS02: a config/models.toml id to benchmark (repeatable). Omit for the "
        "standard model set (tierM-ministral14b-q4km, tierB-qwen9b-q6k, "
        "tierL-qwen27b-q8).",
    )
    parser.add_argument(
        "--benchmark-architecture",
        action="append",
        default=None,
        help="PVPS02: an amdgpu target to benchmark (repeatable). Omit to default to the "
        "intersection of config/recipes.toml's platform.linux-multi targets and the "
        "patch's own validation-architectures.",
    )
    parser.add_argument(
        "--device-map",
        action="append",
        default=None,
        help="PVPS02: ARCH=ID[,ID...] (repeatable) -- the real, ordered device pool for "
        "one architecture. Required with --run-performance-benchmark; never inferred. "
        "Also consumed by --validation-producer (device indices there must be integers).",
    )
    parser.add_argument(
        "--validation-producer",
        dest="validation_producer",
        metavar="PATCH/PRODUCER_ID",
        default=None,
        help="PA36-F step 5: select one patch-local validation producer "
        "(patches/<patch>/validation/producer.toml's [producer.<PRODUCER_ID>]) "
        "and execute it through the generic execute_validation_producer() "
        "dispatcher. Mutually exclusive with every --run-rdXX-*/--run-patchXXXX-* "
        "legacy execution mode -- this is the non-legacy replacement path "
        "(PA36's atomic migration sequence retires the legacy flags one at a "
        "time). Repeatable --producer-input NAME=VALUE supplies its declared "
        "inputs.",
    )
    parser.add_argument(
        "--producer-input",
        dest="producer_inputs",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="PA36-F step 5: one producer-declared input (repeatable). Fails closed "
        "if the producer does not declare NAME, a required NAME is missing, or "
        "the same NAME is given twice.",
    )
    args = parser.parse_args(argv)
    if args.worktree_root is None:
        args.worktree_root = ProjectContext.resolve().work_root / "worktrees"
    if args.validation_producer is not None:
        # PA36-F step 5, GPT design section 5 (req_8ec9b90c05f84a30): generic
        # dispatch plugs in immediately after parse_args()/common patch
        # resolution, before the first other RD-only
        # guard. Reject generically by NAME PATTERN, never a hardcoded tuple
        # of known RD flags -- a new --run-rdNN-* flag added later is caught
        # automatically, with no edit required here.
        legacy_modes = tuple(
            name
            for name, value in vars(args).items()
            if value
            and (
                re.fullmatch(r"run_rd\d+.*", name)
                or re.fullmatch(r"run_patch\d+.*", name)
            )
        )
        if legacy_modes:
            parser.error(
                "--validation-producer is mutually exclusive with legacy execution "
                f"mode(s): {', '.join(sorted(legacy_modes))}"
            )
        selector_patch, producer_id = _parse_validation_producer_selector(
            args.validation_producer
        )
        # --patch stays required at the parser level (retiring that
        # requirement is the atomic migration sequence's job, not step 5's);
        # while it is, this just enforces it can never silently diverge from
        # the selector instead of asking the user to specify the patch twice.
        if args.patch != selector_patch:
            parser.error(
                f"--patch {args.patch!r} does not match --validation-producer's patch "
                f"component {selector_patch!r} -- do not specify a different patch twice"
            )
        provided_inputs = _parse_producer_inputs(args.producer_inputs)
        return _run_validation_producer(
            args, producer_id=producer_id, provided_inputs=provided_inputs
        )
    if (
        not args.framework_configuration
        and not args.run_performance_benchmark
        and (args.model is None or args.manifest is None or args.amdgpu_targets is None)
    ):
        parser.error(
            "runtime qualification requires --model, --manifest, and --amdgpu-targets"
        )
    if args.run_performance_benchmark:
        if args.model_root is None or not args.device_map:
            parser.error(
                "--run-performance-benchmark requires --model-root and --device-map"
            )
        if any(
            getattr(args, name, False)
            for name in (
                "run_rd08_lanes",
                "run_rd08_contract",
                "run_rd73_contract",
            )
        ):
            parser.error(
                "--run-performance-benchmark is mutually exclusive with the legacy RD modes"
            )
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

